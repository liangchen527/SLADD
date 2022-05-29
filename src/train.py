"""
eval pretained model.
"""
import os
import argparse
import numpy as np
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'


import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from dataprocess.FFdata import FaceForensicsDataset
from trainer.whole_model import Model
from utils.metrics import Metrics
import torch.multiprocessing as mp
import torch.distributed as dist

dset = ['FF-DF', 'FF-NT', 'FF-F2F', 'FF-FS', 'ALL']
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3,4,5,6,7'
parser = argparse.ArgumentParser()

# model setting
parser.add_argument('--meta', default="init",
                    type=str, help='the feature space')

# dataset
parser.add_argument('--dset', type=str, choices=dset,
                    help='method in FF++')
parser.add_argument('--train_batchSize', type=int,
                    default=32, help='input batch size')
parser.add_argument('--eval_batchSize', type=int,
                    default=32, help='eval batch size')
parser.add_argument('--workers', type=int,
                    help='number of data loading workers', default=8)
parser.add_argument('--resolution', type=int, default=256,
                    help='the resolution of the output image to network')
parser.add_argument('--test_batchSize', type=int,
                    default=32, help='test batch size')
parser.add_argument('--dataname', type=str, help='dataname')

# setting
parser.add_argument('--save_epoch', type=int, default=1,
                    help='the interval epochs for saving models')
parser.add_argument('--rec_iter', type=int, default=100,
                    help='the interval iterations for recording')

# trainning config
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument("--momentum", default=0.9, type=float,
                    help="Momentum, Default: 0.9")
parser.add_argument("--weight_decay", default=0.0005, type=float,
                    help="Momentum, Default: 0.0005")

parser.add_argument("--nEpochs", type=int, default=30,
                    help="number of epochs to train for")
parser.add_argument("--start_epoch", default=0, type=int,
                    help="Manual epoch number (useful on restarts)")

# for distributed parallel 
parser.add_argument('-n', '--nodes', default=1,
                    type=int, metavar='N')
parser.add_argument('-g', '--gpus', default=1, type=int,
                    help='number of gpus per node')
parser.add_argument('-nr', '--nr', default=0, type=int,
                    help='ranking within the nodes')
parser.add_argument('-mp', '--masterport', default='8888', type=str,
                    help='ranking within the nodes')
parser.add_argument('--ngpu', type=int, default=8,
                    help='number of GPUs to use')

parser.add_argument('--logdir', default='./logs',
                    help='folder to output images')
parser.add_argument('--savedir', default='./saved',
                    help='folder to output images')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument("--pretrained", default=None, type=str,
                    help="path to pretrained model (default: none)")
parser.add_argument('--cuda',default=True, action='store_true', help='enable cuda')




def get_accracy(output, label):
    _, prediction = torch.max(output, 1)    # argmax
    correct = (prediction == label).sum().item()
    accuracy = correct / prediction.size(0)

    return accuracy

def get_prediction(output, label):
    prob = nn.functional.softmax(output, dim=1)[:, 1]
    prob = prob.view(prob.size(0), 1)
    label = label.view(label.size(0), 1)
    datas = torch.cat((prob, label.float()), dim=1)
    return datas


def test_epoch(model, test_data_loaders, step):
    # --------------eval------------
    # print('eval mode: ', mode)
    model.setEval()

    def run(data_loader, name):
        statistic = None
        metric = Metrics()
        losses = []
        acces = []

        print('Len of data_loader=', len(data_loader))
        # test ckpt in whole test set
        for i, batch in tqdm(enumerate(data_loader)):
            data, label = batch
            if isinstance(data, list):
                data = data[0]
            img = data
            img, label = img.cuda(non_blocking=True), label.cuda(non_blocking=True)
            cls_score, loss = model.inference(img, label)

            tmp_data = get_prediction(cls_score.detach(), label)
            if statistic is not None:
                statistic = torch.cat((statistic, tmp_data), dim=0)
            else:
                statistic = tmp_data

            losses.append(loss.cpu().detach().numpy())
            acces.append(get_accracy(cls_score, label))
            metric.update(label.detach(), cls_score.detach())

            # print ('the {} th batch'.format(i) )
        model.update_tensorboard(None, step, acc=None, datas=statistic, name='test/{}'.format(name))

        avg_loss = np.mean(np.array(losses))
        info = "|Test Loss {:.4f}".format(avg_loss)
        mm = metric.get_mean_metrics()
        mm_str = ""
        mm_str += "\t|Acc {:.4f} (~{:.2f})".format(mm[0], mm[1])
        mm_str += "\t|AUC {:.4f} (~{:.2f})".format(mm[2], mm[3])
        mm_str += "\t|EER {:.4f} (~{:.2f})".format(mm[4], mm[5])
        mm_str += "\t|AP {:.4f} (~{:.2f})".format(mm[6], mm[7])
        info += mm_str
        print(info)
        metric.clear()

        return (mm[0], mm[2], mm[4], mm[6])

    keys = test_data_loaders.keys()
    datas = [{}, {}, {}, {}]
    for i, key in enumerate(keys):
        print('[{}/{}]Testing from {} ...'.format(i+1, len(keys), key))
        dataloader = test_data_loaders[key]
        ret = run(dataloader, key)
        for j, data in enumerate(ret):
            datas[j][key] = data
    model.update_tensorboard_test_accs(datas, step, feas=None)


def train_epoch(gpu, model, train_data_loader, epoch, cur_acc,
                eval_data_loader, savedir, test_data_loaders=None, 
                dataset='FF_DF'):

    if gpu== 0:
        print("===> Epoch[{}] start!".format(epoch))
        
    best_acc = cur_acc
    model.setTrain()
    # --------------train------------
    eval_step = len(train_data_loader) // 10    # eval 10 times per epoch
    step_cnt = epoch * len(train_data_loader)
    losses = []
    acces = []
    for iteration, batch in tqdm(enumerate(train_data_loader)):
        model.setTrain()
        data, label = batch
        img, fake_img, real_lmk, fake_lmk, real_mask, fake_mask = data
        img, fake_img, label, real_mask = img.cuda(non_blocking=True), fake_img.cuda(non_blocking=True), \
                              label.cuda(non_blocking=True), real_mask.cuda(non_blocking=True)
        ############
        with torch.autograd.set_detect_anomaly(True):
            label, ret = model.optimize(img, label, fake_img, real_lmk, fake_lmk, real_mask, fake_mask, epoch)
            cls_score, loss = ret
        losses.append(loss.cpu().detach().numpy())
        acces.append(get_accracy(cls_score, label),)

        if iteration % 500 == 0 and gpu == 0:
            info = "[{}/{}]\n".format(iteration, len(train_data_loader))
            avg_loss = np.mean(np.array(losses))
            info += "\tLoss Cls:{:.4f}\n".format(avg_loss)
            avg_acc = np.mean(np.array(acces))
            info += '\tAVG Acc\t{:.4f}'.format(avg_acc)
            acces.clear()
            losses.clear()
            model.update_tensorboard(avg_loss, step_cnt, acc=avg_acc, name='train')
        '''
            An oracle model selection is used:
            choose the model that performs the best on the test set instead of the evaluation set for evaluation.
        '''
        if (step_cnt+1) % eval_step == 0 and gpu == 0:
            if test_data_loaders is not None:
                test_epoch(model, test_data_loaders, step_cnt)

        step_cnt += 1
    if gpu == 0:
        model.save_ckpt(dataset, epoch, iteration, savedir, best=False)

    return best_acc



def train(gpu,args):
    print('start training')
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed) 
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.manualSeed)
    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    logdir = "{}/train".format(args.logdir)

    # ------------------- speed up
    model = Model(args, logdir=logdir, train=True)
    torch.cuda.set_device(gpu)
    model.model.cuda(gpu)
    model.synthesizer.cuda(gpu)
    # restore ckpts
    if args.pretrained is not None:
        model.load_ckpt(args.pretrained, 0)
    model.model = nn.parallel.DistributedDataParallel(model.model, device_ids=[gpu],
                                                      find_unused_parameters = True) # model to gpu
    model.synthesizer = nn.parallel.DistributedDataParallel(model.synthesizer, device_ids=[gpu],
                                                      find_unused_parameters = True) # model to gpu
    model.cls_criterion = model.cls_criterion.cuda(gpu) # criterion to gpu
    model.l1loss = model.l1loss.cuda(gpu)

    # -----------------load dataset--------------------------
    # -------------- use only FF++ train.json 
    train_set = FaceForensicsDataset(
              dataset=args.dset, mode='train', res=args.resolution, train=True)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_data_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batchSize,
                                                    shuffle=False, num_workers=0, pin_memory=True,
                                                    sampler=train_sampler)
    TESTLIST = {
        'FF-DF': "non-input",
        'FF-NT': "non-input",
        'FF-FS': "non-input",
        'FF-F2F': "non-input",
    }

    def get_data_loader(name):
        # -----------------load dataset--------------------------
        test_set = FaceForensicsDataset(dataset=name, mode='test', res=args.resolution, train=False)
        test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batchSize,
                                                       shuffle=False, num_workers=int(args.workers))
        return test_data_loader

    test_data_loaders = {}
    for list_key in TESTLIST.keys():
        test_data_loaders[list_key] = get_data_loader(list_key)

    # ----------------Train by epochs--------------------------
    best_acc = 0
    dataset = 'FF-DF'
    if gpu== 0:
        model.define_summary_writer()

    for epoch in range(args.start_epoch, args.nEpochs + 1):
        best_acc = train_epoch(gpu, model, train_data_loader, epoch, best_acc,
                               None, args.savedir, test_data_loaders=test_data_loaders,
                               dataset=dataset)
        print("===> Epoch[{}] end with acc {:.4f}!".format(epoch, best_acc))

    print("Stop Training on best validation accracy {:.4f}".format(best_acc))

    dist.destroy_process_group()

def main():
    import socket
    hostname = socket.gethostname()
    ip = socket.gethostbyname(hostname)
    opt = parser.parse_args()
    print(opt)
    opt.world_size = opt.gpus * opt.nodes                #
    os.environ['MASTER_ADDR'] = ip                       #
    os.environ['MASTER_PORT'] = opt.masterport           #
    mp.spawn(train, nprocs=opt.gpus, args=(opt,))        #

if __name__ == '__main__':
    main()
