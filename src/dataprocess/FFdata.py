import json
import numpy as np
import cv2
from PIL import Image
import torch
from dataprocess.utils.face_blend import *
from torch.utils import data
from torchvision import transforms as T
import dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./weights/shape_predictor_68_face_landmarks.dat')

def load_rgb(file_path, size=256):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img

def load_mask(file_path, size=256):
    mask = cv2.imread(file_path, 0)
    if mask is None:
        mask = np.zeros((size, size))

    mask = cv2.resize(mask, (size, size))/255
    return np.float32(mask)


class FaceForensicsDataset(data.Dataset):
    data_root = './data/FF/image/'
    mask_root = './data/FF/mask/'

    data_list = {
        'test': './data/FF/config/test.json',
        'train': './data/FF/config/train.json',
        'eval': './data/FF/config/eval.json'
    }

    # frames = {'test': 100, 'eval': 100, 'train': 270}
    frames = {'test': 25, 'eval': 10, 'train': 270}

    def __init__(self, dataset='FF-DF', mode='test', res=256, train=True,
                 sample_num=None):
        self.mode = mode
        self.dataset = dataset
        with open(self.data_list[mode], 'r') as fd:
            data = json.load(fd)
            img_lines = []
            fake_lines = []
            real_lines = []
            for pair in data:
                r1, r2 = pair
                step = 1
                for i in range(0, self.frames[mode], step):

                    img_lines.append(('{}/{}'.format('real', r1), i, 0))
                    img_lines.append(('{}/{}'.format('real', r2), i, 0))

                    real_lines.append(('{}/{}'.format('real', r1), i, 0))
                    real_lines.append(('{}/{}'.format('real', r2), i, 0))

                    if dataset == 'ALL':
                        if i > self.frames[mode]//4:
                            continue
                        for fake_d in ['FF-DF', 'FF-NT', 'FF-FS', 'FF-F2F']:
                            img_lines.append(
                                ('{}/{}_{}'.format(fake_d, r1, r2), i, 1))
                            img_lines.append(
                                ('{}/{}_{}'.format(fake_d, r2, r1), i, 1))

                            fake_lines.append(
                                ('{}/{}_{}'.format(fake_d, r1, r2), i, 1))
                            fake_lines.append(
                                ('{}/{}_{}'.format(fake_d, r2, r1), i, 1))
                    else:
                        img_lines.append(
                            ('{}/{}_{}'.format(dataset, r1, r2), i, 1))
                        img_lines.append(
                            ('{}/{}_{}'.format(dataset, r2, r1), i, 1))

                        fake_lines.append(
                            ('{}/{}_{}'.format(dataset, r1, r2), i, 1))
                        fake_lines.append(
                            ('{}/{}_{}'.format(dataset, r2, r1), i, 1))

        if sample_num is not None:
            img_lines = img_lines[:sample_num]

        self.img_lines = np.random.permutation(img_lines)
        self.fake_lines = np.random.permutation(fake_lines)
        self.real_lines = np.random.permutation(real_lines)

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

        self.transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])

        self.totensor = T.Compose([T.ToTensor()])
        self.res = res

    def load_image(self, name, idx):
        impath = '{}/{}/{:04d}.png'.format(self.data_root, name, int(idx))
        ########
        maskpath = '{}/{}/{:04d}_mask.png'.format(self.mask_root, name, int(idx))
        mask = load_mask(maskpath, size=self.res)
        ########
        img = load_rgb(impath, size=self.res)

        return img, mask

    def __getitem__(self, index):
        name, idx, label = self.img_lines[index]

        label = int(label)
        img, mask = self.load_image(name, idx)

        if self.mode == 'train':
            rect = detector(img)
            if len(rect) == 0:
                if img is None:
                    print('Img is None!!!')
                else:
                    print('Face detetor failed ...')
                real_lmk = np.zeros([68, 2]).astype(np.int64)
                fake_lmk = np.zeros([68, 2]).astype(np.int64)
                fake_img = img
                fake_mask = mask
            else:
                sp = predictor(img, rect[0])
                real_lmk = np.array([[p.x, p.y] for p in sp.parts()])
                while True:
                    if label == 0:
                        random = np.random.randint(len(self.fake_lines))
                        name2, idx2, label2 = self.fake_lines[random]
                    else:
                        random = np.random.randint(len(self.real_lines))
                        name2, idx2, label2 = self.real_lines[random]
                    fake_img, fake_mask = self.load_image(name2, idx2)
                    rect2 = detector(fake_img)
                    if len(rect2) != 0:
                        break
                sp2 = predictor(img, rect2[0])
                fake_lmk = np.array([[p.x, p.y] for p in sp2.parts()])


            img = self.transforms(Image.fromarray(np.array(img, dtype=np.uint8)))
            img = img.type(torch.float32)
            #mask = cv2.resize(mask, (16, 16), interpolation=cv2.INTER_CUBIC)
            fake_img = self.transforms(Image.fromarray(np.array(fake_img, dtype=np.uint8)))
            fake_img = fake_img.type(torch.float32)
            real_lmk = self.totensor(real_lmk)
            fake_lmk = self.totensor(fake_lmk)
            mask = self.totensor(mask)
            fake_mask = self.totensor(fake_mask)
            return (img, fake_img, real_lmk, fake_lmk, mask, fake_mask), label
        else:
            img = Image.fromarray(np.array(img, dtype=np.uint8))
            return self.transforms(img), label

    def __len__(self):
        return len(self.img_lines)

