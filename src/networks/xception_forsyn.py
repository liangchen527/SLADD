"""

Author: Andreas Rössler
"""
import os
import argparse


import math
import torchvision
import torch
# import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F
# from lib.nets.xception import xception

import torch.utils.model_zoo as model_zoo
from torch.nn import init
# from efficientnet_pytorch import EfficientNet


pretrained_settings = {
    'xception': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000,
            'scale': 0.8975  # The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
        }
    }
}


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:   # whether the number of filters grows first
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=2, num_region=7, num_type=2, num_mag=1, inc=6):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_region = num_region
        self.num_type = num_type
        self.num_mag = num_mag
        dropout= 0.5

        # Entry flow
        self.iniconv = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        #self.conv1 = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(
            64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(
            256, 728, 2, 2, start_with_relu=True, grow_first=True)

        # middle flow
        self.block4 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        self.block8 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(
            728, 728, 3, 1, start_with_relu=True, grow_first=True)

        # Exit flow
        self.block12 = Block(
            728, 1024, 2, 2, start_with_relu=True, grow_first=False)

        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)

        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc_region = nn.Sequential(nn.Dropout(p=dropout),nn.Linear(2048, num_region))
        self.fc_type   = nn.Sequential(nn.Dropout(p=dropout),nn.Linear(2048, num_type))
        self.fc_mag    = nn.Sequential(nn.Dropout(p=dropout),nn.Linear(2048, num_mag))

    def fea_part1_0(self, x):
        x = self.iniconv(x)
        x = self.bn1(x)
        x = self.relu(x)  

        return x

    def fea_part1_1(self, x):  
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 

        return x
    
    def fea_part1(self, x):
        x = self.iniconv(x)
        x = self.bn1(x)
        x = self.relu(x)     
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) 

        return x
    
    def fea_part2(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        return x

    def fea_part3(self, x):
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        return x

    def fea_part4(self, x):
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        return x

    def fea_part5(self, x):
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)

        return x
     
    def features(self, input):
        x = self.fea_part1(input)    

        x = self.fea_part2(x)
        x = self.fea_part3(x)
        x = self.fea_part4(x)

        x = self.fea_part5(x)
        return x

    def classifier(self, features):
        x = self.relu(features)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        out = self.last_linear(x)
        return out, x

    def forward(self, input):
        x = self.features(input)
        x = self.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        region_num = self.fc_region(x)
        type_num = self.fc_type(x)
        mag = self.fc_mag(x)

        return region_num, type_num, mag


def xception(num_region=7, num_type=2, num_mag=1, pretrained='imagenet', inc=6):
    model = Xception(num_region=num_region, num_type=num_type, num_mag=num_mag, inc=inc)
    pretrained = False
    if pretrained:
        num_classes = 2
        settings = pretrained_settings['xception'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(
                settings['num_classes'], num_classes)

        model = Xception(num_classes=num_classes)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']
        model.mean = settings['mean']
        model.std = settings['std']

    # TODO: ugly
    #model.last_linear = model.fc
    #del model.fc
    return model



class RPModel(nn.Module):
    def __init__(self,modelchoice,num_out_classes=2,dropout=0.0):
        super(RPModel, self).__init__()
        self.modelchoice = modelchoice

        if modelchoice == 'resnet50':
            self.rmodel = torchvision.models.resnet50(pretrained=True)
        elif modelchoice == 'resnet18':
            self.rmodel = torchvision.models.resnet18(pretrained=True)
        elif modelchoice == 'resnext':
            self.rmodel = torchvision.models.resnext50_32x4d(pretrained=True)
        elif modelchoice == 'inceptionv3':
            self.rmodel = torchvision.models.inception_v3(pretrained=True)
        # elif modelchoice == 'efficientB5':
        #     self.rmodel = EfficientNet.from_pretrained('efficientnet-b5')
        # elif modelchoice == 'efficientB7':
        #     self.rmodel = EfficientNet.from_pretrained('efficientnet-b7')
        else: 
            raise ValueError('No matching model...')


        if modelchoice == 'efficientB5' or modelchoice == 'efficientB7':
            num_ftrs = self.rmodel._fc.in_features
            if not dropout:
                self.rmodel._fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.rmodel._fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'inceptionv3':
            num_ftrs = self.rmodel.AuxLogits.fc.in_features
            self.rmodel.AuxLogits.fc = nn.Linear(num_ftrs, num_out_classes)
            num_ftrs = self.rmodel.fc.in_features
            self.rmodel.fc = nn.Linear(num_ftrs,num_out_classes)
        else:
            num_ftrs = self.rmodel.fc.in_features
            if not dropout:
                self.rmodel.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.rmodel.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )


    def features(self, input):
        return None # logits 

    def classifier(self, features):
        return None, None

    def forward(self, input):
        out = self.rmodel(input)

        return out, None


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """

    def __init__(self, modelchoice, num_region=7, num_type=2, num_mag=1, dropout=0.5, 
    weight_norm=False, return_fea=False, inc=6):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        self.return_fea = return_fea

        if modelchoice == 'xception':

            def return_pytorch04_xception(pretrained=True):
                # Raises warning "src not broadcastable to dst" but thats fine
                model = xception(num_region=num_region, num_type=num_type, num_mag=num_mag, inc=inc, pretrained=False)
                if pretrained:
                    # Load model in torch 0.4+
                    #model.fc = model.last_linear
                    #del model.last_linear
                    state_dict = torch.load(
                        './weights/xception-b5690688.pth')
                    print('Loaded pretrained model (ImageNet)....')
                    for name, weights in state_dict.items():
                        if 'pointwise' in name:
                            state_dict[name] = weights.unsqueeze(
                                -1).unsqueeze(-1)
                    model.load_state_dict(state_dict, strict=False)
                    #model.last_linear = model.fc
                    #del model.fc
                return model

            self.model = return_pytorch04_xception()
            # Replace fc
            
            if inc != 3:
                self.model.iniconv = nn.Conv2d(inc, 32, 3, 2, 0, bias=False)                
                nn.init.xavier_normal(self.model.iniconv.weight.data, gain=0.02)

        elif modelchoice == 'resnet50' or modelchoice == 'resnet18' \
            or modelchoice == 'resnext' or modelchoice == 'inceptionv3' \
            or modelchoice == 'efficientB5'or modelchoice == 'efficientB7':
            
            self.model = RPModel(modelchoice,num_out_classes,dropout)
            
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def set_trainable_up_to(self, boolean=False, layername="Conv2d_4a_3x3"):
        """
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on lib, for inception e.g. Conv2d_4a_3x3
        :return:
        """
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise NotImplementedError('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            elif self.modelchoice in ['efficientB5','efficientB7']:
                # Make fc trainable
                for param in self.model._fc.parameters():
                    param.requires_grad = True           
            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True

    def forward(self, x):
        region_num, type_num, mag = self.model(x)
        return region_num, type_num, mag


    def features(self, x):
        x = self.model.features(x)
        return x

    def classifier(self, x):
        out, x = self.model.classifier(x)
        return out, x


class RawXception(nn.Module):
    """
    Untrained Xception Model
    """

    def __init__(self, num_out_classes=2, inc=3, dropout=0.0):
        super(RawXception, self).__init__()

        self.model = xception(pretrained=None, inc=inc)
        # Replace fc
        num_ftrs = self.model.last_linear.in_features
        if not dropout:
            self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
        else:
            print('Using dropout', dropout)
            self.model.last_linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(num_ftrs, num_out_classes)
            )

    def forward(self, x):
        x = self.model(x)
        return x

    def features(self, x):
        x = self.model.features(x)
        return x

    def classifier(self, x):
        x = self.model.classifier(x)
        return x


def model_selection(modelname, num_out_classes,
                    dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes), 299, \
            True, ['image'], None
    elif modelname == 'resnet18':
        return TransferModel(modelchoice='resnet18', dropout=dropout,
                             num_out_classes=num_out_classes), \
            224, True, ['image'], None
    else:
        raise NotImplementedError(modelname)
