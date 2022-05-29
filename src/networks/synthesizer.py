import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms as T
from skimage import measure
from skimage.transform import PiecewiseAffineTransform, warp
from torch.autograd import Variable
from networks.xception_forsyn import TransferModel
from scipy.ndimage import binary_erosion, binary_dilation
from dataprocess.utils.face_blend import *
from dataprocess.utils.face_align import get_align_mat_new
from dataprocess.utils.color_transfer import color_transfer
from dataprocess.utils.faceswap_utils import blendImages as alpha_blend_fea
from dataprocess.utils import faceswap

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def generate_random_mask(mask, res=256):
    randwl = np.random.randint(10, 60)
    randwr = np.random.randint(10, 60)
    randhu = np.random.randint(10, 60)
    randhd = np.random.randint(10, 60)
    newmask = np.zeros(mask.shape)
    mask = np.where(mask > 0.1, 1, 0)
    props = measure.regionprops(mask)
    if len(props) == 0:
        return newmask
    center_x, center_y = props[0].centroid
    center_x = int(round(center_x))
    center_y = int(round(center_y))
    newmask[max(center_x-randwl, 0):min(center_x+randwr, res-1), max(center_y-randhu, 0):min(center_x+randhd, res-1)]=1
    newmask *= mask
    return newmask

def random_deform(mask, nrows, ncols, mean=0, std=10):
    h, w = mask.shape[:2]
    rows = np.linspace(0, h-1, nrows).astype(np.int32)
    cols = np.linspace(0, w-1, ncols).astype(np.int32)
    rows += np.random.normal(mean, std, size=rows.shape).astype(np.int32)
    rows += np.random.normal(mean, std, size=cols.shape).astype(np.int32)
    rows, cols = np.meshgrid(rows, cols)
    anchors = np.vstack([rows.flat, cols.flat]).T
    assert anchors.shape[1] == 2 and anchors.shape[0] == ncols * nrows
    deformed = anchors + np.random.normal(mean, std, size=anchors.shape)
    np.clip(deformed[:,0], 0, h-1, deformed[:,0])
    np.clip(deformed[:,1], 0, w-1, deformed[:,1])

    trans = PiecewiseAffineTransform()
    trans.estimate(anchors, deformed.astype(np.int32))
    warped = warp(mask, trans)
    warped *= mask
    blured = cv2.GaussianBlur(warped, (5, 5), 3)
    return blured


def get_five_key(landmarks_68):
    # get the five key points by using the landmarks
    leye_center = (landmarks_68[36] + landmarks_68[39])*0.5
    reye_center = (landmarks_68[42] + landmarks_68[45])*0.5
    nose = landmarks_68[33]
    lmouth = landmarks_68[48]
    rmouth = landmarks_68[54]
    leye_left = landmarks_68[36]
    leye_right = landmarks_68[39]
    reye_left = landmarks_68[42]
    reye_right = landmarks_68[45]
    out = [ tuple(x.astype('int32')) for x in [
        leye_center,reye_center,nose,lmouth,rmouth,leye_left,leye_right,reye_left,reye_right
    ]]
    return out

def remove_eyes(image, landmarks, opt):
    ##l: left eye; r: right eye, b: both eye
    if opt == 'l':
        (x1, y1), (x2, y2) = landmarks[5:7]
    elif opt == 'r':
        (x1, y1), (x2, y2) = landmarks[7:9]
    elif opt == 'b':
        (x1, y1), (x2, y2) = landmarks[:2]
    else:
        print('wrong region')
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    if opt != 'b':
        dilation *= 4
    line = binary_dilation(line, iterations=dilation)
    return line

def remove_nose(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[:2]
    x3, y3 = landmarks[2]
    mask = np.zeros_like(image[..., 0])
    x4 = int((x1 + x2) / 2)
    y4 = int((y1 + y2) / 2)
    line = cv2.line(mask, (x3, y3), (x4, y4), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 4)
    line = binary_dilation(line, iterations=dilation)
    return line

def remove_mouth(image, landmarks):
    (x1, y1), (x2, y2) = landmarks[3:5]
    mask = np.zeros_like(image[..., 0])
    line = cv2.line(mask, (x1, y1), (x2, y2), color=(1), thickness=2)
    w = dist((x1, y1), (x2, y2))
    dilation = int(w // 3)
    line = binary_dilation(line, iterations=dilation)
    return line

def blend_fake_to_real(realimg, real_lmk, fakeimg, fakemask, fake_lmk, deformed_fakemask, type, mag):
    # source: fake image
    # taret: real image
    realimg = ((realimg+1)/2 * 255).astype(np.uint8)
    fakeimg = ((fakeimg+1)/2 * 255).astype(np.uint8)
    H, W, C = realimg.shape

    aff_param = np.array(get_align_mat_new(fake_lmk, real_lmk)).reshape(2, 3)
    aligned_src = cv2.warpAffine(fakeimg, aff_param, (W, H),
                                 flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    src_mask = cv2.warpAffine(deformed_fakemask,
                              aff_param, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    src_mask = src_mask > 0  # (H, W)

    tgt_mask = np.asarray(src_mask, dtype=np.uint8)
    tgt_mask = mask_postprocess(tgt_mask)

    ct_modes = ['rct-m', 'rct-fs', 'avg-align', 'faceswap']
    mode_idx = np.random.randint(len(ct_modes))
    mode = ct_modes[mode_idx]

    if mode != 'faceswap':
        c_mask = tgt_mask / 255.
        c_mask[c_mask > 0] = 1
        if len(c_mask.shape) < 3:
            c_mask = np.expand_dims(c_mask, 2)
        src_crop = color_transfer(mode, aligned_src, realimg, c_mask)
    else:
        c_mask = tgt_mask.copy()
        c_mask[c_mask > 0] = 255
        masked_tgt = faceswap.apply_mask(realimg, c_mask)
        masked_src = faceswap.apply_mask(aligned_src, c_mask)
        src_crop = faceswap.correct_colours(masked_tgt, masked_src, np.array(real_lmk))

    if tgt_mask.mean() < 0.005 or src_crop.max()==0:
        out_blend = realimg
    else:
        if type == 0:
            out_blend, a_mask = alpha_blend_fea(src_crop, realimg, tgt_mask,
                                                featherAmount=0.2 * np.random.rand())
        elif type == 1:
            b_mask = (tgt_mask * 255).astype(np.uint8)
            l, t, w, h = cv2.boundingRect(b_mask)
            center = (int(l + w / 2), int(t + h / 2))
            out_blend = cv2.seamlessClone(src_crop, realimg, b_mask, center, cv2.NORMAL_CLONE)
        else:
            out_blend = copy_fake_to_real(realimg, src_crop, tgt_mask, mag)

    return out_blend, tgt_mask

def copy_fake_to_real(realimg, fakeimg, mask, mag):
    mask = np.expand_dims(mask, 2)
    newimg = fakeimg * mask * mag + realimg*(1-mask) + realimg * mask * (1-mag)
    return newimg

class synthesizer(nn.Module):
    def __init__(self):
        super(synthesizer, self).__init__()
        self.netG = TransferModel('xception', num_region=10, num_type=4, num_mag=1, inc=6)
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.transforms = T.Compose([T.ToTensor(), normalize])

    def parse(self, img, reg, real_lmk, fakemask):
        five_key = get_five_key(real_lmk)
        if reg == 0:
            mask = remove_eyes(img, five_key, 'l')
        elif reg == 1:
            mask = remove_eyes(img, five_key, 'r')
        elif reg == 2:
            mask = remove_eyes(img, five_key, 'b')
        elif reg == 3:
            mask = remove_nose(img, five_key)
        elif reg == 4:
            mask = remove_mouth(img, five_key)
        elif reg == 5:
            mask = remove_nose(img, five_key) + remove_eyes(img, five_key, 'l')
        elif reg == 6:
            mask = remove_nose(img, five_key) + remove_eyes(img, five_key, 'r')
        elif reg == 7:
            mask = remove_nose(img, five_key) + remove_eyes(img, five_key, 'b')
        elif reg == 8:
            mask = remove_nose(img, five_key) + remove_mouth(img, five_key)
        elif reg == 9:
            mask = remove_eyes(img, five_key, 'b') + remove_nose(img, five_key) + remove_mouth(img, five_key)
        else:
            mask = generate_random_mask(fakemask)
        mask = random_deform(mask, 5, 5)
        return mask*1.0

    def get_variable(self, inputs, cuda=False, **kwargs):
        if type(inputs) in [list, np.ndarray]:
            inputs = torch.Tensor(inputs)
        if cuda:
            out = Variable(inputs.cuda(), **kwargs)
        else:
            out = Variable(inputs, **kwargs)
        return out
   
    def calculate(self, logits):
        if logits.shape[1] != 1:
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1, self.get_variable(action, requires_grad=False))
        else:
            probs = torch.sigmoid(logits)
            log_prob = torch.log(torch.sigmoid(logits))
            entropy = -(log_prob * probs).sum(1, keepdim=False)
            action = probs
            selected_log_prob = log_prob
        return entropy, selected_log_prob[:, 0], action[:, 0]

    def forward(self, img, fake_img, real_lmk, fake_lmk, real_mask, fake_mask, label):
        region_num, type_num, mag = self.netG(torch.cat((img, fake_img), 1))
        reg_etp, reg_log_prob, reg = self.calculate(region_num)
        type_etp, type_log_prob, type = self.calculate(type_num)
        mag_etp, mag_log_prob, mag = self.calculate(mag)
        entropy = reg_etp + type_etp + mag_etp
        log_prob = reg_log_prob + type_log_prob + mag_log_prob
        newlabel = []
        typelabel = []
        maglabel = []
        magmask = []
        #####################
        alt_img = torch.ones(img.shape)
        alt_mask = np.zeros((img.shape[0],16,16))
        for i in range(img.shape[0]):
            imgcp = np.transpose(img[i].cpu().numpy(), (1,2,0)).copy()
            fake_imgcp = np.transpose(fake_img[i].cpu().numpy(), (1, 2, 0)).copy()
            ##only work for real imgs and not do-nothing choice
            if label[i] == 0 and type[i] != 3:
                mask = self.parse(fake_imgcp, reg[i], fake_lmk[i, 0].cpu().numpy(),
                                  fake_mask[i, 0].cpu().numpy())
                newimg, newmask = blend_fake_to_real(imgcp, real_lmk[i, 0].cpu().numpy(),
                                                     fake_imgcp,fake_mask.cpu().numpy(),
                                                     fake_lmk[i, 0].cpu().numpy(), mask, type[i],
                                                     mag[i].detach().cpu().numpy())
                newimg = self.transforms(Image.fromarray(np.array(newimg, dtype=np.uint8)))
                newlabel.append(int(1))
                typelabel.append(int(type[i].cpu().numpy()))
                if type[i] == 2:
                    magmask.append(int(1))
                else:
                    magmask.append(int(0))
            else:
                newimg = self.transforms(Image.fromarray(np.array((imgcp+1)/2 * 255, dtype=np.uint8)))
                newmask = real_mask[i, 0].cpu().numpy()
                newlabel.append(int(label[i].cpu().numpy()))
                if label[i] == 0:
                    typelabel.append(int(3))
                else:
                    typelabel.append(int(4))
                magmask.append(int(0))
            if newmask is None:
                newmask = np.zeros((16,16))
            newmask = cv2.resize(newmask, (16, 16), interpolation=cv2.INTER_CUBIC)
            alt_img[i] = newimg
            alt_mask[i] = newmask

        alt_mask = torch.from_numpy(alt_mask.astype(np.float32)).unsqueeze(1)
        newlabel = torch.tensor(newlabel)
        typelabel = torch.tensor(typelabel)
        maglabel = mag
        magmask = torch.tensor(magmask)
        return log_prob, entropy, alt_img.detach(), alt_mask.detach(), \
               newlabel.detach(), typelabel.detach(), maglabel.detach(), magmask.detach()

