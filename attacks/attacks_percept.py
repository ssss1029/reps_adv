import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def tensor_clamp(x, a_min, a_max):
    """
    like torch.clamp, except with bounds defined as tensors
    """
    out = torch.clamp(x - a_max, max=0) + a_max
    out = torch.clamp(out - a_min, min=0) + a_min
    return out


def normalize_l2(x):
    """
    Expects x.shape == [N, C, H, W]
    """
    norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
    norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return x / norm


def tensor_clamp_l2(x, center, radius):
    """batched clamp of x into l2 ball around center of given radius"""
    x = x.data
    diff = x - center
    diff_norm = torch.norm(diff.view(diff.size(0), -1), p=2, dim=1)
    project_select = diff_norm > radius
    if project_select.any():
        diff_norm = diff_norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        new_x = x
        new_x[project_select] = (center + (diff / diff_norm) * radius)[project_select]
        return new_x
    else:
        return x


class PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """

        pnet = PNetLin(pnet_type='allconv', pnet_model=model)

        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx)
                loss_ce = F.cross_entropy(logits, by, reduction='sum')
                loss_percept = torch.sum(pnet(adv_bx, bx))

                print("loss_percept = ", loss_percept.item(), "loss_ce = ", loss_ce.item())

                loss = loss_ce + loss_percept

            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx

class PGD_l2(nn.Module):
    def __init__(self, epsilon, num_steps, step_size):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        init_noise = normalize_l2(torch.randn(bx.size()).cuda()) * np.random.rand() * self.epsilon
        adv_bx = (bx + init_noise).clamp(0, 1).requires_grad_()

        for i in range(self.num_steps):
            logits = model(adv_bx)

            loss = F.cross_entropy(logits, by, reduction='sum')

            grad = normalize_l2(torch.autograd.grad(loss, adv_bx, only_inputs=True)[0])
            adv_bx = adv_bx + self.step_size * grad
            adv_bx = tensor_clamp_l2(adv_bx, bx, self.epsilon).clamp(0, 1)
            adv_bx = adv_bx.data.requires_grad_()

        return adv_bx


####################################################################################
###### PNetLin and friends
####################################################################################


import sys
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import numpy as np
from pdb import set_trace as st
from skimage import color
from IPython import embed

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_H=64): # assumes scale factor is same for H and W
    print("Why is it in upsample???")
    exit()
    in_H = in_tens.shape[2]
    scale_factor = 1.*out_H/in_H

    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)(in_tens)

# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(self, pnet_type='allconv', pnet_model=None, version='0.1'):
        super(PNetLin, self).__init__()

        self.pnet_type = pnet_type
        self.version = version
        # self.scaling_layer = ScalingLayer().cuda()
        self.pnet_model = pnet_model

        if self.pnet_type == 'allconv':
            self.net = allconv_backbone(pnet_model=self.pnet_model, requires_grad=False)
            self.chns = [3, 96, 96, 96, 192, 192, 192, 192, 192, 192]
        else:
            raise

        self.L = len(self.chns)
        

    def forward(self, in0, in1):
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = in0, in1
        # in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = self.normalize_tensor(outs0[kk]), self.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk]-feats1[kk])**2

        res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

        val = res[0]
        for l in range(1,self.L):
            val += res[l]
        
        return val
    
    def normalize_tensor(self, in_feat, eps=1e-10):
        norm_factor = torch.sqrt(torch.sum(in_feat**2,dim=1,keepdim=True))
        return in_feat/(norm_factor+eps)


# class ScalingLayer(nn.Module):
#     def __init__(self):
#         super(ScalingLayer, self).__init__()
#         self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
#         self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

#     def forward(self, inp):
#         return (inp - self.shift) / self.scale

from collections import namedtuple
import torch

class allconv_backbone(torch.nn.Module):
    def __init__(self, pnet_model=None, requires_grad=False):
        super(allconv_backbone, self).__init__()

        pnet_model_features = pnet_model.features
        
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.slice6 = torch.nn.Sequential()
        self.slice7 = torch.nn.Sequential()
        self.slice8 = torch.nn.Sequential()
        self.slice9 = torch.nn.Sequential()
        self.N_slices = 9

        for x in range(0, 3):
            self.slice1.add_module(str(x), pnet_model_features[x])
        for x in range(3, 6):
            self.slice2.add_module(str(x), pnet_model_features[x])
        for x in range(6, 9):
            self.slice3.add_module(str(x), pnet_model_features[x])
        for x in range(9, 14):
            self.slice4.add_module(str(x), pnet_model_features[x])
        for x in range(14, 17):
            self.slice5.add_module(str(x), pnet_model_features[x])
        for x in range(17, 20):
            self.slice6.add_module(str(x), pnet_model_features[x])
        for x in range(20, 25):
            self.slice7.add_module(str(x), pnet_model_features[x])
        for x in range(25, 28):
            self.slice8.add_module(str(x), pnet_model_features[x])
        for x in range(28, 30):
            self.slice9.add_module(str(x), pnet_model_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
        

    def forward(self, X):
        input_ = X

        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        h = self.slice8(h)
        h_relu8 = h
        h = self.slice9(h)
        h_relu9 = h
        
        vgg_outputs = namedtuple("VggOutputs", [
            'input',
            'relu_1',
            'relu_2',
            'relu_3',
            'relu_4',
            'relu_5',
            'relu_6',
            'relu_7',
            'relu_8',
            'relu_9',
        ])

        out = vgg_outputs(
            input_,
            h_relu1,
            h_relu2,
            h_relu3,
            h_relu4,
            h_relu5,
            h_relu6,
            h_relu7,
            h_relu8,
            h_relu9,
        )

        return out



