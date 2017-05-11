import torch
from torch import nn
from layers import *
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import rotate
import numpy as np
import os
import sys
sys.path.append('../')
from config_training import config as config_training

config = {}
config['topk'] = 5
config['resample'] = None
config['datadir'] = config_training['preprocess_result_path']
config['preload_train'] = True
config['bboxpath'] = config_training['bbox_path']
config['labelfile'] = './full_label.csv'
config['preload_val'] = True

config['padmask'] = False

config['crop_size'] = [96,96,96]
config['scaleLim'] = [0.85,1.15]
config['radiusLim'] = [6,100]
config['jitter_range'] = 0.15
config['isScale'] = True

config['random_sample'] = True
config['T'] = 1
config['topk'] = 5
config['stride'] = 4
config['augtype'] = {'flip':True,'swap':True,'rotate':True,'scale':True}

config['detect_th'] = 0.05
config['conf_th'] = -1
config['nms_th'] = 0.05
config['filling_value'] = 160

config['startepoch'] = 20
config['lr_stage'] = np.array([50,100,140,160,180])
config['lr'] = [0.01,0.001,0.0001,0.00001,0.000001]
config['miss_ratio'] = 1
config['miss_thresh'] = 0.03

class CaseNet(nn.Module):
    def __init__(self,topk,nodulenet):
        super(CaseNet,self).__init__()
        self.NoduleNet  = nodulenet
        self.fc1 = nn.Linear(128,64)
        self.fc2 = nn.Linear(64,1)
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.baseline = nn.Parameter(torch.Tensor([-30.0]).float())
        self.Relu = nn.ReLU()
    def forward(self,xlist,coordlist):
#         xlist: n x k x 1x 96 x 96 x 96
#         coordlist: n x k x 3 x 24 x 24 x 24
        xsize = xlist.size()
        corrdsize = coordlist.size()
        xlist = xlist.view(-1,xsize[2],xsize[3],xsize[4],xsize[5])
        coordlist = coordlist.view(-1,corrdsize[2],corrdsize[3],corrdsize[4],corrdsize[5])
        
        noduleFeat,nodulePred = self.NoduleNet(xlist,coordlist)
        nodulePred = nodulePred.contiguous().view(corrdsize[0],corrdsize[1],-1)
        
        featshape = noduleFeat.size()#nk x 128 x 24 x 24 x24
        centerFeat = self.pool(noduleFeat[:,:,featshape[2]/2-1:featshape[2]/2+1,
                                          featshape[3]/2-1:featshape[3]/2+1,
                                          featshape[4]/2-1:featshape[4]/2+1])
        centerFeat = centerFeat[:,:,0,0,0]
        out = self.dropout(centerFeat)
        out = self.Relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = out.view(xsize[0],xsize[1])
        base_prob = torch.sigmoid(self.baseline)
        casePred = 1-torch.prod(1-out,dim=1)*(1-base_prob.expand(out.size()[0]))
        return nodulePred,casePred,out
