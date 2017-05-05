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

config = {}
config['topk'] = 5
config['resample'] = None
config['datadir'] = '/run/shm/preprocess_1_3/'
config['preload_train'] = True
config['bboxpath'] = '../cpliangming/results/res18_prep3/bbox/'
config['labelfile'] = '../stage1_labels.csv'
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
config['augtype'] = {'flip':True,'swap':False,'rotate':False,'scale':False}

config['detect_th'] = 0.05
config['conf_th'] = -1
config['nms_th'] = 0.05
config['filling_value'] = 160

config['startepoch'] = 20
config['lr_stage'] = np.array([50,100,140,160])
config['lr'] = [0.01,0.001,0.0001,0.00001]
config['miss_ratio'] = 1
config['miss_thresh'] = 0.03
config['anchors'] = [10,30,60]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True))
        
        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2,2,3,3]
        num_blocks_back = [3,3]
        self.featureNum_forw = [24,32,64,64,64]
        self.featureNum_back =    [128,64,64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i+1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i+1], self.featureNum_forw[i+1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

            
        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i==0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i+1]+self.featureNum_forw[i+2]+addition, self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2,stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2,stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.drop = nn.Dropout3d(p = 0.2, inplace = False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size = 1),
                                    nn.ReLU(),
                                    #nn.Dropout3d(p = 0.3),
                                   nn.Conv3d(64, 5 * len(config['anchors']), kernel_size = 1))

    def forward(self, x, coord):
        out = self.preBlock(x)#16
        out_pool,indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)#32
        out1_pool,indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)#64
        #out2 = self.drop(out2)
        out2_pool,indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)#96
        out3_pool,indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)#96
        #out4 = self.drop(out4)
        
        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))#96+96
        #comb3 = self.drop(comb3)
        rev2 = self.path2(comb3)
        
        feat = self.back2(torch.cat((rev2, out2,coord), 1))#64+64
        comb2 = self.drop(feat)
        out = self.output(comb2)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        #out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        #out = out.view(-1, 5)
        return feat,out

    
class CaseNet(nn.Module):
    def __init__(self,topk):
        super(CaseNet,self).__init__()
        self.NoduleNet  = Net()
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
