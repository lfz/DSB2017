import torch
from torch import nn
from layers import *

config = {}
config['anchors'] = [ 10.0, 25.0, 40.0]
config['chanel'] = 2
config['crop_size'] = [64, 128, 128]
config['stride'] = [2,4,4]
config['max_stride'] = 16
config['num_neg'] = 10
config['th_neg'] = 0.2
config['th_pos'] = 0.5
config['num_hard'] = 1
config['bound_size'] = 12
config['reso'] = [1.5,0.75,0.75]
config['sizelim'] = 6. #mm
config['blacklist'] = ['868b024d9fa388b7ddab12ec1c06af38','d92998a73d4654a442e6d6ba15bbb827','990fbe3f0a1b53878669967b9afd1441','820245d8b211808bd18e78ff5be16fdb',
                       '417','077','188','876','057','087','130','468']

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(2, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True),
            nn.Conv3d(16, 16, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace = True))
        
        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks = [6,6,6,6]
        n_in = [16, 32, 64,96]
        n_out = [32, 64, 96,96]
        for i in range(len(num_blocks)):
            blocks = []
            for j in range(num_blocks[i]):
                if j == 0:
                    if i ==0:
                        blocks.append(nn.MaxPool3d(kernel_size=[1,2,2]))
                        blocks.append(PostRes(n_in[i], n_out[i]))
                    else:
                        blocks.append(nn.MaxPool3d(kernel_size=2))
                        blocks.append(PostRes(n_out[i], n_out[i]))
                else:
                    blocks.append(PostRes(n_out[i], n_out[i]))
            setattr(self, 'group' + str(i + 1), nn.Sequential(*blocks))
        
        self.path1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True))

        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(96, 32, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True))

        self.path3 = nn.Sequential(
            nn.ConvTranspose3d(96, 32, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True),
            nn.ConvTranspose3d(32, 32, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace = True))

        self.combine = nn.Sequential(
            nn.Conv3d(96, 128, kernel_size = 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace = True))

        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        self.output = nn.Conv3d(128, 5 * len(config['anchors']), kernel_size = 1)

    def forward(self, x):
        x = x.view(x.size(0), 2,x.size(2), x.size(3), x.size(4))
        out = self.preBlock(x)

        out1 = self.group1(out)
        out2 = self.group2(out1)
        out3 = self.group3(out2)
        out4 = self.group4(out3)

        out2 = self.path1(out2)
        out3 = self.path2(out3)
        out4 = self.path3(out4)
        out = torch.cat((out2, out3, out4), 1)

        out = self.combine(out)
        out = self.drop(out)
        out = self.output(out)
        size = out.size()
        out = out.view(out.size(0), out.size(1), -1)
        #out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(config['anchors']), 5)
        #out = out.view(-1, 5)
        return out

def get_model():
    net = Net()
    loss = Loss(config['num_hard'])
    get_pbb = GetPBB(config)
    return config, net, loss, get_pbb
