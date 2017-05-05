import sys
import os
import numpy as np
import torch
def getFreeId():
    import pynvml 

    pynvml.nvmlInit()
    def getFreeRatio(id):
        handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        use = pynvml.nvmlDeviceGetUtilizationRates(handle)
        ratio = 0.5*(float(use.gpu+float(use.memory)))
        return ratio

    deviceCount = pynvml.nvmlDeviceGetCount()
    available = []
    for i in range(deviceCount):
        if getFreeRatio(i)<70:
            available.append(i)
    gpus = ''
    for g in available:
        gpus = gpus+str(g)+','
    gpus = gpus[:-1]
    return gpus

def setgpu(gpuinput):
    freeids = getFreeId()
    if gpuinput=='all':
        gpus = freeids
    else:
        gpus = gpuinput
        if any([g not in freeids for g in gpus.split(',')]):
            raise ValueError('gpu'+g+'is being used')
    print('using gpu '+gpus)
    os.environ['CUDA_VISIBLE_DEVICES']=gpus
    return len(gpus.split(','))

class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass    


def split4(data,  max_stride, margin):
    splits = []
    data = torch.Tensor.numpy(data)
    _,c, z, h, w = data.shape

    w_width = np.ceil(float(w / 2 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 2 + margin)/max_stride).astype('int')*max_stride
    pad = int(np.ceil(float(z)/max_stride)*max_stride)-z
    leftpad = pad/2
    pad = [[0,0],[0,0],[leftpad,pad-leftpad],[0,0],[0,0]]
    data = np.pad(data,pad,'constant',constant_values=-1)
    data = torch.from_numpy(data)
    splits.append(data[:, :, :, :h_width, :w_width])
    splits.append(data[:, :, :, :h_width, -w_width:])
    splits.append(data[:, :, :, -h_width:, :w_width])
    splits.append(data[:, :, :, -h_width:, -w_width:])
    
    return torch.cat(splits, 0)

def combine4(output, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        splits[0].shape[0],
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    h0 = output.shape[1] / 2
    h1 = output.shape[1] - h0
    w0 = output.shape[2] / 2
    w1 = output.shape[2] - w0

    splits[0] = splits[0][:, :h0, :w0, :, :]
    output[:, :h0, :w0, :, :] = splits[0]

    splits[1] = splits[1][:, :h0, -w1:, :, :]
    output[:, :h0, -w1:, :, :] = splits[1]

    splits[2] = splits[2][:, -h1:, :w0, :, :]
    output[:, -h1:, :w0, :, :] = splits[2]

    splits[3] = splits[3][:, -h1:, -w1:, :, :]
    output[:, -h1:, -w1:, :, :] = splits[3]

    return output

def split8(data,  max_stride, margin):
    splits = []
    if isinstance(data, np.ndarray):
        c, z, h, w = data.shape
    else:
        _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 2 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 2 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 2 + margin)/max_stride).astype('int')*max_stride
    for zz in [[0,z_width],[-z_width,None]]:
        for hh in [[0,h_width],[-h_width,None]]:
            for ww in [[0,w_width],[-w_width,None]]:
                if isinstance(data, np.ndarray):
                    splits.append(data[np.newaxis, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
                else:
                    splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])

                
    if isinstance(data, np.ndarray):
        return np.concatenate(splits, 0)
    else:
        return torch.cat(splits, 0)

    

def combine8(output, z, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = z / 2
    h_width = h / 2
    w_width = w / 2
    i = 0
    for zz in [[0,z_width],[z_width-z,None]]:
        for hh in [[0,h_width],[h_width-h,None]]:
            for ww in [[0,w_width],[w_width-w,None]]:
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :]
                i = i+1
                
    return output


def split16(data,  max_stride, margin):
    splits = []
    _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 4 + margin)/max_stride).astype('int')*max_stride
    z_pos = [z*3/8-z_width/2,
             z*5/8-z_width/2]
    h_width = np.ceil(float(h / 2 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 2 + margin)/max_stride).astype('int')*max_stride
    for zz in [[0,z_width],[z_pos[0],z_pos[0]+z_width],[z_pos[1],z_pos[1]+z_width],[-z_width,None]]:
        for hh in [[0,h_width],[-h_width,None]]:
            for ww in [[0,w_width],[-w_width,None]]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
    
    return torch.cat(splits, 0)

def combine16(output, z, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = z / 4
    h_width = h / 2
    w_width = w / 2
    splitzstart = splits[0].shape[0]/2-z_width/2
    z_pos = [z*3/8-z_width/2,
             z*5/8-z_width/2]
    i = 0
    for zz,zz2 in zip([[0,z_width],[z_width,z_width*2],[z_width*2,z_width*3],[z_width*3-z,None]],
                      [[0,z_width],[splitzstart,z_width+splitzstart],[splitzstart,z_width+splitzstart],[z_width*3-z,None]]):
        for hh in [[0,h_width],[h_width-h,None]]:
            for ww in [[0,w_width],[w_width-w,None]]:
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz2[0]:zz2[1], hh[0]:hh[1], ww[0]:ww[1], :, :]
                i = i+1
                
    return output

def split32(data,  max_stride, margin):
    splits = []
    _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 2 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 4 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 4 + margin)/max_stride).astype('int')*max_stride
    
    w_pos = [w*3/8-w_width/2,
             w*5/8-w_width/2]
    h_pos = [h*3/8-h_width/2,
             h*5/8-h_width/2]

    for zz in [[0,z_width],[-z_width,None]]:
        for hh in [[0,h_width],[h_pos[0],h_pos[0]+h_width],[h_pos[1],h_pos[1]+h_width],[-h_width,None]]:
            for ww in [[0,w_width],[w_pos[0],w_pos[0]+w_width],[w_pos[1],w_pos[1]+w_width],[-w_width,None]]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
    
    return torch.cat(splits, 0)

def combine32(splits, z, h, w):
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = int(np.ceil(float(z) / 2))
    h_width = int(np.ceil(float(h) / 4))
    w_width = int(np.ceil(float(w) / 4))
    splithstart = splits[0].shape[1]/2-h_width/2
    splitwstart = splits[0].shape[2]/2-w_width/2
    
    i = 0
    for zz in [[0,z_width],[z_width-z,None]]:
        
        for hh,hh2 in zip([[0,h_width],[h_width,h_width*2],[h_width*2,h_width*3],[h_width*3-h,None]],
                          [[0,h_width],[splithstart,h_width+splithstart],[splithstart,h_width+splithstart],[h_width*3-h,None]]):
            
            for ww,ww2 in zip([[0,w_width],[w_width,w_width*2],[w_width*2,w_width*3],[w_width*3-w,None]],
                              [[0,w_width],[splitwstart,w_width+splitwstart],[splitwstart,w_width+splitwstart],[w_width*3-w,None]]):
                
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz[0]:zz[1], hh2[0]:hh2[1], ww2[0]:ww2[1], :, :]
                i = i+1
                
    return output



def split64(data,  max_stride, margin):
    splits = []
    _,c, z, h, w = data.size()
    
    z_width = np.ceil(float(z / 4 + margin)/max_stride).astype('int')*max_stride
    w_width = np.ceil(float(w / 4 + margin)/max_stride).astype('int')*max_stride
    h_width = np.ceil(float(h / 4 + margin)/max_stride).astype('int')*max_stride
    
    z_pos = [z*3/8-z_width/2,
             z*5/8-z_width/2]
    w_pos = [w*3/8-w_width/2,
             w*5/8-w_width/2]
    h_pos = [h*3/8-h_width/2,
             h*5/8-h_width/2]

    for zz in [[0,z_width],[z_pos[0],z_pos[0]+z_width],[z_pos[1],z_pos[1]+z_width],[-z_width,None]]:
        for hh in [[0,h_width],[h_pos[0],h_pos[0]+h_width],[h_pos[1],h_pos[1]+h_width],[-h_width,None]]:
            for ww in [[0,w_width],[w_pos[0],w_pos[0]+w_width],[w_pos[1],w_pos[1]+w_width],[-w_width,None]]:
                splits.append(data[:, :, zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1]])
    
    return torch.cat(splits, 0)

def combine64(output, z, h, w):
    splits = []
    for i in range(len(output)):
        splits.append(output[i])
 
    output = np.zeros((
        z,
        h,
        w,
        splits[0].shape[3],
        splits[0].shape[4]), np.float32)

    
    z_width = int(np.ceil(float(z) / 4))
    h_width = int(np.ceil(float(h) / 4))
    w_width = int(np.ceil(float(w) / 4))
    splitzstart = splits[0].shape[0]/2-z_width/2
    splithstart = splits[0].shape[1]/2-h_width/2
    splitwstart = splits[0].shape[2]/2-w_width/2
    
    i = 0
    for zz,zz2 in zip([[0,z_width],[z_width,z_width*2],[z_width*2,z_width*3],[z_width*3-z,None]],
                          [[0,z_width],[splitzstart,z_width+splitzstart],[splitzstart,z_width+splitzstart],[z_width*3-z,None]]):
        
        for hh,hh2 in zip([[0,h_width],[h_width,h_width*2],[h_width*2,h_width*3],[h_width*3-h,None]],
                          [[0,h_width],[splithstart,h_width+splithstart],[splithstart,h_width+splithstart],[h_width*3-h,None]]):
            
            for ww,ww2 in zip([[0,w_width],[w_width,w_width*2],[w_width*2,w_width*3],[w_width*3-w,None]],
                              [[0,w_width],[splitwstart,w_width+splitwstart],[splitwstart,w_width+splitwstart],[w_width*3-w,None]]):
                
                output[zz[0]:zz[1], hh[0]:hh[1], ww[0]:ww[1], :, :] = splits[i][zz2[0]:zz2[1], hh2[0]:hh2[1], ww2[0]:ww2[1], :, :]
                i = i+1
                
    return output
