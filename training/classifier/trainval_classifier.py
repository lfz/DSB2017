import time

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn.functional import binary_cross_entropy
from torch.utils.data import DataLoader


def get_lr(epoch, args):
    assert epoch <= args.lr_stage2[-1]
    if args.lr == None:
        lrstage = np.sum(epoch > args.lr_stage2)
        lr = args.lr_preset2[lrstage]
    else:
        lr = args.lr
    return lr


def train_casenet(epoch, model, data_loader, optimizer, args):
    model.train()
    if args.freeze_batchnorm:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.eval()

    starttime = time.time()
    lr = get_lr(epoch, args)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    loss1Hist = []
    loss2Hist = []
    missHist = []
    lossHist = []
    accHist = []
    lenHist = []
    tpn = 0
    fpn = 0
    fnn = 0
    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i, (x, coord, isnod, y) in enumerate(data_loader):
        if args.debug:
            if i > 4:
                break
        coord = Variable(coord).cuda()
        x = Variable(x).cuda()
        xsize = x.size()
        isnod = Variable(isnod).float().cuda()
        ydata = y.numpy()[:, 0]
        y = Variable(y).float().cuda()
        #         weight = 3*torch.ones(y.size()).float().cuda()
        optimizer.zero_grad()
        nodulePred, casePred, casePred_each = model(x, coord)
        loss2 = binary_cross_entropy(casePred, y[:, 0])
        missMask = (casePred_each < args.miss_thresh).float()
        missLoss = -torch.sum(missMask * isnod * torch.log(casePred_each + 0.001)) / xsize[0] / xsize[1]
        loss = loss2 + args.miss_ratio * missLoss
        loss.backward()
        # torch.nn.utils.clip_grad_norm(model.parameters(), 1)

        optimizer.step()
        loss2Hist.append(loss2.data[0])
        missHist.append(missLoss.data[0])
        lenHist.append(len(x))
        outdata = casePred.data.cpu().numpy()

        pred = outdata > 0.5
        tpn += np.sum(1 == pred[ydata == 1])
        fpn += np.sum(1 == pred[ydata == 0])
        fnn += np.sum(0 == pred[ydata == 1])
        acc = np.mean(ydata == pred)
        accHist.append(acc)

    endtime = time.time()
    lenHist = np.array(lenHist)
    loss2Hist = np.array(loss2Hist)
    lossHist = np.array(lossHist)
    accHist = np.array(accHist)

    mean_loss2 = np.sum(loss2Hist * lenHist) / np.sum(lenHist)
    mean_missloss = np.sum(missHist * lenHist) / np.sum(lenHist)
    mean_acc = np.sum(accHist * lenHist) / np.sum(lenHist)
    print('Train, epoch %d, loss2 %.4f, miss loss %.4f, acc %.4f, tpn %d, fpn %d, fnn %d, time %3.2f, lr % .5f '
          % (epoch, mean_loss2, mean_missloss, mean_acc, tpn, fpn, fnn, endtime - starttime, lr))


def val_casenet(epoch, model, data_loader, args):
    model.eval()
    starttime = time.time()
    loss1Hist = []
    loss2Hist = []
    lossHist = []
    missHist = []
    accHist = []
    lenHist = []
    tpn = 0
    fpn = 0
    fnn = 0

    for i, (x, coord, isnod, y) in enumerate(data_loader):
        coord = Variable(coord, volatile=True).cuda()
        x = Variable(x, volatile=True).cuda()
        xsize = x.size()
        ydata = y.numpy()[:, 0]
        y = Variable(y).float().cuda()
        isnod = Variable(isnod).float().cuda()

        nodulePred, casePred, casePred_each = model(x, coord)

        loss2 = binary_cross_entropy(casePred, y[:, 0])
        missMask = (casePred_each < args.miss_thresh).float()
        missLoss = -torch.sum(missMask * isnod * torch.log(casePred_each + 0.001)) / xsize[0] / xsize[1]

        # loss2 = binary_cross_entropy(sigmoid(casePred),y[:,0])
        loss2Hist.append(loss2.data[0])
        missHist.append(missLoss.data[0])
        lenHist.append(len(x))
        outdata = casePred.data.cpu().numpy()
        # print([i,data_loader.dataset.split[i,1],sigmoid(casePred).data.cpu().numpy()])
        pred = outdata > 0.5
        tpn += np.sum(1 == pred[ydata == 1])
        fpn += np.sum(1 == pred[ydata == 0])
        fnn += np.sum(0 == pred[ydata == 1])
        acc = np.mean(ydata == pred)
        accHist.append(acc)
    endtime = time.time()
    lenHist = np.array(lenHist)
    loss2Hist = np.array(loss2Hist)
    accHist = np.array(accHist)
    mean_loss2 = np.sum(loss2Hist * lenHist) / np.sum(lenHist)
    mean_missloss = np.sum(missHist * lenHist) / np.sum(lenHist)
    mean_acc = np.sum(accHist * lenHist) / np.sum(lenHist)
    print('Valid, epoch %d, loss2 %.4f, miss loss %.4f, acc %.4f, tpn %d, fpn %d, fnn %d,  time %3.2f'
          % (epoch, mean_loss2, mean_missloss, mean_acc, tpn, fpn, fnn, endtime - starttime))


def test_casenet(model, testset):
    data_loader = DataLoader(
        testset,
        batch_size=4,
        shuffle=False,
        num_workers=32,
        pin_memory=True)
    # model = model.cuda()
    model.eval()
    predlist = []

    #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
    for i, (x, coord) in enumerate(data_loader):
        coord = Variable(coord).cuda()
        x = Variable(x).cuda()
        nodulePred, casePred, _ = model(x, coord)
        predlist.append(casePred.data.cpu().numpy())
        # print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
    predlist = np.concatenate(predlist)
    return predlist
