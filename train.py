#CUDA_VISIBLE_DEVICES=X python train.py --cuda --outpath ./outputs
from __future__ import print_function
import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch import nn
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
from net import NetS, NetC
from LoadData import Dataset, loader, Dataset_val
from util import hausdorf_distance

# custom weights initialization called on NetS and NetC
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def dice_loss(input,target):
    num=input*target
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=2)

    den1=input*input
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=2)

    den2=target*target
    den2=torch.sum(den2,dim=2)
    den2=torch.sum(den2,dim=2)

    dice=2*(num/(den1+den2))

    dice_total=1-1*torch.sum(dice)/dice.size(0)#divide by batchsize

    return dice_total


def main():
    # setup optimizer
    lr = opt.lr
    decay = opt.decay
    optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(opt.beta1, 0.999))

    max_iou = 0
    min_hd = 0
    NetS.train()
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 1):
            #train C
            NetC.zero_grad()
            input, label = Variable(data[0]), Variable(data[1])
            if cuda:
                input = input.cuda()
                target = label.cuda()
            target = target.type(torch.FloatTensor)
            target = target.cuda()
            output = NetS(input)
            #output = F.sigmoid(output*k)
            output = F.sigmoid(output)
            output = output.detach()
            output_masked = input.clone()
            input_mask = input.clone()
            #detach G from the network
            for d in range(1):
                # print(output.shape)
                # print(output_masked.shape)
                # print(input_mask.shape)
                # print(input_mask[:,d,:,:].unsqueeze(1).shape)                
                output_masked = input_mask[:,d,:,:].unsqueeze(1) * output
            if cuda:
                output_masked = output_masked.cuda()
            result = NetC(output_masked)
            target_masked = input.clone()
            for d in range(1):
                target_masked = input_mask[:,d,:,:].unsqueeze(1) * target
            if cuda:
                target_masked = target_masked.cuda()
            target_D = NetC(target_masked)
            loss_D = - torch.mean(torch.abs(result - target_D))
            loss_D.backward()
            optimizerD.step()
            #clip parameters in D
            for p in NetC.parameters():
                p.data.clamp_(-0.05, 0.05)
            #train G
            NetS.zero_grad()
            output = NetS(input)
            output = F.sigmoid(output)

            for d in range(1):
                output_masked = input_mask[:,d,:,:].unsqueeze(1) * output
            if cuda:
                output_masked = output_masked.cuda()
            result = NetC(output_masked)
            for d in range(1):
                target_masked = input_mask[:,d,:,:].unsqueeze(1) * target
            if cuda:
                target_masked = target_masked.cuda()
            target_G = NetC(target_masked)
            loss_dice = dice_loss(output,target)
            loss_G = torch.mean(torch.abs(result - target_G))
            loss_G_joint = torch.mean(torch.abs(result - target_G)) + loss_dice
            loss_G_joint.backward()
            optimizerG.step()

        print("===> Epoch[{}]({}/{}): Batch Dice: {:.4f}".format(epoch, i, len(dataloader), 1 - loss_dice.item()))
        print("===> Epoch[{}]({}/{}): G_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_G.item()))
        print("===> Epoch[{}]({}/{}): D_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_D.item()))
        vutils.save_image(data[0],
                '%s/input.png' % opt.outpath, 
                normalize=True)
        vutils.save_image(data[1],
                '%s/label.png' % opt.outpath,
                normalize=True)
        vutils.save_image(output.data,
                '%s/result.png' % opt.outpath,
                normalize=True)
        # if epoch % 10 == 0:
        NetS.eval()
        IoUs, dices, hds = [], [], []
        for i, data in enumerate(dataloader_val, 1):
            input, gt = Variable(data[0]), Variable(data[1])
            if cuda:
                input = input.cuda()
                gt = gt.cuda()
            pred = NetS(input)
            pred[pred < 0.5] = 0
            pred[pred >= 0.5] = 1
            pred = pred.type(torch.LongTensor)
            pred_np = pred.data.cpu().numpy()
            gt = gt.data.cpu().numpy()
            

            for x in range(input.size()[0]):
                IoU = np.sum(pred_np[x][gt[x]==1]) / float(np.sum(pred_np[x]) + np.sum(gt[x]) - np.sum(pred_np[x][gt[x]==1]))
                dice = np.sum(pred_np[x][gt[x]==1])*2 / float(np.sum(pred_np[x]) + np.sum(gt[x]))
                
                pred_locations = np.argwhere(pred_np[x] == 1)
                label_locations = np.argwhere(gt[x] == 1) 

                hd = hausdorf_distance(pred_locations, label_locations)
                IoUs.append(IoU)
                dices.append(dice)
                hds.append(hd)

        NetS.train()
        IoUs = np.array(IoUs, dtype=np.float64)
        dices = np.array(dices, dtype=np.float64)
        hds = np.array(hds, dtype=np.float64)        
        mIoU = np.mean(IoUs, axis=0)
        mdice = np.mean(dices, axis=0)
        mhd = np.mean(hds, axis=0)

        print('mIoU: {:.4f}'.format(mIoU))
        print('mDice: {:.4f}'.format(mdice))
        print('mHd: {:.4f}'.format(mhd))

        # if mhd < min_hd or min_hd == 0:
        #     min_hd = mhd
        torch.save(NetS.state_dict(), '{}/NetS_epoch_{}_mHd_{:.4f}_mIoU_{:.4f}.pth'.format(opt.outpath, epoch, mhd, mIoU))
        vutils.save_image(data[0], '{}/input_val_epoch_{}.png'.format(opt.outpath, str(epoch)), normalize=True)
        vutils.save_image(data[1], '{}/label_val_epoch_{}.png'.format(opt.outpath, str(epoch)), normalize=True)
        pred = pred.type(torch.FloatTensor)
        vutils.save_image(pred.data, '{}/result_val_epoch_{}.png'.format(opt.outpath, str(epoch)), normalize=True)
        if epoch % 5 == 0 and epoch > 0:
            lr = lr*decay
            if lr <= 0.00000001:
                lr = 0.00000001
            print('Learning Rate: {:.6f}'.format(lr))
            # print('K: {:.4f}'.format(k))
            print('Max mIoU: {:.4f}'.format(max_iou))
            optimizerG = optim.Adam(NetS.parameters(), lr=lr, betas=(opt.beta1, 0.999))
            optimizerD = optim.Adam(NetC.parameters(), lr=lr, betas=(opt.beta1, 0.999))

if __name__ == "__main__":
    
    # Training settings
    parser = argparse.ArgumentParser(description='Example')
    parser.add_argument('--batchSize', type=int, default=36, help='training batch size')
    parser.add_argument('--niter', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.02')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use, for now it only supports one GPU')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='using GPU or not')
    parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=1111')
    parser.add_argument('--outpath', default='./outputs', help='folder to output images and model checkpoints')
    opt = parser.parse_args()

    print(opt)


    try:
        os.makedirs(opt.outpath)
    except OSError:
        pass

    cuda = opt.cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
    print('===> Building model')
    NetS = NetS(ngpu = opt.ngpu)
    # NetS.apply(weights_init)
    print(NetS)
    NetC = NetC(ngpu = opt.ngpu)
    # NetC.apply(weights_init)
    print(NetC)

    if cuda:
        NetS = NetS.cuda()
        NetC = NetC.cuda()
        # criterion = criterion.cuda()

    # load training data
    dataloader = loader(Dataset('./'),opt.batchSize)
    # load testing data
    dataloader_val = loader(Dataset_val('./'), opt.batchSize)
    
    main()  