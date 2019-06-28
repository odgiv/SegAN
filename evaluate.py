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
from LoadData import loader, Dataset_test
from scipy.spatial.distance import directed_hausdorff

#CUDA_VISIBLE_DEVICE=0 python evaluate.py --weight_path /home/tensorflow/git/odgiiv/code/segan/SegAN/outputs/NetS_epoch_0.pth

def hausdorf_distance(a, b):
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outpath", default="./test_outputs")
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--weight_path', type=str)
    opt = parser.parse_args()

    try:
        os.makedirs(opt.outpath)
    except OSError:
        pass
    
    cuda = True
    dataloader_test = loader(Dataset_test('./'), opt.batchSize)

    cudnn.benchmark = True

    IoUs = []
    hds = []

    NetS = NetS(ngpu = opt.ngpu)
    NetS.load_state_dict(torch.load(opt.weight_path))
    NetS.cuda()
    NetS.eval()
    for i, data in enumerate(dataloader_test, 1):
        input, label = Variable(data[0]), Variable(data[1])
        if cuda:
            input = input.cuda()
            label = label.cuda()

        pred = NetS(input)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        pred = pred.type(torch.FloatTensor)

        pred_np = pred.data.cpu().numpy()
        label = label.data.cpu().numpy()

        IoU = np.sum(pred_np[label == 1]) / float(np.sum(pred_np) + np.sum(label) - np.sum(pred_np[label == 1]))
        print("Iou: ", IoU)
        IoUs.append(IoU)

        label = np.squeeze(label) 
        pred_np = np.squeeze(pred_np)

        pred_locations = np.argwhere(pred_np == 1)
        label_locations = np.argwhere(label == 1)        

        hd = hausdorf_distance(pred_locations, label_locations)
        hds.append(hd)

        vutils.save_image(pred.data,
                '%s/%d.png' % (opt.outpath, i),
                normalize=True)

    
    # IoUs = np.array(IoUs, dtype=np.float64)
    mIoU = np.mean(IoUs, axis=0)
    mhd = np.mean(hds, axis=0)
    print("mIoU: ", mIoU, "mHausdorf:", mhd)


