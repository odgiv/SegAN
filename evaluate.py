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

    NetS = NetS(ngpu = opt.ngpu)
    NetS.load_state_dict(torch.load(opt.weight_path))
    NetS.cuda()
    NetS.eval()
    for i, data in enumerate(dataloader_test, 1):
        input, gt = Variable(data[0]), Variable(data[1])
        if cuda:
            input = input.cuda()
            gt = gt.cuda()

        pred = NetS(input)
        pred[pred < 0.5] = 0
        pred[pred >= 0.5] = 1
        pred = pred.type(torch.FloatTensor)
        vutils.save_image(pred.data,
                '%s/%d.png' % (opt.outpath, i),
                normalize=True)

        # img = Image.fromarray((pred_np * 255).astype(np.uint8))
        # img.save(opt.outpath + "/" + str(i) + '.jpg') 


