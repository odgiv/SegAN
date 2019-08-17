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
from util import hausdorf_distance
from skimage.morphology import skeletonize, label as find_connected

#CUDA_VISIBLE_DEVICE=0 python evaluate.py --weight_path /home/tensorflow/git/odgiiv/code/segan/SegAN/outputs/NetS_epoch_0.pth --store_images

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outpath", default="./test_outputs")
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--ngpu', type=int, default=1)
    parser.add_argument('--weight_path', type=str)
    parser.add_argument("--exp_id", type=int)
    parser.add_argument("--store_images", default=False, action="store_true")
    parser.add_argument("--thinning", default=False, action="store_true")
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

        if opt.thinning:            
            label = skeletonize(np.squeeze(label))
            pred_np = skeletonize(np.squeeze(pred_np))
            new_pred = np.array(pred_np)

            labels, num = find_connected(pred_np, return_num=True)
            for n in range(1, num):
                if np.sum(labels == n) <= 20:
                    new_pred[labels == n] = 0
            
            pred_np = new_pred

        IoU = np.sum(pred_np[label == 1]) / float(np.sum(pred_np) + np.sum(label) - np.sum(pred_np[label == 1]))
        print("Iou: ", IoU)
        IoUs.append(IoU)

        label = np.squeeze(label) 
        pred_np = np.squeeze(pred_np)

        pred_locations = np.argwhere(pred_np == 1)
        label_locations = np.argwhere(label == 1)        

        hd = hausdorf_distance(pred_locations, label_locations)
        hds.append(hd)
        print("Hausdorf: ", hd)

        img = np.squeeze(input.data.cpu().numpy()) * 255
        pred_img = pred_np * 255
        label_img = label * 255
        if opt.store_images:
            pred_img = Image.fromarray(pred_img.astype(np.uint8), mode='P')
            label_img = Image.fromarray(label_img.astype(np.uint8), mode='P')
            img = Image.fromarray(img.astype(np.uint8), mode='P')

            I = Image.new('RGB', (img.size[0]*5, img.size[1]))
            I.paste(img, (0, 0))
            I.paste(label_img, (img.size[0], 0))
            I.paste(pred_img, (img.size[0]*2, 0))
            I.paste(Image.blend(img.convert("L"), label_img.convert("L"), 0.2), (img.size[0]*3, 0))
            I.paste(Image.blend(img.convert("L"), pred_img.convert("L"), 0.2), (img.size[0]*4, 0))
            

            name = 'img_{}_iou_{:.4f}_hausdorf_{:.4f}.jpg'.format(i, IoU, hd)
            I.save(os.path.join(opt.outpath, name))

        # vutils.save_image(pred.data,
        #         '%s/%d.png' % (opt.outpath, i),
        #         normalize=True)

    
    # IoUs = np.array(IoUs, dtype=np.float64)
    mIoU = np.mean(IoUs, axis=0)
    mHd = np.mean(hds, axis=0)
    print("mIoU: ", mIoU, "mHausdorf:", mHd)
    file_name = os.path.basename(opt.weight_path)
    epoch = file_name[len("NetS_epoch_"):file_name.find("_mHd")]
    print("epoch", epoch)
    os.rename(opt.outpath, os.path.join("./exp{}_epoch{}_mIoU_{:.4f}_mHd_{:.4f}".format(opt.exp_id, epoch, mIoU, mHd)))


