from __future__ import print_function
import sys

from PIL import Image

from utils.flowlib import flow_to_image, read_flow

sys.path.insert(0,'utils/')
#sys.path.insert(0,'dataloader/')
sys.path.insert(0,'models/')
import cv2
import os
import re
import pdb
import argparse
import numpy as np
import skimage.io
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
from matplotlib import pyplot as plt
from scipy.interpolate import RectBivariateSpline
from torch.autograd import Variable
import time
from utils.io import mkdir_p
from utils.util_flow import write_flow, save_pfm
cudnn.benchmark = False

parser = argparse.ArgumentParser(description='VCN')
parser.add_argument('--left', default='2015',
                    help='{2015: KITTI-15, sintel}')
parser.add_argument('--right', default='/ssd/kitti_scene/training/',
                    help='data path')
parser.add_argument('--flow', default=None,
                    help='model path')
parser.add_argument('--model', default='VCN',
                    help='VCN or VCN_small')
parser.add_argument('--testres', type=float, default=1,
                    help='resolution, {1: original resolution, 2: 2X resolution}')
parser.add_argument('--maxdisp', type=int ,default=64,
                    help='maxium disparity. Only affect the coarsest cost volume size')
parser.add_argument('--fac', type=float ,default=1,
                    help='controls the shape of search grid. Only affect the coarse cost volume size')
args = parser.parse_args()

if args.model == 'VCN':
    from models.VCN import VCN
elif args.model == 'VCN_small':
    from models.VCN_small import VCN

model = VCN([1, 256, 256], md=[int(4*(args.maxdisp/256)),4,4,4,4], fac=args.fac)
    
model = nn.DataParallel(model, device_ids=[0])
model.cuda()
if args.loadmodel is not None:

    pretrained_dict = torch.load(args.loadmodel)
    mean_L=pretrained_dict['mean_L']
    mean_R=pretrained_dict['mean_R']
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'grid' not in k and (('flow_reg' not in k) or ('conv1' in k))}

    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    mean_L = [[1.]]
    mean_R = [[1.]]
    print('dry run')

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def arrow_pic(field, fname):
    s = np.array(field.shape[:-1])
    sz = np.min(s / 40)
    
    Y, X = s
    ys = np.arange(0.5, Y, sz)
    ny = len(ys)
    xs = np.arange(0.5, X, sz)
    nx = len(xs)
    x_mesh, y_mesh = np.meshgrid(xs, ys, indexing='ij')
    
    ipu = RectBivariateSpline(np.arange(X), np.arange(Y), field[..., 0])
    uz_mesh = np.zeros_like(x_mesh)
    for i in range(nx):
        for j in range(ny):
            uz_mesh[i,j] = ipu(x_mesh[i,j], y_mesh[i,j])
            
    ipv = RectBivariateSpline(np.arange(X), np.arange(Y), field[..., 1])
    vz_mesh = np.zeros_like(x_mesh)
    for i in range(nx):
        for j in range(ny):
            vz_mesh[i,j] = ipv(x_mesh[i,j], y_mesh[i,j])
            
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(flow_to_image(field))
    ax.quiver(xs, ys, uz_mesh, vz_mesh, angles='xy')
    ax.axis('off')
    fig.savefig(fname)

def main():
    model.eval()
    ttime_all = []

    rmses = 0
    nrmses = 0

    flo = read_flow(args.flow)
    imgL_o = np.asarray(Image.open(args.left))
    imgR_o = np.asarray(Image.open(args.right))

    # resize
    maxh = imgL_o.shape[0]*args.testres
    maxw = imgL_o.shape[1]*args.testres
    max_h = int(maxh // 64 * 64)
    max_w = int(maxw // 64 * 64)
    if max_h < maxh: max_h += 64
    if max_w < maxw: max_w += 64

    input_size = imgL_o.shape
    imgL = cv2.resize(imgL_o,(max_w, max_h))
    imgR = cv2.resize(imgR_o,(max_w, max_h))

    # flip channel, subtract mean
    imgL = imgL[:, :, None].copy() / 255. - np.asarray(mean_L).mean(0)[np.newaxis,np.newaxis,:]
    imgR = imgR[:, :, None].copy() / 255. - np.asarray(mean_R).mean(0)[np.newaxis,np.newaxis,:]
    print(imgL.shape)
    imgL = np.transpose(imgL, [2,0,1])[np.newaxis]
    imgR = np.transpose(imgR, [2,0,1])[np.newaxis]

    # forward
    imgL = Variable(torch.FloatTensor(imgL).cuda())
    imgR = Variable(torch.FloatTensor(imgR).cuda())
    with torch.no_grad():
        imgLR = torch.cat([imgL,imgR],0)
        model.eval()
        torch.cuda.synchronize()
        start_time = time.time()
        rts = model(imgLR)
        torch.cuda.synchronize()
        ttime = (time.time() - start_time); print('time = %.2f' % (ttime*1000) )
        ttime_all.append(ttime)
        pred_disp, entropy = rts

    # upsampling
    pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()
    pred_disp = cv2.resize(np.transpose(pred_disp,(1,2,0)), (input_size[1], input_size[0]))
    pred_disp[:,:,0] *= input_size[1] / max_w
    pred_disp[:,:,1] *= input_size[0] / max_h
    flow = np.ones([pred_disp.shape[0],pred_disp.shape[1],3])
    flow[:,:,:2] = pred_disp
    rmse = np.sqrt((np.linalg.norm(flow[:,:,:2] - flo[:,:,:2], ord=2, axis=-1) ** 2).mean())
    rmses += rmse
    nrmses += rmse / np.sqrt((np.linalg.norm(flo[:,:,:2], ord=2, axis=-1) ** 2).mean())
    error = np.linalg.norm(flow[:,:,:2] - flo[:,:,:2], ord=2, axis=-1) ** 2
    error = 255 * error / error.max()
    entropy = torch.squeeze(entropy).data.cpu().numpy()
    entropy = cv2.resize(entropy, (input_size[1], input_size[0]))

    idxname = args.left.split('/')[-1]

    cv2.imwrite('%s.png' % idxname.rsplit('.',1)[0], flow_to_image(flow)[:, :, ::-1])
    cv2.imwrite('%s-gt.png' % idxname.rsplit('.', 1)[0], flow_to_image(flo)[:, :, ::-1])
    arrow_pic(flo, '%s-vec-gt.png' % idxname.rsplit('.', 1)[0])
    arrow_pic(flow, '%s-vec.png' % idxname.rsplit('.', 1)[0])
    cv2.imwrite('%s-err.png' % idxname.rsplit('.', 1)[0], error)

    torch.cuda.empty_cache()
    print(ttime_all, rmses, nrmses)

if __name__ == '__main__':
    main()

