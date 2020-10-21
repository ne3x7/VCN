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
import seaborn as sns
from scipy.interpolate import RectBivariateSpline
from torch.autograd import Variable
import time
from utils.io import mkdir_p
from utils.util_flow import write_flow, save_pfm
from utils.util_flow import (
    calc_compressibility,
    calc_energy_spectrum,
    calc_intermittency,
    pspec
)
cudnn.benchmark = False
sns.set(style="whitegrid", font_scale=1.5)
sns.set_color_codes()
sns.despine()

parser = argparse.ArgumentParser(description='VCN')
parser.add_argument('--image_start', default="SQG_00001_img1.tif",
                    help='first image in series left image name')
parser.add_argument('--image_end', default="SQG_00005_img1.tif",
                    help='last image in series left image name')
parser.add_argument('--dataset', default="SQG",
                    help='folder name')
parser.add_argument('--datapath', default='/gpfs/gpfs0/y.maximov/kolya/piv',
                    help='data path')
parser.add_argument('--loadmodel', default=None,
                    help='model path')
parser.add_argument('--outdir', default='output',
                    help='output path')
parser.add_argument('--model', default='VCN',
                    help='VCN or VCN_small')
parser.add_argument('--testres', type=float, default=1,
                    help='resolution, {1: original resolution, 2: 2X resolution}')
parser.add_argument('--maxdisp', type=int ,default=256,
                    help='maxium disparity. Only affect the coarsest cost volume size')
parser.add_argument('--fac', type=float ,default=1,
                    help='controls the shape of search grid. Only affect the coarse cost volume size')
args = parser.parse_args()



# dataloader
if args.dataset == '2015':
    #from dataloader import kitti15list as DA
    from dataloader import kitti15list_val as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2015test':
    from dataloader import kitti15list as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'tumclip':
    from dataloader import kitticliplist as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'kitticlip':
    from dataloader import kitticliplist as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2012':
    from dataloader import kitti12list as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == '2012test':
    from dataloader import kitti12list as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'mb':
    from dataloader import mblist as DA
    maxw,maxh = [int(args.testres*640), int(args.testres*512)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'chairs':
    from dataloader import chairslist as DA
    maxw,maxh = [int(args.testres*512), int(args.testres*384)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'sinteltest':
    from dataloader import sintellist as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'sintel':
    #from dataloader import sintellist_clean as DA
    from dataloader import sintellist_val as DA
    #from dataloader import sintellist_val_2s as DA
    maxw,maxh = [int(args.testres*1024), int(args.testres*448)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'hd1k':
    from dataloader import hd1klist as DA
    maxw,maxh = [int(args.testres*2560), int(args.testres*1088)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'mbstereo':
    from dataloader import MiddleburySubmit as DA
    maxw,maxh = [int(args.testres*900), int(args.testres*750)]
    test_left_img, test_right_img ,_= DA.dataloader(args.datapath)  
elif args.dataset == 'k15stereo':
    from dataloader import stereo_kittilist15 as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_,_,_,_= DA.dataloader(args.datapath, typ='trainval')  
elif args.dataset == 'k12stereo':
    from dataloader import stereo_kittilist12 as DA
    maxw,maxh = [int(args.testres*1280), int(args.testres*384)]
    test_left_img, test_right_img ,_,_,_,_= DA.dataloader(args.datapath)
elif 'piv' in args.dataset or 'piv' in args.datapath:
    from dataloader import pivlist as DA

    test_left_img, test_right_img, test_flow = DA.dataloader('%s/%s/' % (args.datapath, args.dataset))
    maxw, maxh = [256 * args.testres, 256 * args.testres]
if args.dataset == 'chairs':
    with open('FlyingChairs_train_val.txt', 'r') as f:
        split = [int(i) for i in f.readlines()]
    test_left_img = [test_left_img[i] for i,flag in enumerate(split)     if flag==2]
    test_right_img = [test_right_img[i] for i,flag in enumerate(split)     if flag==2]

if args.model == 'VCN':
    from models.VCN import VCN
elif args.model == 'VCN_small':
    from models.VCN_small import VCN
#if '2015' in args.dataset:
#    model = VCN([1, maxw, maxh], md=[8,4,4,4,4], fac=2)
#elif 'sintel' in args.dataset:
#    model = VCN([1, maxw, maxh], md=[7,4,4,4,4], fac=1.4)
#else:
#    model = VCN([1, maxw, maxh], md=[4,4,4,4,4], fac=1)
model = VCN([1, maxw, maxh], md=[int(4*(args.maxdisp/256)),4,4,4,4], fac=args.fac)
    
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
            
    fig, ax = plt.subplots()
    ax.imshow(flow_to_image(field))
    ax.quiver(xs, ys, uz_mesh, vz_mesh, angles='xy')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(fname)


def test_compressibility(pairs, fname):
    fig, ax = plt.subplots()

    c_trues = []
    c_preds = []

    for v_true, v_pred in pairs:
        c_trues.append(calc_compressibility(v_true))
        c_preds.append(calc_compressibility(v_pred))

    c_trues = np.asarray(c_trues)
    c_preds = np.asarray(c_preds)

    c_trues = c_trues / c_trues.sum(axis=1, keepdims=True)
    c_preds = c_preds / c_preds.sum(axis=1, keepdims=True)

    bins = np.histogram_bin_edges(
        np.concatenate([c_preds.flatten(), c_trues.flatten()]),
        bins=50
    )

    mean_true = np.mean(np.asarray([np.histogram(arr, bins=bins)[0] for arr in c_trues]), axis=0)
    mean_pred = np.mean(np.asarray([np.histogram(arr, bins=bins)[0] for arr in c_preds]), axis=0)

    sns.kdeplot(mean_true, c="b", label="true", ax=ax, alpha=0.1)
    sns.kdeplot(mean_pred, c="g", label="pred", ax=ax, alpha=0.1)

    ax.set_xlabel("Velocity Divergence")
    ax.set_ylabel("Average Probability Density")
    handles, labels = ax.get_legend_handles_labels()
    labels, indices = np.unique(labels, return_index=True)
    labels = labels.tolist()
    handles = np.array(handles)[indices].tolist()
    ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(fname)


def test_energy_spectrum(pairs, fname):
    fig, ax = plt.subplots()
    pspecs_pred = []
    pspecs_true = []
    for v_true, v_pred in pairs:
        xvals_true, yvals_true = pspec(np.absolute(calc_energy_spectrum(v_true)) ** 2, wavenumber=True)
        xvals_pred, yvals_pred = pspec(np.absolute(calc_energy_spectrum(v_pred)) ** 2, wavenumber=True)
        ax.loglog(xvals_true, yvals_true, c="b", alpha=0.1)
        ax.loglog(xvals_pred, yvals_pred, c="g", alpha=0.1)
        pspecs_pred.append(yvals_pred)
        pspecs_true.append(yvals_true)
    ax.loglog(xvals_true, np.mean(np.asarray(pspecs_true), axis=0), c="b", label="true", alpha=1)
    ax.loglog(xvals_pred, np.mean(np.asarray(pspecs_pred), axis=0), c="g", label="pred", alpha=1)
    ax.set_xlabel("Wave number")
    ax.set_ylabel("Average Power Spectrum")
    handles, labels = ax.get_legend_handles_labels()
    labels, indices = np.unique(labels, return_index=True)
    labels = labels.tolist()
    handles = np.array(handles)[indices].tolist()
    ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(fname)


def test_intermittency_r(pairs, fname):
    fig, ax = plt.subplots()
    r_vals = 10. ** np.arange(-3, 4)
    interms_true = []
    interms_pred = []
    for v_true, v_pred in pairs:
        vals_true = [calc_intermittency(v_true, r=r, a=np.deg2rad(0), n=2, n_pts=1000) for r in r_vals]
        vals_pred = [calc_intermittency(v_pred, r=r, a=np.deg2rad(0), n=2, n_pts=1000) for r in r_vals]
        ax.loglog(r_vals, vals_true, c="b", alpha=0.1)
        ax.loglog(r_vals, vals_pred, c="g", alpha=0.1)
        interms_true.append(vals_true)
        interms_pred.append(vals_pred)
    ax.loglog(r_vals, np.mean(np.asarray(interms_true), axis=0), c="b", label="true", alpha=1)
    ax.loglog(r_vals, np.mean(np.asarray(interms_pred), axis=0), c="g", label="pred", alpha=1)
    ax.set_xlabel("Scale, $r$")
    ax.set_ylabel("Average $2$-order Structure Function")
    handles, labels = ax.get_legend_handles_labels()
    labels, indices = np.unique(labels, return_index=True)
    labels = labels.tolist()
    handles = np.array(handles)[indices].tolist()
    ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(fname)


def test_intermittency_n(pairs, fname):
    fig, ax = plt.subplots()
    n_vals = np.arange(2, 20)
    interms_true = []
    interms_pred = []
    for v_true, v_pred in pairs:
        vals_true = [calc_intermittency(v_true, r=3, a=np.deg2rad(0), n=n, n_pts=1000) for n in n_vals]
        vals_pred = [calc_intermittency(v_pred, r=3, a=np.deg2rad(0), n=n, n_pts=1000) for n in n_vals]
        ax.semilogy(n_vals, vals_true, c="b", alpha=0.1)
        ax.semilogy(n_vals, vals_pred, c="g", alpha=0.1)
        interms_true.append(vals_true)
        interms_pred.append(vals_pred)
    ax.semilogy(n_vals, np.mean(np.asarray(interms_true), axis=0), c="b", label="true", alpha=1)
    ax.semilogy(n_vals, np.mean(np.asarray(interms_pred), axis=0), c="g", label="pred", alpha=1)
    ax.set_xlabel("Order, $n$")
    ax.set_ylabel("Average $n$-th order Structure Function")
    handles, labels = ax.get_legend_handles_labels()
    labels, indices = np.unique(labels, return_index=True)
    labels = labels.tolist()
    handles = np.array(handles)[indices].tolist()
    ax.legend(handles, labels)
    fig.tight_layout()
    fig.savefig(fname)


inx_st = test_left_img.index(args.image_start)
inx_fn = test_left_img.index(args.image_end)


mkdir_p('%s/%s' % (args.outdir, args.dataset))
def main():
    model.eval()

    flow_pairs = []

    for inx in range(inx_st, inx_fn+1):
        print(test_left_img[inx])
        flo = read_flow(test_flow[inx])
        imgL_o = np.asarray(Image.open(test_left_img[inx]))
        imgR_o = np.asarray(Image.open(test_right_img[inx]))

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
            rts = model(imgLR)
            torch.cuda.synchronize()
            pred_disp, entropy = rts

        # upsampling
        pred_disp = torch.squeeze(pred_disp).data.cpu().numpy()
        pred_disp = cv2.resize(np.transpose(pred_disp,(1,2,0)), (input_size[1], input_size[0]))
        pred_disp[:,:,0] *= input_size[1] / max_w
        pred_disp[:,:,1] *= input_size[0] / max_h
        flow = np.ones([pred_disp.shape[0],pred_disp.shape[1],3])
        flow[:,:,:2] = pred_disp

        flow_pairs.append((flo, flow))

    test_compressibility(
        flow_pairs,
        '%s/%s/%d-%d-compr.png' % (args.outdir, args.dataset, inx_st, inx_fn)
    )
    test_energy_spectrum(
        flow_pairs,
        '%s/%s/%d-%d-energy.png' % (args.outdir, args.dataset, inx_st, inx_fn)
    )
    test_intermittency_r(
        flow_pairs,
        '%s/%s/%d-%d-interm-r.png' % (args.outdir, args.dataset, inx_st, inx_fn)
    )
    test_intermittency_n(
        flow_pairs,
        '%s/%s/%d-%d-interm-n.png' % (args.outdir, args.dataset, inx_st, inx_fn)
    )

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

