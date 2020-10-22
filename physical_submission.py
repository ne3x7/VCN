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
from utils.util_flow import write_flow, save_pfm, calc_velocity_gradient
from utils.util_flow import (
    calc_compressibility,
    calc_energy_spectrum,
    calc_intermittency,
    pspec
)
cudnn.benchmark = False
sns.set(style="whitegrid", font_scale=1.5)
sns.despine()

parser = argparse.ArgumentParser(description='VCN')
parser.add_argument('--image', default="SQG_00001_img1.tif",
                    help='left image name')
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
    from dataloader import pivlist_val as DA

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


def test_compressibility(v_true, v_pred, fname):
    c_true = calc_compressibility(v_true)
    c_pred = calc_compressibility(v_pred)
    vel_grad = calc_velocity_gradient(v_true)

    fig, ax = plt.subplots()
    sns.distplot(c_true.flatten(), hist=True, bins=50, kde=True, label="true divergence", ax=ax)
    sns.distplot(c_pred.flatten(), hist=True, bins=50, kde=True, label="pred divergence", ax=ax)
    sns.kdeplot(vel_grad.flatten(), label="true velocity gradient", ax=ax, ls='--')
    ax.set_xlabel("Velocity Divergence")
    ax.set_ylabel("Probability Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname)


def test_energy_spectrum(v_true, v_pred, fname):
    fig, ax = plt.subplots()
    ax.loglog(*pspec(np.absolute(calc_energy_spectrum(v_true)) ** 2), label="true")
    ax.loglog(*pspec(np.absolute(calc_energy_spectrum(v_pred)) ** 2), label="pred")
    ax.set_xlabel("Wave number")
    ax.set_ylabel("Power Spectrum")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname)


def test_intermittency_r(v_true, v_pred, fname):
    fig, ax = plt.subplots()
    r_vals = 10. ** np.arange(-3, 4)
    vals_true = [calc_intermittency(v_true, r=r, a=np.deg2rad(0), n=2, n_pts=1000) for r in r_vals]
    vals_pred = [calc_intermittency(v_pred, r=r, a=np.deg2rad(0), n=2, n_pts=1000) for r in r_vals]
    ax.loglog(r_vals, vals_true, label="true")
    ax.loglog(r_vals, vals_pred, label="pred")
    ax.set_xlabel("Scale, $r$")
    ax.set_ylabel("$2$-order Structure Function")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname)


def test_intermittency_n(v_true, v_pred, fname):
    fig, ax = plt.subplots()
    n_vals = np.arange(2, 20)
    vals_true = [calc_intermittency(v_true, r=3, a=np.deg2rad(0), n=n, n_pts=1000) for n in n_vals]
    vals_pred = [calc_intermittency(v_pred, r=3, a=np.deg2rad(0), n=n, n_pts=1000) for n in n_vals]
    ax.semilogy(n_vals, vals_true, label="true")
    ax.semilogy(n_vals, vals_pred, label="pred")
    ax.set_xlabel("Order, $n$")
    ax.set_ylabel("$n$-th order Structure Function")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fname)


mkdir_p('%s/%s/'% (args.outdir, args.dataset))
def main():
    model.eval()
    ttime_all = []

    rmses = 0
    nrmses = 0

    inx = test_left_img.index(args.image)

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
    error = 255 - 255 * error / error.max()
    entropy = torch.squeeze(entropy).data.cpu().numpy()
    entropy = cv2.resize(entropy, (input_size[1], input_size[0]))

    # save predictions
    if args.dataset == 'mbstereo':
        dirname = '%s/%s/%s'%(args.outdir, args.dataset, test_left_img[inx].split('/')[-2])
        mkdir_p(dirname)
        idxname = ('%s/%s')%(dirname.rsplit('/',1)[-1],test_left_img[inx].split('/')[-1])
    else:
        idxname = test_left_img[inx].split('/')[-1]

    if args.dataset == 'mbstereo':
        with open(test_left_img[inx].replace('im0.png','calib.txt')) as f:
            lines = f.readlines()
            #max_disp = int(int(lines[9].split('=')[-1]))
            max_disp = int(int(lines[6].split('=')[-1]))
        with open('%s/%s/%s'% (args.outdir, args.dataset,idxname.replace('im0.png','disp0IO.pfm')),'w') as f:
            save_pfm(f,np.clip(-flow[::-1,:,0].astype(np.float32),0,max_disp) )
        with open('%s/%s/%s/timeIO.txt'%(args.outdir, args.dataset,idxname.split('/')[0]),'w') as f:
            f.write(str(ttime))
    elif args.dataset == 'k15stereo' or args.dataset == 'k12stereo':
        skimage.io.imsave('%s/%s/%s.png'% (args.outdir, args.dataset,idxname.split('.')[0]),(-flow[:,:,0].astype(np.float32)*256).astype('uint16'))
    else:
        # write_flow('%s/%s/%s.png'% (args.outdir, args.dataset,idxname.rsplit('.',1)[0]), flow.copy())
        cv2.imwrite('%s/%s/%s.png' % (args.outdir, args.dataset,idxname.rsplit('.',1)[0]), flow_to_image(flow)[:, :, ::-1])
        cv2.imwrite('%s/%s/%s-gt.png' % (args.outdir, args.dataset, idxname.rsplit('.', 1)[0]), flow_to_image(flo)[:, :, ::-1])
        arrow_pic(flo, '%s/%s/%s-vec-gt.png' % (args.outdir, args.dataset, idxname.rsplit('.', 1)[0]))
        arrow_pic(flow, '%s/%s/%s-vec.png' % (args.outdir, args.dataset, idxname.rsplit('.', 1)[0]))
        test_compressibility(
            flo, flow,
            '%s/%s/%s-compr.png' % (args.outdir, args.dataset, idxname.rsplit('.', 1)[0])
        )
        test_energy_spectrum(
            flo, flow,
            '%s/%s/%s-energy.png' % (args.outdir, args.dataset, idxname.rsplit('.', 1)[0])
        )
        test_intermittency_r(
            flo, flow,
            '%s/%s/%s-interm-r.png' % (args.outdir, args.dataset, idxname.rsplit('.', 1)[0])
        )
        test_intermittency_n(
            flo, flow,
            '%s/%s/%s-interm-n.png' % (args.outdir, args.dataset, idxname.rsplit('.', 1)[0])
        )
        cv2.imwrite('%s/%s/%s-err.png' % (args.outdir, args.dataset, idxname.rsplit('.', 1)[0]), error)
    # cv2.imwrite('%s/%s/ent-%s.png'% (args.outdir, args.dataset,idxname.rsplit('.',1)[0]), entropy*200)

    torch.cuda.empty_cache()
    rmses /= len(test_left_img)
    nrmses /= len(test_left_img)
    print(np.mean(ttime_all), rmses, nrmses)
                
            

if __name__ == '__main__':
    main()

