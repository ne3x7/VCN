from __future__ import print_function
import sys

from PIL import Image

from dataloader import flow_transforms
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
from dataloader.robloader import default_loader
from utils.util_flow import write_flow, save_pfm, random_incompressible_flow, image_from_flow
from utils.flowlib import read_flow

cudnn.benchmark = False
sns.set(style="whitegrid", font_scale=1.5)
sns.despine()

parser = argparse.ArgumentParser(description='VCN')
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

if args.model == 'VCN':
    from models.VCN import VCN
elif args.model == 'VCN_small':
    from models.VCN_small import VCN

maxw, maxh = [256 * args.testres, 256 * args.testres]

model = VCN([1, maxw, maxh], md=[int(4 * (args.maxdisp / 256)), 4, 4, 4, 4], fac=args.fac)

model = nn.DataParallel(model, device_ids=[0])
model.cuda()
if args.loadmodel is not None:

    pretrained_dict = torch.load(args.loadmodel)
    mean_L = pretrained_dict['mean_L']
    mean_R = pretrained_dict['mean_R']
    pretrained_dict['state_dict'] = {k: v for k, v in pretrained_dict['state_dict'].items() if
                                     'grid' not in k and (('flow_reg' not in k) or ('conv1' in k))}

    model.load_state_dict(pretrained_dict['state_dict'], strict=False)
else:
    mean_L = [[1.]]
    mean_R = [[1.]]
    print('dry run')


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
            uz_mesh[i, j] = ipu(x_mesh[i, j], y_mesh[i, j])

    ipv = RectBivariateSpline(np.arange(X), np.arange(Y), field[..., 1])
    vz_mesh = np.zeros_like(x_mesh)
    for i in range(nx):
        for j in range(ny):
            vz_mesh[i, j] = ipv(x_mesh[i, j], y_mesh[i, j])

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(flow_to_image(field))
    ax.quiver(xs, ys, uz_mesh, vz_mesh, angles='xy')
    ax.axis('off')
    fig.savefig(fname)


mkdir_p('%s/%s/' % (args.outdir, "generated"))
def main():
    model.eval()

    flowl0 = "/gpfs/gpfs0/y.maximov/kolya/piv/SQG/SQG_00001_flow.flo"
    iml0 = "/gpfs/gpfs0/y.maximov/kolya/piv/SQG/SQG_00001_img1.tif"
    iml1 = "/gpfs/gpfs0/y.maximov/kolya/piv/SQG/SQG_00001_img2.tif"

    iml0 = default_loader(iml0)
    iml1 = default_loader(iml1)
    iml1 = np.asarray(iml1) / 255.
    iml0 = np.asarray(iml0) / 255.
    iml0 = iml0[:, :, None].copy()  # iml0[:,:,::-1].copy()
    iml1 = iml1[:, :, None].copy()  # iml1[:,:,::-1].copy()
    flowl0 = read_flow(flowl0)

    # flowl0 = random_incompressible_flow(
    #     1,
    #     [256, 256],
    #     np.random.choice([30, 40, 50]), # 10. ** (2 * np.random.rand()),
    #     incompressible=False
    # )
    # iml0, iml1 = image_from_flow(
    #     ppp=np.random.uniform(0.008, 0.1),
    #     pip=np.random.uniform(0.95, 1.0),
    #     flow=flowl0,
    #     intensity_bounds=(0.8, 1),
    #     diameter_bounds=(0.35, 6)
    # )
    # iml0 = iml0.transpose(1, 2, 0).copy()
    # iml1 = iml1.transpose(1, 2, 0).copy()
    # flowl0 = flowl0[0]
    # flowl0 = np.concatenate([
    #     flowl0,
    #     np.ones(flowl0.shape[:-1] + (1,), dtype=flowl0.dtype)
    # ], axis=-1)


    flowl0 = np.ascontiguousarray(flowl0, dtype=np.float32)
    flowl0[np.isnan(flowl0)] = 1e6  # set to max

    cv2.imwrite('%s/%s/%s.png' % (args.outdir, "generated", "flow-orig"), flow_to_image(flowl0)[:, :, ::-1])

    schedule_aug_coeff = 1.0

    scl = 0.  # 0.2 * schedule_aug_coeff
    if scl > 0:
        scl = [0.2 * schedule_aug_coeff, 0., 0.2 * schedule_aug_coeff]
    else:
        scl = None
    rot = 0.17 * schedule_aug_coeff
    if rot > 0:
        rot = [0.17 * schedule_aug_coeff, 0.0]
    else:
        rot = None
    trans = 0.2 * schedule_aug_coeff
    if trans > 0:
        trans = [0.2 * schedule_aug_coeff, 0.0]
    else:
        trans = None

    co_transform = flow_transforms.Compose([
        flow_transforms.Scale(1, order=0),
        flow_transforms.SpatialAug([256, 256], scale=scl,
                                   rot=rot,
                                   trans=trans,
                                   schedule_coeff=1,
                                   order=0,
                                   black=False),
        # flow_transforms.PCAAug(schedule_coeff=schedule_coeff),
        # flow_transforms.ChromaticAug(schedule_coeff=schedule_coeff, noise=self.noise),
    ])

    augmented, flowl0 = co_transform([iml0, iml1], flowl0)
    iml0 = augmented[0]
    iml1 = augmented[1]

    cv2.imwrite('%s/%s/%s.png' % (args.outdir, "generated", "flow"), flow_to_image(flowl0)[:, :, ::-1])
    cv2.imwrite('%s/%s/%s.png' % (args.outdir, "generated", "img1"), iml0[:, :, ::-1])
    cv2.imwrite('%s/%s/%s.png' % (args.outdir, "generated", "img2"), iml1[:, :, ::-1])

if __name__ == '__main__':
    main()
