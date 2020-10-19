import math
import png
import struct
import array
import numpy as np
import cv2
import pdb
from phi.flow import *
from typing import Tuple, List, Optional
from scipy.interpolate import SmoothBivariateSpline, RectBivariateSpline
from scipy import ndimage
from scipy.fftpack import fft2, fftshift
from scipy.special import erf

from io import *

UNKNOWN_FLOW_THRESH = 1e9;
UNKNOWN_FLOW = 1e10;

# Middlebury checks
TAG_STRING = 'PIEH'    # use this when WRITING the file
TAG_FLOAT = 202021.25  # check for this when READING the file

def readPFM(file):
    import re
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(b'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def save_pfm(file, image, scale = 1):
  import sys
  color = None

  if image.dtype.name != 'float32':
    raise Exception('Image dtype must be float32.')

  if len(image.shape) == 3 and image.shape[2] == 3: # color image
    color = True
  elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
    color = False
  else:
    raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

  file.write('PF\n' if color else 'Pf\n')
  file.write('%d %d\n' % (image.shape[1], image.shape[0]))

  endian = image.dtype.byteorder

  if endian == '<' or endian == '=' and sys.byteorder == 'little':
    scale = -scale

  file.write('%f\n' % scale)

  image.tofile(file)


def ReadMiddleburyFloFile(path):
    """ Read .FLO file as specified by Middlebury.

    Returns tuple (width, height, u, v, mask), where u, v, mask are flat
    arrays of values.
    """

    with open(path, 'rb') as fil:
        tag = struct.unpack('f', fil.read(4))[0]
        width = struct.unpack('i', fil.read(4))[0]
        height = struct.unpack('i', fil.read(4))[0]

        assert tag == TAG_FLOAT
        
        #data = np.fromfile(path, dtype=np.float, count=-1)
        #data = data[3:]

        fmt = 'f' * width*height*2
        data = struct.unpack(fmt, fil.read(4*width*height*2))

        u = data[::2]
        v = data[1::2]

        mask = map(lambda x,y: abs(x)<UNKNOWN_FLOW_THRESH and abs(y) < UNKNOWN_FLOW_THRESH, u, v)
        mask = list(mask)
        u_masked = map(lambda x,y: x if y else 0, u, mask)
        v_masked = map(lambda x,y: x if y else 0, v, mask)

    return width, height, list(u_masked), list(v_masked), list(mask)

def ReadKittiPngFile(path):
    """ Read 16-bit .PNG file as specified by KITTI-2015 (flow).

    Returns a tuple, (width, height, u, v, mask), where u, v, mask
    are flat arrays of values.
    """
    # Read .png file.
    png_reader = png.Reader(path)
    data = png_reader.read()
    if data[3]['bitdepth'] != 16:
        raise Exception('bitdepth of ' + path + ' is not 16')

    width = data[0]
    height = data[1]

    # Get list of rows.
    rows = list(data[2])

    u = array.array('f', [0]) * width*height
    v = array.array('f', [0]) * width*height
    mask = array.array('f', [0]) * width*height

    for y, row in enumerate(rows):
        for x in range(width):
            ind = width*y+x
            u[ind] = (row[3*x] - 2**15) / 64.0
            v[ind] = (row[3*x+1] - 2**15) / 64.0
            mask[ind] = row[3*x+2]

            # if mask[ind] > 0:
            #     print(u[ind], v[ind], mask[ind], row[3*x], row[3*x+1], row[3*x+2])

    #png_reader.close()

    return (width, height, u, v, mask)


def WriteMiddleburyFloFile(path, width, height, u, v, mask=None):
    """ Write .FLO file as specified by Middlebury.
    """

    if mask is not None:
        u_masked = map(lambda x,y: x if y else UNKNOWN_FLOW, u, mask)
        v_masked = map(lambda x,y: x if y else UNKNOWN_FLOW, v, mask)
    else:
        u_masked = u
        v_masked = v

    fmt = 'f' * width*height*2
    # Interleave lists
    data = [x for t in zip(u_masked,v_masked) for x in t]

    with open(path, 'wb') as fil:
        fil.write(str.encode(TAG_STRING))
        fil.write(struct.pack('i', width))
        fil.write(struct.pack('i', height))
        fil.write(struct.pack(fmt, *data))


def write_flow(path,flow):
    
    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = flow[:, :, 0:2]*64.+ 2 ** 15
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0

    flow = flow.astype(np.uint16)
    flow = cv2.imwrite(path, flow[:,:,::-1])

    #WriteKittiPngFile(path,
    #     flow.shape[1], flow.shape[0], flow[:,:,0].flatten(), 
    #    flow[:,:,1].flatten(), flow[:,:,2].flatten())
    


def WriteKittiPngFile(path, width, height, u, v, mask=None):
    """ Write 16-bit .PNG file as specified by KITTI-2015 (flow).

    u, v are lists of float values
    mask is a list of floats, denoting the *valid* pixels.
    """

    data = array.array('H',[0])*width*height*3

    for i,(u_,v_,mask_) in enumerate(zip(u,v,mask)):
        data[3*i] = int(u_*64.0+2**15)
        data[3*i+1] = int(v_*64.0+2**15)
        data[3*i+2] = int(mask_)

        # if mask_ > 0:
        #     print(data[3*i], data[3*i+1],data[3*i+2])

    with open(path, 'wb') as png_file:
        png_writer = png.Writer(width=width, height=height, bitdepth=16, compression=3, greyscale=False)
        png_writer.write_array(png_file, data)


def ConvertMiddleburyFloToKittiPng(src_path, dest_path):
    width, height, u, v, mask = ReadMiddleburyFloFile(src_path)
    WriteKittiPngFile(dest_path, width, height, u, v, mask=mask)

def ConvertKittiPngToMiddleburyFlo(src_path, dest_path):
    width, height, u, v, mask = ReadKittiPngFile(src_path)
    WriteMiddleburyFloFile(dest_path, width, height, u, v, mask=mask)


def ParseFilenameKitti(filename):
    # Parse kitti filename (seq_frameno.xx),
    # return seq, frameno, ext.
    # Be aware that seq might contain the dataset name (if contained as prefix)
    ext = filename[filename.rfind('.'):]
    frameno = filename[filename.rfind('_')+1:filename.rfind('.')]
    frameno = int(frameno)
    seq = filename[:filename.rfind('_')]
    return seq, frameno, ext


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data

def load_calib_cam_to_cam(cam_to_cam_file):
    # We'll return the camera calibration as a dictionary
    data = {}

    # Load and parse the cam-to-cam calibration data
    filedata = read_calib_file(cam_to_cam_file)

    # Create 3x4 projection matrices
    P_rect_00 = np.reshape(filedata['P_rect_00'], (3, 4))
    P_rect_10 = np.reshape(filedata['P_rect_01'], (3, 4))
    P_rect_20 = np.reshape(filedata['P_rect_02'], (3, 4))
    P_rect_30 = np.reshape(filedata['P_rect_03'], (3, 4))

    # Compute the camera intrinsics
    data['K_cam0'] = P_rect_00[0:3, 0:3]
    data['K_cam1'] = P_rect_10[0:3, 0:3]
    data['K_cam2'] = P_rect_20[0:3, 0:3]
    data['K_cam3'] = P_rect_30[0:3, 0:3]

    data['b00'] = P_rect_00[0, 3] / P_rect_00[0, 0]
    data['b10'] = P_rect_10[0, 3] / P_rect_10[0, 0]
    data['b20'] = P_rect_20[0, 3] / P_rect_20[0, 0]
    data['b30'] = P_rect_30[0, 3] / P_rect_30[0, 0]

    return data


def random_incompressible_flow(
    batch_size: int,
    size: List[int],
    power: int,
    incompressible: bool = True
) -> np.ndarray:
    """Produces random (possibly incompressible) flow with fixed resolution.

    :param batch_size: number of flows to generate
    :param size: grid size list [sy, sx]
    :param power: kind of magnitude of k (wave vector)
    :param incompressible: removes divergence
    :return: flow at grid shape (batch_size, sy, sx, 2)
    """
    world = World(batch_size=batch_size)
    domain = Domain(size, boundaries=PERIODIC)
    initial_velocity = domain.staggered_grid(
        data=math.randfreq([batch_size] + size + [2], power=power), name='velocity'
    )
    if incompressible:
        initial_velocity = divergence_free(initial_velocity, domain)
    return initial_velocity.at_centers().data


def flow_from_coordinates(
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    coords: np.ndarray,
    flow: np.ndarray,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Get flow at coordinates by interpolating flow on grid.

    :param x_mesh: coordinates in flow
    :param y_mesh: coordinates in flow
    :param coords: shape (n_particles, 2)
    :param vmax: max value for flow
    :param flow: shape (sy, sx, 2)
    :return: flow at coordinates, shape (n_particles, 2)
    """
    u = SmoothBivariateSpline(x=x_mesh.flatten(), y=y_mesh.flatten(), z=flow[:, :, 0].flatten())
    vec_u = u(coords[:, 0], coords[:, 1], grid=False)

    v = SmoothBivariateSpline(x=x_mesh.flatten(), y=y_mesh.flatten(), z=flow[:, :, 1].flatten())
    vec_v = v(coords[:, 0], coords[:, 1], grid=False)

    vec = np.stack([vec_u, vec_v], axis=-1)  # (n_particles, 2)

    if vmax is not None:
        vec = vec * vmax / np.max(np.abs(vec))

    return vec


def particles_from_flow(
    ppp: float,
    pip: float,
    x_mesh: np.ndarray,
    y_mesh: np.ndarray,
    flow: np.ndarray,
    intensity_bounds: Tuple[float, float],
    diameter_bounds: Tuple[float, float],
    vmax: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get particle positions before and after advection.

    :param x_mesh: coordinates in flow
    :param y_mesh: coordinates in flow
    :param ppp: density, particles per pixes
    :param pip: loss, particles in plane percent
    :param flow: shape (batch_size, sy, sx, 2)
    :param vmax: max value for flow
    :param intensity_bounds:
    :param diameter_bounds:
    :return:
        - particle coordinates in first image (batch_size, n_particles, 2)
        - particle coordinates in second image (batch_size, n_particles, 2)
        - particle intensities (batch_size, n_particles)
        - particle diameters (batch_size, n_particles)
    """
    batch_size = flow.shape[0]
    y, x = res = flow.shape[1:-1]
    num_particles = int(np.prod(res) * ppp)
    num_particles_image = int(np.prod(res) * ppp / (pip ** 2))

    coords = np.stack(
        [
            np.random.uniform(1, x, (batch_size, num_particles)),
            np.random.uniform(1, y, (batch_size, num_particles)),
        ],
        axis=-1,
    )

    intens = np.random.uniform(*intensity_bounds, (batch_size, num_particles))
    diams = np.random.uniform(*diameter_bounds, (batch_size, num_particles))

    vecs = np.stack([
        flow_from_coordinates(x_mesh, y_mesh, c, f, vmax=vmax) for c, f in zip(coords, flow)
    ], axis=0)

    coords1 = coords - 0.5 * vecs
    coords2 = coords + 0.5 * vecs

    coords1 = coords1[:, :num_particles_image]
    coords2 = coords2[:, -num_particles_image:]

    return coords1, coords2, intens, diams


def image_from_flow(
    ppp: float,
    pip: float,
    flow: np.ndarray,
    **options
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates images from flow.

    :param ppp: density, particles per pixes
    :param pip: loss, particles in plane percent
    :param flow: shape (batch_size, sy, sx, 2)
    :param options:
    :return:
    """
    y, x = res = flow.shape[1:-1]
    x_mesh, y_mesh = meshgrid = np.meshgrid(np.arange(x), np.arange(y), indexing='ij')

    coords1, coords2, intens, diams = particles_from_flow(ppp, pip, x_mesh, y_mesh, flow, **options)

    images1 = faster_images_from_particles(res, meshgrid, coords1, intens, diams)
    images2 = faster_images_from_particles(res, meshgrid, coords2, intens, diams)

    return images1, images2


def images_from_particles(
    res: List[int],
    meshgrid: Tuple[np.ndarray, np.ndarray],
    coords: np.ndarray,
    intens: np.ndarray,
    diams: np.ndarray
) -> np.ndarray:
    """Generates images assuming particles are Gaussians.

    :param res: resolution
    :param meshgrid: coordinates in flow
    :param coords: particle coordinates (batch_size, n_particles, 2)
    :param intens: particle intensities (batch_size, n_particles)
    :param diams: particle diameters (batch_size, n_particles)
    :return: images (batch_size, res)
    """
    bs = coords.shape[0]
    image = np.zeros([bs] + list(res), dtype=np.float32)

    xg, yg = meshgrid

    # @nb.jit(nopython=True)
    def transform_image(
        image: np.ndarray,
        coords: np.ndarray,
        intens: np.ndarray,
        diams: np.ndarray,
    ) -> np.ndarray:
        for i in range(coords.shape[0]):
            x = coords[i, 0]
            y = coords[i, 1]
            diam = diams[i]
            inten = intens[i]
            for batch_index in range(bs):
                new_image = inten[batch_index] * np.exp(- 8 * ((xg - x[batch_index]) ** 2 +
                                                               (yg - y[batch_index]) ** 2) /
                                                        (diam[batch_index] ** 2))
                image[batch_index] += new_image
        return image

    image = transform_image(image, coords.transpose(1, 2, 0), intens.T, diams.T)

    return image


def faster_images_from_particles(
    res: List[int],
    meshgrid: Tuple[np.ndarray, np.ndarray],
    coords: np.ndarray,
    intens: np.ndarray,
    diams: np.ndarray,
) -> np.ndarray:
    """Generates images assuming particles are Gaussians.

    :param res:
    :param meshgrid:
    :param coords: particle coordinates (batch_size, n_particles, 2)
    :param intens: particle intensities (batch_size, n_particles)
    :param diams: particle diameters (batch_size, n_particles)
    :return: images (batch_size, res)
    """
    bs, nps, _ = coords.shape
    image = np.zeros([bs] + list(res), dtype=np.float32)

    xg, yg = meshgrid

    # @nb.njit()
    def transform_image(
        image: np.ndarray,
        coords: np.ndarray,
        intens: np.ndarray,
        diams: np.ndarray,
    ) -> np.ndarray:
        for i in range(bs):
            ids_x_0 = np.maximum(coords[i, :, 0] - diams[i], 0).astype(np.int32)
            ids_x_1 = np.minimum(coords[i, :, 0] + diams[i], res[0]).astype(np.int32)
            ids_y_0 = np.maximum(coords[i, :, 1] - diams[i], 0).astype(np.int32)
            ids_y_1 = np.minimum(coords[i, :, 1] + diams[i], res[1]).astype(np.int32)

            for j in range(nps):
                x = coords[i, j, 0]
                y = coords[i, j, 1]
                d = diams[i, j]
                I = intens[i, j]
                # D = vec_erf(2.8284 * 0.5 / d)
                D = erf(2.8284 * 0.5 / d)
                x_st = ids_x_0[j]
                x_fn = ids_x_1[j]
                y_st = ids_y_0[j]
                y_fn = ids_y_1[j]

                #                 M1 = vec_erf(2.8284 * ((xg[x_st:x_fn, y_st:y_fn] - x) + 0.5) / d)
                #                 M2 = vec_erf(2.8284 * ((xg[x_st:x_fn, y_st:y_fn] - x) - 0.5) / d)
                #                 M3 = vec_erf(2.8284 * ((yg[x_st:x_fn, y_st:y_fn] - y) + 0.5) / d)
                #                 M4 = vec_erf(2.8284 * ((yg[x_st:x_fn, y_st:y_fn] - y) - 0.5) / d)

                M1 = erf(2.8284 * ((xg[x_st:x_fn, y_st:y_fn] - x) + 0.5) / d)
                M2 = erf(2.8284 * ((xg[x_st:x_fn, y_st:y_fn] - x) - 0.5) / d)
                M3 = erf(2.8284 * ((yg[x_st:x_fn, y_st:y_fn] - y) + 0.5) / d)
                M4 = erf(2.8284 * ((yg[x_st:x_fn, y_st:y_fn] - y) - 0.5) / d)

                new_image = I / D * (M1 - M2) * (M3 - M4) / 4
                image[i, ids_x_0[j]:ids_x_1[j], ids_y_0[j]:ids_y_1[j]] += new_image

        return image

    return transform_image(image, coords, intens, diams)


def calc_compressibility(v):
    sx = ndimage.sobel(v[0], axis=1, mode='constant')
    sy = ndimage.sobel(v[1], axis=0, mode='constant')
    return sx + sy


def calc_energy_spectrum(v):
    return fftshift(fft2(v[..., 0] + 1j * v[..., 1]))


def calc_intermittency(v, r, a, n, n_pts=1):
    ny, nx, _ = v.shape

    x_mesh, y_mesh = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')

    x0s = np.random.randint(0, nx, n_pts)
    y0s = np.random.randint(0, ny, n_pts)

    dx = r * np.cos(a)
    dy = r * np.sin(a)

    coords1 = np.stack([x0s, y0s], axis=-1)
    coords2 = np.stack([x0s + dx, y0s + dy], axis=-1)

    u = SmoothBivariateSpline(x=x_mesh.flatten(), y=y_mesh.flatten(), z=v[:, :, 0].flatten())
    vec_u1 = u(coords1[:, 0], coords1[:, 1], grid=False)
    vec_u2 = u(coords2[:, 0], coords2[:, 1], grid=False)

    v = SmoothBivariateSpline(x=x_mesh.flatten(), y=y_mesh.flatten(), z=v[:, :, 1].flatten())
    vec_v1 = v(coords1[:, 0], coords1[:, 1], grid=False)
    vec_v2 = v(coords2[:, 0], coords2[:, 1], grid=False)

    vec1 = np.stack([vec_u1, vec_v1], axis=-1)  # (n_pts, 2)
    vec2 = np.stack([vec_u2, vec_v2], axis=-1)  # (n_pts, 2)

    return np.mean(np.linalg.norm(vec1 - vec2, ord=2, axis=-1) ** n, axis=0)


def pspec(psd2, return_index=True, wavenumber=False, return_stddev=False, azbins=1, binsize=1.0, **kwargs):
    """
    Create a Power Spectrum (radial profile of a PSD) from a Power Spectral Density image

    return_index - if true, the first return item will be the indexes
    wavenumber - if one dimensional and return_index set, will return a normalized wavenumber instead
    view - Plot the PSD (in logspace)?
    """
    # freq = 1 + numpy.arange( numpy.floor( numpy.sqrt((image.shape[0]/2)**2+(image.shape[1]/2)**2) ) )

    azbins, (freq, zz) = azimuthalAverageBins(psd2, azbins=azbins, interpnan=True, binsize=binsize, **kwargs)
    if len(zz) == 1: zz = zz[0]
    # the "Frequency" is the spatial frequency f = 1/x for the standard numpy fft, which follows the convention
    # A_k =  \sum_{m=0}^{n-1} a_m \exp\left\{-2\pi i{mk \over n}\right\}
    # or
    # F_f = Sum( a_m e^(-2 pi i f x_m)  over the range m,m_max where a_m are the values of the pixels, x_m are the
    # indices of the pixels, and f is the spatial frequency
    freq = freq.astype(
        'float')  # there was a +1.0 here before, presumably to deal with div-by-0, but that shouldn't happen and shouldn't have been "accounted for" anyway

    if return_index:
        if wavenumber:
            return_vals = list((len(freq) / freq, zz))
        else:
            return_vals = list((freq / len(freq), zz))
    else:
        return_vals = list(zz)
    if return_stddev:
        zzstd = azimuthalAverageBins(psd2, azbins=azbins, stddev=True, interpnan=True, binsize=binsize, **kwargs)
        return_vals.append(zzstd)

    return return_vals


def azimuthalAverageBins(image,azbins,symmetric=None, center=None, **kwargs):
    """ Compute the azimuthal average over a limited range of angles
    kwargs are passed to azimuthalAverage """
    y, x = np.indices(image.shape)
    if center is None:
        center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])
    r = np.hypot(x - center[0], y - center[1])
    theta = np.arctan2(x - center[0], y - center[1])
    theta[theta < 0] += 2*np.pi
    theta_deg = theta*180.0/np.pi

    if isinstance(azbins,np.ndarray):
        pass
    elif isinstance(azbins,int):
        if symmetric == 2:
            azbins = np.linspace(0,90,azbins)
            theta_deg = theta_deg % 90
        elif symmetric == 1:
            azbins = np.linspace(0,180,azbins)
            theta_deg = theta_deg % 180
        elif azbins == 1:
            return azbins,azimuthalAverage(image,center=center,returnradii=True,**kwargs)
        else:
            azbins = np.linspace(0,359.9999999999999,azbins)
    else:
        raise ValueError("azbins must be an ndarray or an integer")

    azavlist = []
    for blow,bhigh in zip(azbins[:-1],azbins[1:]):
        mask = (theta_deg > (blow % 360)) * (theta_deg < (bhigh % 360))
        rr,zz = azimuthalAverage(image,center=center,mask=mask,returnradii=True,**kwargs)
        azavlist.append(zz)

    return azbins,rr,azavlist


def azimuthalAverage(image, center=None, stddev=False, returnradii=False, return_nr=False,
                     binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None,
                     mask=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fractional pixels).
    stddev - if specified, return the azimuthal standard deviation instead of the average
    returnradii - if specified, return (radii_array,radial_profile)
    return_nr   - if specified, return number of pixels per radius *and* radius
    binsize - size of the averaging bin.  Can lead to strange results if
        non-binsize factors are used to specify the center and the binsize is
        too large
    weights - can do a weighted average instead of a simple average if this keyword parameter
        is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
        set weights and stddev.
    steps - if specified, will return a double-length bin array and radial
        profile so you can plot a step-form radial profile (which more accurately
        represents what's going on)
    interpnan - Interpolate over NAN values, i.e. bins where there is no data?
        left,right - passed to interpnan; they set the extrapolated values
    mask - can supply a mask (boolean array same size as image with True for OK and False for not)
        to average over only select data.

    If a bin contains NO DATA, it will have a NAN value because of the
    divide-by-sum-of-weights component.  I think this is a useful way to denote
    lack of data, but users let me know if an alternative is prefered...

    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if center is None:
        center = np.array([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = np.hypot(x - center[0], y - center[1])

    if weights is None:
        weights = np.ones(image.shape)
    elif stddev:
        raise ValueError("Weighted standard deviation is not defined.")

    if mask is None:
        mask = np.ones(image.shape, dtype='bool')
    # obsolete elif len(mask.shape) > 1:
    # obsolete     mask = mask.ravel()

    # the 'bins' as initially defined are lower/upper bounds for each bin
    # so that values will be in [lower,upper)
    nbins = int(np.round(r.max() / binsize) + 1)
    maxbin = nbins * binsize
    bins = np.linspace(0, maxbin, nbins + 1)
    # but we're probably more interested in the bin centers than their left or right sides...
    bin_centers = (bins[1:] + bins[:-1]) / 2.0

    # how many per bin (i.e., histogram)?
    # there are never any in bin 0, because the lowest index returned by digitize is 1
    # nr = np.bincount(whichbin)[1:]
    nr = np.histogram(r, bins)[0]

    # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or xrange(1,nbins+1) )
    # radial_prof.shape = bin_centers.shape
    if stddev:
        # Find out which radial bin each point in the map belongs to
        whichbin = np.digitize(r.flat, bins)
        # This method is still very slow; is there a trick to do this with histograms?
        radial_prof = np.array([image.flat[mask.flat * (whichbin == b)].std() for b in range(1, nbins + 1)])
    else:
        radial_prof = np.histogram(r, bins, weights=(image * weights * mask))[0] / \
                      np.histogram(r, bins, weights=(mask * weights))[0]

    if interpnan:
        radial_prof = np.interp(bin_centers, bin_centers[radial_prof == radial_prof],
                                radial_prof[radial_prof == radial_prof], left=left, right=right)

    if steps:
        xarr = np.array(zip(bins[:-1], bins[1:])).ravel()
        yarr = np.array(zip(radial_prof, radial_prof)).ravel()
        return xarr, yarr
    elif returnradii:
        return bin_centers, radial_prof
    elif return_nr:
        return nr, bin_centers, radial_prof
    else:
        return radial_prof
