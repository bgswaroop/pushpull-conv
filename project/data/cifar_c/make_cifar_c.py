# -*- coding: utf-8 -*-

import collections
import ctypes
import warnings
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage as sk
import torchvision.datasets as dset
import torchvision.transforms as trn
from PIL import Image as PILImage
from PIL.Image import Resampling
from scipy.ndimage import map_coordinates
from scipy.ndimage import zoom as scizoom
from skimage.filters import gaussian
from torchvision.transforms import InterpolationMode
from wand.api import library as wandlibrary
from wand.image import Image as WandImage

# /////////////// Distortion Helpers ///////////////

warnings.simplefilter("ignore", UserWarning)


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                              ctypes.c_double,  # radius
                                              ctypes.c_double,  # sigma
                                              ctypes.c_double)  # angle


# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
def plasma_fractal(mapsize=32, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()


def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))

    top = (h - ch) // 2
    img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2

    return img[trim_top:trim_top + h, trim_top:trim_top + h]


# /////////////// End Distortion Helpers ///////////////


# /////////////// Distortions ///////////////

def gaussian_noise(x, severity=1):
    # c = [0.04, 0.06, .08, .10, 0.12, 0.14, 0.16, 0.18, 0.19, 0.2][severity - 1]
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def shot_noise(x, severity=1):
    # c = [500, 250, 100, 80, 60, 50, 40, 30, 20, 15][severity - 1]
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


def impulse_noise(x, severity=1):
    # c = [.01, .02, .03, .05, .07, .09, .11, .13, .15, .17][severity - 1]
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]
    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255


def speckle_noise(x, severity=1):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255


def gaussian_blur(x, severity=1):
    c = [1, 2, 3, 4, 6][severity - 1]
    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=2)
    return np.clip(x, 0, 1) * 255


def glass_blur(x, severity=1, height=224, width=224):
    # sigma, max_delta, iterations
    # c = [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2),
    #      (0.5, 1, 1), (0.5, 1, 2), (0.6, 1, 2), (0.6, 1, 3), (0.7, 1, 2)][severity - 1]
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=2) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(height - c[1], c[1], -1):
            for w in range(width - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=2), 0, 1) * 255


def defocus_blur(x, severity=1):
    # c = [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1),
    #      (1.5, 0.5), (1.9, 0.1), (2.2, 0.1), (2.5, 0.1), (3, 0.1)][severity - 1]
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]
    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x32x32 -> 32x32x3

    return np.clip(channels, 0, 1) * 255


def motion_blur(x, severity=1, width=224, height=224):
    # c = [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5),
    #      (9, 3), (10, 3), (11, 3), (12, 3), (13, 4)][severity - 1]
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)

    # if x.shape != (32, 32):
    if x.shape != (width, height):
        return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    else:  # greyscale to RGB
        return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


def zoom_blur(x, severity=1):
    # c = [np.arange(1, 1.06, 0.01),
    #      np.arange(1, 1.11, 0.01),
    #      np.arange(1, 1.14, 0.01),
    #      np.arange(1, 1.18, 0.01),
    #      np.arange(1, 1.20, 0.01),
    #      np.arange(1, 1.22, 0.01),
    #      np.arange(1, 1.24, 0.01),
    #      np.arange(1, 1.26, 0.01),
    #      np.arange(1, 1.28, 0.01),
    #      np.arange(1, 1.31, 0.01)][severity - 1]
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.31, 0.03)][severity - 1]
    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        out += clipped_zoom(x, zoom_factor)

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255


def fog(x, severity=1, height=224, width=224):
    # c = [(.2, 3), (.3, 4), (.5, 3), (.6, 3), (0.75, 2.5),
    #      (1, 2), (1, 2.5), (1.2, 3), (1.5, 1.75), (1.5, 2)][severity - 1]
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]
    x = np.array(x) / 255.
    max_val = x.max()

    if height == 32 and width == 32:
        map_size = 32
    elif height == 224 and width == 224:
        map_size = 256  # This is for image of size 224 (the assigned value is correct)

    x += c[0] * plasma_fractal(map_size, wibbledecay=c[1])[:height, :width][..., np.newaxis]
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255


def frost(x, severity=1, height=224, width=224):
    # c = [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.7, 0.45),
    #      (0.7, 0.48), (0.70, 0.50), (0.70, 0.55), (0.75, 0.55), (0.8, 0.6)][severity - 1]
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    filename = ['./frost1.png', './frost2.png', './frost3.png', './frost4.jpg', './frost5.jpg', './frost6.jpg'][idx]
    frost = cv2.imread(filename)
    if height == 32 and width == 32:
        frost = cv2.resize(frost, (0, 0), fx=0.2, fy=0.2)
    # randomly crop and convert to rgb
    x_start, y_start = np.random.randint(0, frost.shape[0] - height), np.random.randint(0, frost.shape[1] - width)
    frost = frost[x_start:x_start + height, y_start:y_start + width][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)


def snow(x, severity=1, height=224, width=224):
    # c = [(0.1, 0.1, 1, 0.6, 8, 3, 0.95),
    #      (0.1, 0.2, 1, 0.6, 8, 3, 0.95),
    #      (0.12, 0.1, 1, 0.5, 10, 4, 0.9),
    #      (0.12, 0.2, 1, 0.5, 10, 4, 0.9),
    #      (0.15, 0.2, 1.75, 0.55, 10, 4, 0.9),
    #      (0.15, 0.3, 1.75, 0.55, 10, 4, 0.9),
    #      (0.25, 0.2, 2.25, 0.6, 12, 6, 0.85),
    #      (0.25, 0.3, 2.25, 0.6, 12, 6, 0.85),
    #      (0.3, 0.2, 1.25, 0.65, 14, 12, 0.8),
    #      (0.3, 0.3, 1.25, 0.65, 14, 12, 0.8)][severity - 1]
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]

    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(height, width, 1) * 1.5 + 0.5)
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255


def spatter(x, severity=1):
    # c = [(0.62, 0.1, 0.7, 0.7, 0.5, 0),
    #      (0.65, 0.1, 0.8, 0.7, 0.5, 0),
    #      (0.65, 0.3, 1, 0.69, 0.5, 0),
    #      (0.65, 0.1, 0.7, 0.69, 0.6, 1),
    #      (0.65, 0.1, 0.5, 0.68, 0.6, 1)][severity - 1]
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turquoise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255


def contrast(x, severity=1):
    # c = [.75, .6, .55, .5, .45, .4, .35, .3, .25, .15][severity - 1]
    c = [0.4, .3, .2, .1, .05][severity - 1]
    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255


def brightness(x, severity=1):
    # c = [.02, .05, .12, .15, .18, .21, .25, .3, .35, .4][severity - 1]
    c = [.1, .2, .3, .4, .5][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def saturate(x, severity=1):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]
    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255


def jpeg_compression(x, severity=1):
    # c = [80, 70, 65, 60, 55, 50, 45, 40, 35, 25][severity - 1]
    c = [25, 18, 15, 10, 7][severity - 1]
    output = BytesIO()
    x.save(output, 'JPEG', quality=c)
    x = PILImage.open(output)

    return x


def pixelate(x, severity=1, height=224, width=224):
    # c = [0.95, 0.9, 0.85, 0.80, 0.84, .82, .76, .70, .65, .6][severity - 1]
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    x = x.resize((int(height * c), int(width * c)), Resampling.BOX)
    x = x.resize((height, width), Resampling.BOX)

    return x


# mod of https://gist.github.com/erniejunior/601cdf56d2b424757de5
def elastic_transform(image, severity=1, img_size=224):
    # c = [(img_size * 0, img_size * 0, img_size * 0.08),
    #      (img_size * 0.05, img_size * 0.2, img_size * 0.07),
    #      (img_size * 0.08, img_size * 0.06, img_size * 0.06),
    #      (img_size * 0.1, img_size * 0.04, img_size * 0.05),
    #      (img_size * 0.1, img_size * 0.03, img_size * 0.03),
    #      (img_size * 0.12, img_size * 0.04, img_size * 0.04),
    #      (img_size * 0.14, img_size * 0.05, img_size * 0.06),
    #      (img_size * 0.16, img_size * 0.06, img_size * 0.08),
    #      (img_size * 0.18, img_size * 0.06, img_size * 0.09),
    #      (img_size * 2, img_size * 0.7, img_size * 0.1)][severity - 1]
    c = [(img_size * 2, img_size * 0.7, img_size * 0.1),
         (img_size * 2, img_size * 0.08, img_size * 0.2),
         (img_size * 0.05, img_size * 0.01, img_size * 0.02),
         (img_size * 0.07, img_size * 0.01, img_size * 0.02),
         (img_size * 0.12, img_size * 0.01, img_size * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255


# /////////////// End Distortions ///////////////
def process_data(d, method_name, severity, test_data, convert_img):
    labels = []
    cifar_c = []
    corruption = lambda clean_img: d[method_name](clean_img, severity)
    # for img, label in zip([test_data.data[4]], [test_data.targets[4]]):
    for img, label in zip(test_data.data, test_data.targets):
        labels.append(label)
        cifar_c.append(np.uint8(corruption(convert_img(img))))

    return labels, cifar_c


def run_flow(index=None):
    print('Using CIFAR-10 data')

    d = collections.OrderedDict()
    # d['Gaussian Noise'] = gaussian_noise
    # d['Shot Noise'] = shot_noise
    # d['Impulse Noise'] = impulse_noise
    # d['Defocus Blur'] = defocus_blur
    # d['Glass Blur'] = glass_blur
    # d['Motion Blur'] = motion_blur
    # d['Zoom Blur'] = zoom_blur
    if index == 0:
        d['Snow'] = snow
    if index == 1:
        d['Frost'] = frost
    if index == 2:
        d['Fog'] = fog
    if index == 3:
        d['Brightness'] = brightness
    if index == 4:
        d['Contrast'] = contrast
    if index == 5:
        d['Elastic'] = elastic_transform
    if index == 6:
        d['Pixelate'] = pixelate
    if index == 7:
        d['JPEG'] = jpeg_compression

    # d['Speckle Noise'] = speckle_noise
    # d['Gaussian Blur'] = gaussian_blur
    # d['Spatter'] = spatter
    # d['Saturate'] = saturate

    test_data = dset.CIFAR10('/data/p288722/datasets/cifar', train=False)
    convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage(), trn.Resize((224, 224), InterpolationMode.BICUBIC)])
    num_severities = 5

    for method_name in d.keys():
        print('Creating images for the corruption', method_name)

        with Pool(num_severities) as p:
            processed_data = p.starmap(process_data, [
                (d, method_name, x, test_data, convert_img) for x in range(1, num_severities + 1)
            ])

        labels = np.array([y for x in processed_data for y in x[0]]).astype(np.uint8)
        cifar_c = np.array([y for x in processed_data for y in x[1]]).astype(np.uint8)

        # labels = []
        # cifar_c = []
        # for severity in range(1, num_severities + 1):
        #     corruption = lambda clean_img: d[method_name](clean_img, severity)
        #     for img, label in zip([test_data.data[4]], [test_data.targets[4]]):
        #         # for img, label in tqdm(zip(test_data.data, test_data.targets)):
        #         labels.append(label)
        #         cifar_c.append(np.uint8(corruption(convert_img(img))))
        # labels = np.array(labels).astype(np.uint8)
        # cifar_c = np.array(cifar_c).astype(np.uint8)

        root_dir = Path(r'/data/p288722/datasets/cifar/CIFAR-10-C-224x224')
        root_dir.mkdir(exist_ok=True, parents=True)
        np.save(root_dir.joinpath(d[method_name].__name__ + '.npy'), cifar_c)
        np.save(root_dir.joinpath('labels.npy'), labels)

    print('Finished processing!')


def plot_samples():
    root_dir = Path(r'/data/p288722/datasets/cifar/CIFAR-10-C-224x224_Sample')
    root_dir.mkdir(exist_ok=True, parents=True)
    plot_data = (
        # CIFAR-10-C-EnhancedSeverities
        # ('gaussian_noise.npy', [0.04, 0.06, .08, .10, 0.12, 0.14, 0.16, 0.18, 0.19, 0.2]),
        # ('shot_noise.npy', [500, 250, 100, 80, 60, 50, 40, 30, 20, 15]),
        # ('impulse_noise.npy', [.01, .02, .03, .05, .07, .09, .11, .13, .15, .17]),
        # ('defocus_blur.npy', [(0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (1, 0.2), (1.5, 0.1),
        #                       (1.5, 0.5), (1.9, 0.1), (2.2, 0.1), (2.5, 0.1), (3, 0.1)]),
        # ('glass_blur.npy', [(0.05, 1, 1), (0.25, 1, 1), (0.4, 1, 1), (0.25, 1, 2), (0.4, 1, 2),
        #                     (0.5, 1, 1), (0.5, 1, 2), (0.6, 1, 2), (0.6, 1, 3), (0.7, 1, 2)]),
        # ('motion_blur.npy', [(6, 1), (6, 1.5), (6, 2), (8, 2), (9, 2.5),
        #                      (9, 3), (10, 3), (11, 3), (12, 3), (13, 4)]),
        # ('zoom_blur.npy', [1.06, 1.11, 1.14, 1.18, 1.20, 1.22, 1.24, 1.26, 1.28, 1.31]),
        # ('snow.npy', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        # ('frost.npy', [(1, 0.2), (1, 0.3), (0.9, 0.4), (0.85, 0.4), (0.7, 0.45),
        #                (0.7, 0.48), (0.70, 0.50), (0.70, 0.55), (0.75, 0.55), (0.8, 0.6)]),
        # ('fog.npy', [(.2, 3), (.3, 4), (.5, 3), (.6, 3), (0.75, 2.5),
        #              (1, 2), (1, 2.5), (1.2, 3), (1.5, 1.75), (1.5, 2)]),
        # ('brightness.npy', [.02, .05, .12, .15, .18, .21, .25, .3, .35, .4]),
        # ('contrast.npy', [.75, .6, .55, .5, .45, .4, .35, .3, .25, .15]),
        # ('elastic_transform.npy', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        # ('pixelate.npy', [0.95, 0.9, 0.85, 0.80, 0.84, .82, .76, .70, .65, .6]),
        # ('jpeg_compression.npy', [80, 70, 65, 60, 55, 50, 45, 40, 35, 25]),

        # ImageNet-C-Severities
        # ('gaussian_noise.npy', [.08, .12, 0.18, 0.26, 0.38]),
        # ('shot_noise.npy', [60, 25, 12, 5, 3]),
        # ('impulse_noise.npy', [.03, .06, .09, 0.17, 0.27]),
        # ('defocus_blur.npy', [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)]),
        # ('glass_blur.npy', [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)]),
        # ('motion_blur.npy', [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)]),
        # ('zoom_blur.npy', [1.11, 1.16, 1.21, 1.26, 1.31]),

        ('snow.npy', [1, 2, 3, 4, 5]),
        ('frost.npy', [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)]),
        ('fog.npy', [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)]),
        ('brightness.npy', [.1, .2, .3, .4, .5]),
        ('contrast.npy', [0.4, .3, .2, .1, .05]),
        ('elastic_transform.npy', [1, 2, 3, 4, 5]),
        ('pixelate.npy', [0.6, 0.5, 0.4, 0.3, 0.25]),
        ('jpeg_compression.npy', [25, 18, 15, 10, 7]),

    )
    for name, labels in plot_data:
        plt.figure()
        fig, ax = plt.subplots(1, len(labels), figsize=(len(labels), 2))
        images = np.load(root_dir.joinpath(name))

        for idx, severity_level in enumerate(labels):
            ax[idx].imshow(images[idx])
            ax[idx].set_title(f'{severity_level}')
            ax[idx].axis('off')

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.4)
        fig.suptitle(f"{name[:-4]} - for various levels of severity", fontsize="x-large")
        fig.tight_layout()
        fig.savefig(rf'{Path().resolve().parent.joinpath("misc")}/_plot_figures/CIFAR-10-C_simulated_{name[:-4]}.png')
        fig.show()
        plt.gca()
        plt.gcf()
        plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process cmd inputs')
    parser.add_argument('--index', type=int, default=0, help='index for data')
    args = parser.parse_args()
    run_flow(index=args.index)

    # plot_samples()
