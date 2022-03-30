import os
import numpy as np
from skimage import io
import random
import utils
from PIL import Image


im_count = 0
for im in os.listdir('images'):
    im = io.imread('images/' + im)
    if (np.max(im) <= 256):
        im = im / 255
    else:
        print("The image format is not correct. --> %d (supposed to be in [0,255])" % (np.max(im)))

    row, col, ch = im.shape
    var = random.randint(100, 2000) / 10000
    gauss = utils.generateGaussNoise(im, 0, var)
    noise_im = utils.validate_im(im + gauss)
    guide = im[:, :, 0]
    inputs = np.concatenate((noise_im, guide[:, :, None]), 2)
    gt = im
    h1 = int(row / 8) * 8
    w1 = int(col / 8) * 8
    x = inputs[0:h1, 0:w1, :]
    #x *= 255.0/x.max()
    x = Image.fromarray(np.uint8(x)).convert('RGB')
    im_count += 1
    x = x.save('noisy images/' + str(im_count) + '.jpg')