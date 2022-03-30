import numpy as np
import h5py
import os
from PIL import Image

f = h5py.File('nyu_depth_v2_labeled.mat', 'r')
d = np.array(f['images'])  # Takes a while to convert to np array
d = d.transpose(0, 3, 2, 1)

dir = r'images'
if not os.path.exists(dir):
    os.makedirs(dir)

im_count = 0
for arr in d:
    im = Image.fromarray(np.uint8(arr)).convert('RGB')
    im_count += 1
    im = im.save('images/' + str(im_count) + '.jpg')


