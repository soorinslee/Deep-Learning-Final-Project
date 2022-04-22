from email.policy import strict
import os, cv2

from zmq import device
import numpy as np
from skimage import io, transform
import torch.nn as nn
import torch
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from models.InvDN_model import InvDN_Model
import random

import utils

from Zero_DCE_model.Zero_DCE.Zero_DCE_code.model import enhance_net_nopool as DCEmodel

class RedFlashDataset(Dataset):
    def __init__(self, data_root, is_cropped = True, is_train = True):
        self.fileID = []
        self.data_root = data_root
        self.is_cropped = is_cropped
        self.is_train = is_train
        self.size_input = 128

        self.device = torch.device("cuda")
        
        for file in os.listdir(data_root):
            if file[0]!='.':
                filename = os.path.join( data_root, file )
                #                self.fileID.append(file)
                self.fileID.append(filename)

    def __getitem__(self, idx):
        filename = self.fileID[idx]
        im = io.imread(filename)
        if ( np.max(im) <= 256 ):
            im = im / 255
        else:
            print("The image format is not correct. --> %d (supposed to be in [0,255])" % (np.max(im)) )
            
        row, col, ch = im.shape
        
        if self.is_train:  
            # two kinds of noise generation model

#             scale = random.randint(5,25)
#             noise_im = np.round( (im**2.5) * 255 / scale )
#             noise_im = np.random.poisson( noise_im )
#             var = 0.001
#             gauss = utils.generateGaussNoise(noise_im, 0, var)
#             noise_im = noise_im / 255
#             noise_im = utils.validate_im( noise_im + gauss )
#             noise_im = utils.validate_im( (noise_im*scale)**0.4 )

            
            var = random.randint(100,2000)/10000
            gauss = utils.generateGaussNoise(im, 0, var)
            noise_im = utils.validate_im( im + gauss )
        
        guide = im[:,:,0]
        if self.is_cropped:
            guide = utils.modulate(guide)             # guided signal modulation
        
        inputs = np.concatenate((noise_im, guide[:,:,None]), 2)
        gt = im
        
        if self.is_cropped:
            
            h1 = random.randint(0, row - self.size_input)
            w1 = random.randint(0, col - self.size_input)
            x = inputs[ h1 : h1+self.size_input, w1 : w1+self.size_input, : ]
            y =     gt[ h1 : h1+self.size_input, w1 : w1+self.size_input, : ]
        
            rotate = random.randint(0, 3)
            if rotate != 0:
                x = np.rot90(x, rotate)
                y = np.rot90(y, rotate)
                    
            if np.random.random() >= 0.5:
                x = cv2.flip(x, flipCode=0)
                y = cv2.flip(y, flipCode=0)
        else:
            h1 = int(row/8) * 8
            w1 = int(col/8) * 8
            x = inputs[ 0:h1, 0:w1, : ]
            y =     gt[ 0:h1, 0:w1, : ]

        x = np.transpose(x,(2,0,1))
        y = np.transpose(y,(2,0,1))
        x = torch.from_numpy(x.copy())
        y = torch.from_numpy(y.copy())
        return (x, y)

    def __len__(self):
        return len(self.fileID)



class RedFlashDatasetDCE(Dataset):
    def __init__(self, data_root, is_cropped = True, is_train = True):
        self.fileID = []
        self.data_root = data_root
        self.is_cropped = is_cropped
        self.is_train = is_train
        self.size_input = 128

        opt = dict(
            is_train=False,
            dist=False,
            train=dict(
                lr_G=float(2e-4),
                beta1=0.9,
                beta2=0.999,
                niter=600000,
                warmup_iter=-1,  # no warm up

                lr_scheme='MultiStepLR',
                lr_steps=[100000, 200000, 300000, 400000, 500000],
                lr_gamma=0.5,

                manual_seed=10,

                val_freq=float(1),

                lambda_fit_forw=16,
                lambda_rec_back=1,
                lambda_ce_forw=1,
                weight_decay_G=float(1e-8),
                gradient_clipping=10,
            ),
            test=dict(
                name='Test_InvDN',
                suffix='~',
                model='InvDN',
                scale= 4,
                crop_border= '~',  # crop border when evaluation. If None(~), crop the scale pixels
                gpu_ids= [0],
                self_ensemble= True,
                gaussian_scale=None,
            ),
            datasets= dict(),
            #### network
            network_G= dict(
                which_model_G=dict(subnet_type= 'Resnet'),
                in_nc= 3,
                out_nc= 3,
                block_num= [8, 8],
                scale= 4,
                init= 'xavier'
            ),
            #### path
            path=dict(
                pretrain_model_G="drive/My Drive/image_filtering/InvDN_ResUnit_x4.pth",
                strict_load=False
                )
        )
        
        
        self.device = torch.device("cuda")
        self.denoiser = DCEmodel().cuda()
        self.denoiser.load_state_dict(torch.load('/content/drive/MyDrive/image_filtering/Zero_DCE_model/Zero_DCE/Zero_DCE_code/snapshots/Epoch99.pth'))
        
        # self.denoiser.to(self.device)
        
        for file in os.listdir(data_root):
            if file[0]!='.':
                filename = os.path.join( data_root, file )
                #                self.fileID.append(file)
                self.fileID.append(filename)

    def __getitem__(self, idx):
        filename = self.fileID[idx]
        im = io.imread(filename)
        if ( np.max(im) <= 256 ):
            im = im / 255
        else:
            print("The image format is not correct. --> %d (supposed to be in [0,255])" % (np.max(im)) )
            
        row, col, ch = im.shape
        
        if self.is_train:  
            # two kinds of noise generation model

#             scale = random.randint(5,25)
#             noise_im = np.round( (im**2.5) * 255 / scale )
#             noise_im = np.random.poisson( noise_im )
#             var = 0.001
#             gauss = utils.generateGaussNoise(noise_im, 0, var)
#             noise_im = noise_im / 255
#             noise_im = utils.validate_im( noise_im + gauss )
#             noise_im = utils.validate_im( (noise_im*scale)**0.4 )

            
            var = random.randint(100,2000)/10000
            gauss = utils.generateGaussNoise(im, 0, var)
            noise_im = utils.validate_im( im + gauss )
        
        guide = im[:,:,0]
        if self.is_cropped:
            guide = utils.modulate(guide)             # guided signal modulation
        
        # run denoiser first
        with torch.no_grad():
            _, denoise_im, _ = self.denoiser(torch.from_numpy(np.expand_dims(np.moveaxis(noise_im, -1, 0), axis=0)).float().to(self.device))
            denoise_im = np.moveaxis(np.squeeze(denoise_im.cpu().numpy()), 0, -1)
        
        inputs = np.concatenate((denoise_im, guide[:,:,None]), 2)
        gt = im
        
        if self.is_cropped:
            
            h1 = random.randint(0, row - self.size_input)
            w1 = random.randint(0, col - self.size_input)
            x = inputs[ h1 : h1+self.size_input, w1 : w1+self.size_input, : ]
            y =     gt[ h1 : h1+self.size_input, w1 : w1+self.size_input, : ]
        
            rotate = random.randint(0, 3)
            if rotate != 0:
                x = np.rot90(x, rotate)
                y = np.rot90(y, rotate)
                    
            if np.random.random() >= 0.5:
                x = cv2.flip(x, flipCode=0)
                y = cv2.flip(y, flipCode=0)
        else:
            h1 = int(row/8) * 8
            w1 = int(col/8) * 8
            x = inputs[ 0:h1, 0:w1, : ]
            y =     gt[ 0:h1, 0:w1, : ]

        x = np.transpose(x,(2,0,1))
        y = np.transpose(y,(2,0,1))
        x = torch.from_numpy(x.copy())
        y = torch.from_numpy(y.copy())
        return (x, y)

    def __len__(self):
        return len(self.fileID)
