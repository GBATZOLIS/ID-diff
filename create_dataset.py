#testing code

# load and show an image with Pillow
from PIL import Image
import torch
import numpy as np
from iunets.layers import InvertibleDownsampling2D
from torchvision.utils import make_grid, save_image
from pathlib import Path
import torch
from torch import nn
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def permute_channels(haar_image, forward=True):
        permuted_image = torch.zeros_like(haar_image)
        if forward:
            for i in range(4):
                if i == 0:
                    k = 1
                elif i == 1:
                    k = 0
                else:
                    k = i
                for j in range(3):
                    permuted_image[:, 3*k+j, :, :] = haar_image[:, 4*j+i, :, :]
        else:
            for i in range(4):
                if i == 0:
                    k = 1
                elif i == 1:
                    k = 0
                else:
                    k = i
                
                for j in range(3):
                    permuted_image[:,4*j+k,:,:] = haar_image[:, 3*i+j, :, :]

        return permuted_image

def normalise(x, value_range=None):
    if value_range is None:
        x -= x.min()
        x /= x.max()
    else:
        x -= value_range[0]
        x /= value_range[1]
    return x

def create_train_val_test_index_dict(total_num_images, split):
        #return a dictionary that maps each index to the corresponding phase dataset (train, val, test)
        indices = np.arange(total_num_images)
        np.random.shuffle(indices) #in-place operation
        phase_dataset = {}
        for counter, index in enumerate(indices):
            if counter < split[0]*total_num_images:
                folder = 'train'
            elif counter < (split[0]+split[1])*total_num_images:
                folder = 'val'
            else:
                folder = 'test'
            phase_dataset[index] = folder
        return phase_dataset

def create_level_folders(base_image_dir, dataset, target_resolution, levels):
    for i in range(0, levels+1):
        intermediate_resolution = target_resolution // 2**i 
        Path(os.path.join(base_image_dir, dataset+'_'+str(intermediate_resolution))).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(base_image_dir, dataset+'_'+str(intermediate_resolution), 'train')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(base_image_dir, dataset+'_'+str(intermediate_resolution), 'val')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(base_image_dir, dataset+'_'+str(intermediate_resolution), 'test')).mkdir(parents=True, exist_ok=True)

def center_crop(img, crop_left, crop_right, crop_top, crop_bottom):
    width, height = img.size
    left = crop_left
    right = width - crop_right
    top = crop_top
    bottom = height - crop_bottom
    return img.crop((left, top, right, bottom))


def create_haar_dataset(base_image_dir, dataset, target_resolution, levels, split):
    create_level_folders(base_image_dir, dataset, target_resolution, levels)

    haar_transform = InvertibleDownsampling2D(3, stride=2, method='cayley', init='haar', learnable=False)

    haar_level_ranges={}
    approx_level_ranges={}

    total_num_images = len([path for path in Path(os.path.join(base_image_dir, dataset)).iterdir()])
    phase_dataset = create_train_val_test_index_dict(total_num_images, split)

    for counter, img_file in tqdm(enumerate(sorted(os.listdir(os.path.join(base_image_dir, dataset))))):
        image = Image.open(os.path.join(base_image_dir, dataset, img_file))

        print('dimensions: (%d,%d)'%(image.size[0], image.size[1]))
        if image.size[0]!=178 and image.size[1]!=218:
            os.remove(os.path.join(base_image_dir, dataset, img_file))
            continue
            
        if dataset in ['celeba', 'celebA']:
            image = center_crop(image, 9, 9, 39, 19)

        assert image.size[0]==image.size[1], 'Image size is not square, revisit the data generation code. Image dimension: (%d, %d)' % (image.size[0], image.size[1])
        
        try:
            assert image.size[0] == target_resolution
        except AssertionError:
            print('image.size[0] is not equal to target resolution: %d - %d' % (image.size[0], target_resolution))

        if image.size[0] > target_resolution:
            image = image.resize((target_resolution, target_resolution))

        image = torch.from_numpy(np.array(image)).float().unsqueeze(0)
        image = normalise(image, value_range=[0, 255])
        image = image.permute(0, 3, 1, 2)

        save_file = os.path.join(base_image_dir, dataset+'_'+str(target_resolution), phase_dataset[counter], img_file.split('.')[0]+'.png')
        image_grid = make_grid(image, nrow=1, normalize=False)
        save_image(tensor=image_grid, fp=save_file)

        if 0 in approx_level_ranges.keys():
            approx_level_ranges[0].append([image.min(), image.max()])
        else:
            approx_level_ranges[0] = [[image.min(), image.max()]]

        for i in range(1, levels+1):
            intermediate_resolution = target_resolution // 2**i #intermediate resolution
            haar_image = haar_transform(image)
            if i in haar_level_ranges.keys():
                haar_level_ranges[i].append([haar_image.min(), haar_image.max()])
            else:
                haar_level_ranges[i] = [[haar_image.min(), haar_image.max()]]

            permuted_haar_image = permute_channels(haar_image)
            image = permuted_haar_image[:, :3, :, :]

            if i in approx_level_ranges.keys():
                approx_level_ranges[i].append([image.min(), image.max()])
            else:
                approx_level_ranges[i] = [[image.min(), image.max()]]

            save_file = os.path.join(base_image_dir, dataset+'_'+str(intermediate_resolution), phase_dataset[counter], img_file.split('.')[0]+'.npy')
            np.save(file=save_file, arr=np.squeeze(image, axis=0))

        counter+=1


    print('----------- Haar Transform ranges ---------')
    for level in haar_level_ranges.keys():
        min_maxs = np.array(haar_level_ranges[level])
        minimum, maximum= np.mean(min_maxs[:, 0]), np.mean(min_maxs[:, 1])
        print('level: %d - min: %.3f - max: %.3f' % (level, minimum, maximum))

    print('------- Approximation coefficient ranges --------')
    for level in approx_level_ranges.keys():
        min_maxs = np.array(approx_level_ranges[level])
        minimum, maximum= np.mean(min_maxs[:, 0]), np.mean(min_maxs[:, 1])
        print('level: %d - min: %.3f - max: %.3f' % (level, minimum, maximum))

def create_dataset(config):
    base_image_dir = config.data.base_dir
    dataset = config.data.dataset
    target_resolution = config.data.target_resolution
    levels = config.data.max_haar_depth
    split = config.data.split
    create_haar_dataset(base_image_dir, dataset, target_resolution, levels, split)