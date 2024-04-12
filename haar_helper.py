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

def normalise_per_band(permuted_haar_image):
    normalised_image = permuted_haar_image.clone()
    for i in range(4):
        normalised_image[:, 3*i:3*(i+1), :, :] = normalise(permuted_haar_image[:, 3*i:3*(i+1), :, :])
    return normalised_image #normalised permuted haar transformed image

def create_supergrid(normalised_permuted_haar_images):
    haar_super_grid = []
    for i in range(normalised_permuted_haar_images.size(0)):
        shape = normalised_permuted_haar_images[i].shape
        haar_grid = make_grid(normalised_permuted_haar_images[i].reshape((-1, 3, shape[1], shape[2])), nrow=2)
        haar_super_grid.append(haar_grid)
    
    super_grid = make_grid(haar_super_grid, nrow=int(np.sqrt(normalised_permuted_haar_images.size(0))))
    return super_grid

def create_haar_dataset(base_image_dir, highest_resolution, target_resolution, levels, split):
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

    for i in range(0, levels+1):
        intermediate_resolution = target_resolution // 2**i 
        Path(os.path.join(base_image_dir, str(intermediate_resolution))).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(base_image_dir, str(intermediate_resolution), 'train')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(base_image_dir, str(intermediate_resolution), 'val')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(base_image_dir, str(intermediate_resolution), 'test')).mkdir(parents=True, exist_ok=True)

    haar_transform = InvertibleDownsampling2D(3, stride=2, method='cayley', init='haar', learnable=False)

    haar_level_ranges={}
    approx_level_ranges={}

    total_num_images = len(os.listdir(os.path.join(base_image_dir, 'resolution_'+str(highest_resolution))))
    phase_dataset = create_train_val_test_index_dict(total_num_images, split)

    for counter, img_file in tqdm(enumerate(sorted(os.listdir(os.path.join(base_image_dir, 'resolution_'+str(highest_resolution)))))):
        image = Image.open(os.path.join(base_image_dir, 'resolution_'+str(highest_resolution), img_file))
        assert image.size[0]==image.size[1], 'Image size is not square, revisit the data generation code.'

        if image.size[0] > target_resolution:
            image = image.resize((target_resolution,target_resolution))

        image = torch.from_numpy(np.array(image)).float().unsqueeze(0)
        image = normalise(image, value_range=[0, 255])
        image = image.permute(0, 3, 1, 2)

        save_file = os.path.join(base_image_dir, str(target_resolution), phase_dataset[counter], img_file.split('.')[0]+'.png')
        image_grid = make_grid(image, nrow=1, normalize=False)
        save_image(tensor=image_grid, fp=save_file)
        

        #loading correctly
        '''
        loaded_image = Image.open(save_file)
        loaded_image = torch.from_numpy(np.array(loaded_image)).float().unsqueeze(0)
        loaded_image = loaded_image.permute(0, 3, 1, 2)
        loaded_image = normalise(loaded_image, value_range=[0, 255])

        assert torch.mean(torch.abs(loaded_image - image)) == 0., 'reconstruction error is not zero.'
        print(torch.mean(torch.abs(loaded_image - image)))
        '''
        
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

            save_file = os.path.join(base_image_dir, str(intermediate_resolution), phase_dataset[counter], img_file.split('.')[0]+'.npy')
            np.save(file=save_file, arr=np.squeeze(image, axis=0))

            '''
            #print(image.max(), 2**i)
            #image_grid = make_grid(image, nrow=1, normalize=True, range=(0, 2**i))
            save_file = os.path.join(base_image_dir, str(intermediate_resolution), phase_dataset[counter], img_file.split('.')[0]+'.npy')
            #save_image(tensor=image_grid, fp=save_file)
            np.save(file=save_file, arr=np.squeeze(image, axis=0))

            loaded_image = np.load(save_file)
            loaded_image = torch.from_numpy(loaded_image).float().unsqueeze(0)
            #loaded_image = loaded_image.permute(0, 3, 1, 2)
            #loaded_image = 2**i*loaded_image
            print(torch.mean(torch.abs(loaded_image - image)))
            '''

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


#base_image_dir = '/Users/gbatz97/Desktop/ScoreBasedConditionalGeneration/datasets/celebaHQ'
#highest_resolution, target_resolution, levels = 1024, 64, 3
#create_haar_dataset(base_image_dir, highest_resolution, target_resolution, levels, split=[0.9, 0.05, 0.05])

##testing of the forward and inverse haar pipeline.
'''
haar_transform = InvertibleDownsampling2D(3, stride=2, method='cayley', init='haar', learnable=False)
save_file = '/Users/gbatz97/Desktop/MRI_to_PET/datasets/celebA/train/001527.jpg'
loaded_image = Image.open(save_file)
loaded_image = torch.from_numpy(np.array(loaded_image)).float().unsqueeze(0)
loaded_image = loaded_image.permute(0, 3, 1, 2)
loaded_image = normalise(loaded_image, value_range=[0, 255])
haar_image = haar_transform(loaded_image)
haar_image = permute_channels(haar_image)
haar_image = permute_channels(haar_image, forward=False)
haar_image = haar_transform.inverse(haar_image)
save_image(torch.clamp(haar_image, min=0, max=1), '/Users/gbatz97/Desktop/c.png')
'''