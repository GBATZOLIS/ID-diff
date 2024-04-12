from lightning_modules.utils import create_lightning_module
from lightning_data_modules.utils import create_lightning_datamodule
from tqdm import tqdm
import os
import torch 
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

def max_pairwise_L2_distance(batch):
    num_images = batch.size(0)
    max_distance = float("-inf")
    for i in range(num_images):
      for j in range(i+1, num_images):
        distance = torch.norm(batch[i]-batch[j], p=2)
        if distance > max_distance:
          max_distance = distance
    return max_distance


def compute_dataset_statistics(config):
  if config.data.dataset=='celebA':
    mean_save_dir = os.path.join(config.data.base_dir, 'datasets_mean', config.data.dataset+'_'+str(config.data.image_size))
    Path(mean_save_dir).mkdir(parents=True, exist_ok=True)

    config.training.batch_size = 128
    DataModule = create_lightning_datamodule(config)
    DataModule.setup()
    train_dataloader = DataModule.train_dataloader()

    LightningModule = create_lightning_module(config).to('cuda:0')


    with torch.no_grad():
      total_sum = None
      total_num_images = 0
      max_val = float('-inf')
      min_val = float('inf')
      #max_distance = float('-inf')
      for i, batch in tqdm(enumerate(train_dataloader)):
          hf = LightningModule.get_hf_coefficients(batch.to('cuda:0'))
          
          #calculate max pairwise distance
          #max_batch_pairwise_distance = max_pairwise_L2_distance(hf)
          #if max_batch_pairwise_distance > max_distance:
          #  max_distance = max_batch_pairwise_distance

          if hf.min() < min_val:
            min_val = hf.min()
          if hf.max() > max_val:
            max_val = hf.max()

          num_images = hf.size(0)
          total_num_images += num_images
          batch_sum = torch.sum(hf, dim=0)

          if total_sum is None:
            total_sum = batch_sum
          else:
            total_sum += batch_sum
    
    #print('Max pairwise distance: %.4f' % max_distance)

    print('range: [%.5f, %.5f]' % (min_val, max_val))
    print('total_num_images: %d' % total_num_images)
    mean = total_sum / total_num_images
    mean = mean.cpu()
    print(mean.size())
    
    torch.save(mean, f=os.path.join(mean_save_dir, 'mean.pt'))

    mean = mean.numpy().flatten()

    print('Maximum mean value: ', np.amax(mean))
    print('Minimum mean value: ', np.amin(mean))

    plt.figure()
    plt.title('Mean values histogram')
    _ = plt.hist(mean, bins='auto')
    plt.savefig(os.path.join(mean_save_dir, 'mean_histogram.png'))
  
  elif config.data.dataset == 'mri_to_pet':
    dataset_info_dir = os.path.join(config.data.base_dir, 'datasets_info', config.data.dataset)
    Path(dataset_info_dir).mkdir(parents=True, exist_ok=True)
    config.training.batch_size = 1
    DataModule = create_lightning_datamodule(config)
    DataModule.setup()
    train_dataloader = DataModule.train_dataloader()

    info = {'min_vals':{'mri':[], 'pet':[]}, 'max_vals':{'mri':[], 'pet':[]}, 'ranges':{'mri':[], 'pet':[]}}

    for i, batch in tqdm(enumerate(train_dataloader)):
      mri, pet = batch
      mri_min, mri_max = torch.min(mri).item(), torch.max(mri).item()
      pet_min, pet_max = torch.min(pet).item(), torch.max(pet).item()

      info['min_vals']['mri'].append(mri_min)
      info['min_vals']['pet'].append(pet_min)

      info['max_vals']['mri'].append(mri_max)
      info['max_vals']['pet'].append(pet_max)

      info['ranges']['mri'].append(mri_max-mri_min)
      info['ranges']['pet'].append(pet_max-pet_min)
    
    for quantity in info.keys():
      for modality in info[quantity].keys():
        plt.figure()
        plt.title('%s %s histogram' % (modality, quantity))
        _ = plt.hist(info[quantity][modality], bins='auto')
        plt.savefig(os.path.join(dataset_info_dir, '%s-%s-histogram.png' % (modality, quantity)))
    
    def get_max_value_until_threshold(vals, threshold):
      below_threshold_vals = []
      above_threshold_vals = []
      for val in vals:
        if val<=threshold:
          below_threshold_vals.append(val)
        elif val>threshold:
          above_threshold_vals.append(val)
      return below_threshold_vals, above_threshold_vals
    

    threshold = 1e4
    below_threshold_vals, above_threshold_vals = get_max_value_until_threshold(info['max_vals']['pet'], threshold)

    print('Dataset Info related to the threshold: %d' % threshold)
    print('Num of max values below %d: %d'% (threshold, len(below_threshold_vals)))
    print('Maximum below threshold maximum value: %.3f' % max(below_threshold_vals))
    print('Minimum below threshold maximum value: %.7f' % min(below_threshold_vals))

    print('Num of max values above %d: %d'% (threshold, len(above_threshold_vals)))
    print('Maximum above threshold maximum value: %.3f' % max(above_threshold_vals))
    print('Minimum above threshold maximum value: %.3f' % min(above_threshold_vals))

    '''
    plt.figure()
    plt.title('%s %s histogram' % ('pet', 'max_vals'))
    _ = plt.hist(below_threshold_vals, bins='auto')
    plt.savefig(os.path.join(dataset_info_dir, '%s-%s-histogram-%d.png' % ('pet', 'max_vals', threshold)))
    '''
    plt.figure()
    plt.title('Number of scans with max value below threshold')
    plt.plot(np.linspace(0, 1e5, num=100), [len(get_max_value_until_threshold(info['max_vals']['pet'], threshold)[0]) for threshold in np.linspace(0, 1e5, num=100)])
    plt.savefig(os.path.join(dataset_info_dir, 'Number of scans with max value below threshold.png'))



    #create a paired video of the outliers - inspect those scans
    def convert_to_3D(x):
        if len(x.shape[1:]) == 3:
            x = torch.swapaxes(x, 1, -1).unsqueeze(1)
            print(x.size())
            return x
        elif len(x.shape[1:]) == 4:
            return x
        else:
            raise NotImplementedError('x dimensionality is not supported.')
    
    def normalise(c, value_range=None):
      x = c.clone()
      if value_range is None:
          x -= x.min()
          x /= x.max()
      else:
          x -= value_range[0]
          x /= value_range[1]
      return x

    def generate_paired_video(writer, Y, I, dim, batch_idx, mri_max_value, pet_max_value):
        #dim: the sliced dimension (choices: 1,2,3)
        B = Y.size(0)
        raw_length = 2

        frames = Y.size(dim+1)
        video_grid = []
        for frame in range(frames):
            if dim==1:
                dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[3], I.shape[4]])).type_as(Y)
            elif dim==2:
                dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[2], I.shape[4]])).type_as(Y)
            elif dim==3:
                dim_cut = torch.zeros(tuple([B*raw_length, 1, I.shape[2], I.shape[3]])).type_as(Y)

            for i in range(B):
                if dim==1:
                    dim_cut[i*raw_length] = normalise(Y[i, 0, frame, :, :]).unsqueeze(0)
                    dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, frame, :, :]).unsqueeze(0)
                elif dim==2:
                    dim_cut[i*raw_length] = normalise(Y[i, 0, :, frame, :]).unsqueeze(0)
                    dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, :, frame, :]).unsqueeze(0)
                elif dim==3:
                    dim_cut[i*raw_length] = normalise(Y[i, 0, :, :, frame]).unsqueeze(0)
                    dim_cut[(i+1)*raw_length-1] = normalise(I[i, 0, :, :, frame]).unsqueeze(0)

            grid_cut = make_grid(tensor=dim_cut, nrow=raw_length, normalize=False)
            video_grid.append(grid_cut)

        video_grid = torch.stack(video_grid, dim=0).unsqueeze(0)
        str_title = 'paired_video_batch_%d_dim_%d_mri_max_value_%.5f_max_pet_value_%.5f' % (batch_idx, dim, mri_max_value, pet_max_value)
        writer.add_video(str_title, video_grid)

    writer = SummaryWriter("mri_to_pet_inspection")
    under_1e3 = 0
    above_5e5 = 0
    for i, batch in tqdm(enumerate(train_dataloader)):
      mri, pet = batch
      mri_min, mri_max = torch.min(mri).item(), torch.max(mri).item()
      pet_min, pet_max = torch.min(pet).item(), torch.max(pet).item()
      
      if pet_max < 1e3 and under_1e3<10:
        generate_paired_video(writer, mri, pet, 3, i, mri_max, pet_max)
        under_1e3+=1
      
      if pet_max > 5e5 and above_5e5<10:
        generate_paired_video(writer, mri, pet, 3, i, mri_max, pet_max)
        above_5e5+=1