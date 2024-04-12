import torch.utils.data as data
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from . import utils
import glob
import os
from PIL import Image

class HaarDecomposedDataset(data.Dataset):
  def __init__(self,  config, phase='train'):
    self.dataset = config.data.dataset
    self.level = config.data.level #target resolution - level 0.
    if config.data.level == 0: #data are saved as png files.
      self.image_files = glob.glob(os.path.join(config.data.base_dir, config.data.dataset+'_'+str(config.data.image_size), phase, '*.png'))
    elif config.data.level >= 1: #data are saved as numpy arrays to minimise the reconstruction.
      self.image_files = glob.glob(os.path.join(config.data.base_dir, config.data.dataset+'_'+str(config.data.image_size), phase, '*.npy'))
    else:
      raise Exception('Invalid haar level.')

    
    #preprocessing operations
    #self.random_flip = config.data.random_flip
  
  def __getitem__(self, index):
    if self.level==0:
      image = Image.open(self.image_files[index])
      image = torch.from_numpy(np.array(image)).float()
      image = image.permute(2, 0, 1)
      image /= 255
      return image
    else:
      image = np.load(self.image_files[index])
      image = torch.from_numpy(image).float()
      return image
        
  def __len__(self):
      """Return the total number of images."""
      return len(self.image_files)

@utils.register_lightning_datamodule(name='haar_multiscale')
class HaarDecomposedDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        #DataLoader arguments
        self.config = config
        self.train_workers = config.training.workers
        self.val_workers = config.eval.workers
        self.test_workers = config.eval.workers

        self.train_batch = config.training.batch_size
        self.val_batch = config.eval.batch_size
        self.test_batch = config.eval.batch_size

    def setup(self, stage=None): 
        self.train_dataset = HaarDecomposedDataset(self.config, phase='train')
        self.val_dataset = HaarDecomposedDataset(self.config, phase='val')
        self.test_dataset = HaarDecomposedDataset(self.config, phase='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.train_batch, shuffle=True, num_workers=self.train_workers) 
  
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.val_batch, shuffle=False, num_workers=self.val_workers) 
  
    def test_dataloader(self): 
        return DataLoader(self.test_dataset, batch_size = self.test_batch, shuffle=False, num_workers=self.test_workers) 


