from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, DistributedSampler
import pytorch_lightning as pl
import torch.distributed as dist
import data
import dataset_interface
from data import dataset_interface
import argparse
import helper
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume

class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class WaveletDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=48, 
                 ray_base_folder=None,
                 num_workers=8,
                 use_texturefield_id=False,
                 return_geo_latent=False,
                 return_sdf=True,
                 **kwargs):
        super().__init__()
        self.dataset_path = ['ShapeNet_V2']
        self.wavelet_transform = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_texturefield_id = use_texturefield_id
        self.return_geo_latent = return_geo_latent
        self.return_sdf = return_sdf
        self.args = Config(**kwargs)
        self.ray_base_folder = ray_base_folder

        #dist.init_process_group(backend='nccl')
        #self.local_rank = dist.get_rank()
        #self.world_size = dist.get_world_size()
        #print(self.world_size)
        #self.local_batch_size = self.batch_size // self.world_size
        for key, value in kwargs.items():
            setattr(self, key, value)

    def train_dataloader(self,data_args):
        self.train_data = dataset_interface.Dataset_Build3D_S3(
            self.dataset_path, split="train", n_px=224, 
            wavelet_transform=self.wavelet_transform, 
            image_transform=None, base_rep="Wavelet",  
            return_reps=["Wavelet"], args=data_args
        )

        return self.train_data
        #train_sampler = DistributedSampler(self.train_data, num_replicas=self.world_size, rank=self.local_rank)
        #return DataLoader(
         #   self.train_data, batch_size=self.local_batch_size, shuffle=False, 
          #  num_workers=self.num_workers, drop_last=True, 
           # collate_fn=dataset_interface.my_collate_fn     #,sampler=train_sampler
        #)

    def val_dataloader(self,data_args):
        self.val_data = dataset_interface.Dataset_Build3D_S3(
            self.dataset_path, split="val", n_px=224, 
            image_transform=None, base_rep="Wavelet",  
            return_reps=["Wavelet", "SDF_GRID_256"], args=data_args
        )
        #val_sampler = DistributedSampler(self.val_data, num_replicas=self.world_size, rank=self.local_rank)
        #return DataLoader(
         #   self.val_data, batch_size=self.local_batch_size, shuffle=False, 
          #  num_workers=self.num_workers, drop_last=False, 
           # collate_fn=dataset_interface.my_collate_fn     #,sampler=val_sampler
        #)
        return self.val_data

    def test_dataloader(self):
        self.test_data = dataset_interface.Dataset_Build3D_S3(
            self.dataset_path, split="val", n_px=224, 
            image_transform=None, base_rep="Wavelet",  
            return_reps=["Wavelet", "SDF_GRID_256"], args=self.args
        )
        test_sampler = DistributedSampler(self.test_data, num_replicas=self.world_size, rank=self.local_rank)
        return DataLoader(
            self.test_data, batch_size=self.local_batch_size, shuffle=False, 
            num_workers=self.num_workers, drop_last=False, 
            collate_fn=dataset_interface.my_collate_fn    #,sampler=test_sampler
        )

    

    

# Assuming W is your WaveletDataModule instance and you have a train_dataloader method or attribute
#W = WaveletDataModule()
#train_dataload = W.train_dataloader()

# Print the length of the dataloader (number of batches)
#print(f'Number of batches in train_dataloader: {len(train_dataload)}')

# Get and inspect the first batch
#for batch in train_dataload:
 #   print(f'First batch: {batch}')
  #  break  # Break after the first batch to avoid printing everything



