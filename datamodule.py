from torch.utils.data import DataLoader
import pytorch_lightning as pl
import data
from data import *
from data.dataset import *

class TextureDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=1, 
                 ray_base_folder=None,
                 num_workers=4,
                 use_texturefield_id=False,
                 return_geo_latent=False,
                 return_sdf = True):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ray_base_folder = ray_base_folder
        self.use_texturefield_id = use_texturefield_id
        self.return_geo_latent = return_geo_latent
        self.return_sdf = return_sdf

    def setup(self, stage=None):

        self.dataset_train = TextureFieldDataset(split='train',
                                                 ray_base_folder=self.ray_base_folder,
                                                 use_texturefield_id=self.use_texturefield_id,
                                                 return_geo_latent=self.return_geo_latent,
                                                 return_sdf = True)
        self.dataset_valid = TextureFieldDataset(split='valid',
                                                 ray_base_folder=self.ray_base_folder,
                                                 use_texturefield_id=self.use_texturefield_id,
                                                 return_geo_latent=self.return_geo_latent,
                                                 return_sdf = True)
        
        self.dataset_test = TextureFieldDataset(split='test',
                                                 ray_base_folder=self.ray_base_folder,
                                                 use_texturefield_id=self.use_texturefield_id,
                                                 return_geo_latent=self.return_geo_latent,
                                                 return_sdf = True)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def normalize_data(self, data):
        min_val = data.min()
        max_val = data.max()
        return (data - min_val) / (max_val - min_val) * 2 - 1