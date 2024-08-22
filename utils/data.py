import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
import os.path as osp
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from data.dataset import TextureFieldDataset

class TextureField(pl.LightningDataModule):
    def __init__(self,ray_base_folder='./ray_results', batch_size=1, num_workers=4, use_texturefield_id=False, return_geo_latent=False, return_sdf=True):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.ray_base_folder = ray_base_folder
        self.use_texturefield_id = use_texturefield_id
        self.return_geo_latent = return_geo_latent
        self.return_sdf = return_sdf

        # Initialize datasets as None
        self.dataset_train = None
        self.dataset_valid = None
        self.dataset_test = None

    def setup(self, stage=None):
        self.dataset_train = TextureFieldDataset(
            split='train',
            ray_base_folder=self.ray_base_folder,
            use_texturefield_id=self.use_texturefield_id,
            return_geo_latent=self.return_geo_latent,
            return_sdf=self.return_sdf
        )
        self.dataset_valid = TextureFieldDataset(
            split='valid',
            ray_base_folder=self.ray_base_folder,
            use_texturefield_id=self.use_texturefield_id,
            return_geo_latent=self.return_geo_latent,
            return_sdf=self.return_sdf
        )
        self.dataset_test = TextureFieldDataset(
            split='test',
            ray_base_folder=self.ray_base_folder,
            use_texturefield_id=self.use_texturefield_id,
            return_geo_latent=self.return_geo_latent,
            return_sdf=self.return_sdf
        )

    def build_dataset(self):
        # Return the datasets directly
        num_classes = None
        self.setup(stage='train')
        train_set = self.get_train_dataset()
        self.setup(stage='valid')
        #self.dataset_valid = self.setup(stage='valid')
        val_set = self.get_valid_dataset()
        return num_classes, train_set, val_set
    
    def get_train_dataset(self):
        # Return the train dataset
        return self.dataset_train

    def get_valid_dataset(self):
        # Return the validation dataset
        return self.dataset_valid

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset_valid, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)




def pil_loader(path):
    with open(path, 'rb') as f:
        img: PImage.Image = PImage.open(f).convert('RGB')
    return img


def print_aug(transform, label):
    print(f'Transform {label} = ')
    if hasattr(transform, 'transforms'):
        for t in transform.transforms:
            print(t)
    else:
        print(transform)
    print('---------------------------\n')
