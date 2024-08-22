import io
import os
import tempfile
import h5py
import gzip
import backoff
import boto3
from boto3.s3.transfer import TransferConfig
import botocore
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from data.utils import *


class TextureFieldDataset(Dataset):
    def __init__(
            self,
            split,
            ray_base_folder='./ray_results',
            bucket_name="build3d-textures",
            dataset_name="ShapeNet_V2",
            use_texturefield_id=True,
            return_geo_latent=True,
            img_res=224,
            N_points=5000,
            return_sdf = True
    ):
        super().__init__()
        self.split = split
        self.bucket_name  = bucket_name
        self.dataset_name = dataset_name
        self.image_bucket_name  = "build3d-renders-shapnet-v2" # TODO map texture dataset_name to renders bucket
        self.latent_bucket_name = "build3d-latent-code"
        self.sdf_bucket_name = "build3d-sdfs"
        self.return_sdf = return_sdf
        #self.latent_bucket_name = "texturegen-training-output-ue1"
        self.s3_region = "us-east-1"
        self.set_s3_config()
        print("Reading item ids...")
        if use_texturefield_id:
            print("DATASET TYPE: Only cars category")
            self.download_file_s3('texturefield_id_list_cleaned.pkl', 'texturegen-training-output', '.')
            all_item_id_list = load_pickle('./texturefield_id_list_cleaned.pkl')
        else:
            print(f"DATASET TYPE: {dataset_name}")
            folders_path = self.download_file_s3(f'{self.image_bucket_name}/folders-d1.txt.gz', 'build3d-metadata', '.')
            print(folders_path)
            with gzip.open(folders_path, 'rt') as file: 
                all_item_id_list = [line.strip()[:-1] for line in file]
            print(len(all_item_id_list))

        self.data_split(all_item_id_list)
        print(f'Dataset size ({split}): {len(self.item_id_list)}')
        # Setting
        self.N_points = N_points
        self.return_geo_latent = return_geo_latent
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert a NumPy image to a PyTorch tensor
            transforms.Resize((img_res, img_res), antialias=False),  # Resize to 224x224
        ])

    def data_split(self, all_list):
        total_items = len(all_list)
        indices = np.arange(total_items)   
        # Determine the sizes of each split
        train_size = int(total_items * 0.90)
        valid_size = int(total_items * 0.05)
        test_size = total_items - train_size - valid_size
        # Shuffle the indices to ensure random splitting
        np.random.shuffle(indices) 
        # Get the indices for each split
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]
        
        if self.split == 'train':
            self.item_id_list = [all_list[i] for i in train_indices]
        elif self.split == 'valid':
            self.item_id_list = [all_list[i] for i in valid_indices]
        elif self.split == 'test':
            self.item_id_list = [all_list[i] for i in test_indices]
        else:
            raise ValueError("split must be 'train', 'valid', or 'test'.")

    def __len__(self):
        return len(self.item_id_list)

    def __getitem__(self, idx):
        idx = idx % len(self.item_id_list)
        data = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            # try:
            # return shape ID
            id = self.item_id_list[idx]
            data['id'] = id
            # return point cloud
            xyz, col = self.get_pc(id, tmp_dir)
            data['points'] = torch.tensor(xyz, dtype=torch.float32).permute(1, 0)
            data['colors'] = torch.tensor(col, dtype=torch.float32).permute(1, 0)
            # return shape latent code
            if self.return_geo_latent:
                latent = self.get_latent(id, tmp_dir)
                data['latent'] = latent
            elif self.return_sdf:
                sdf = self.get_sdf(id)
                data['sdf'] = sdf
        return data

    def set_s3_config(self):

        max_concurrency = 10000
        multipart_size = 1024 * 1024 * 8
        self.boto3_config = botocore.config.Config(
            max_pool_connections=max_concurrency,
            s3={'max_queue_size': max_concurrency}
        )
        self.s3 = boto3.resource('s3', config=self.boto3_config, region_name=self.s3_region)
        self.config = TransferConfig(
            max_concurrency=max_concurrency,
            max_io_queue=max_concurrency,
            multipart_threshold=multipart_size,
            multipart_chunksize=multipart_size,
            use_threads=True
        )

    # Retry on: botocore.exceptions.ClientError: An error occurred (500) when calling the HeadObject operation (reached max retries: 4)
    @backoff.on_exception(backoff.expo, botocore.exceptions.ClientError, max_time=120)
    def get_file_s3(self, s3_file_path, bucket):
        try:
            s3_client = get_s3_client('s3')
            s3_response_object = s3_client.get_object(Bucket=bucket, Key=s3_file_path)
            object_content = s3_response_object['Body'].read()
        except botocore.exceptions.ClientError:
            print(f"Error downloading '{bucket}/{s3_file_path}'")
            raise  # re-raise last exception
        return object_content

    # Retry on: botocore.exceptions.ClientError: An error occurred (500) when calling the HeadObject operation (reached max retries: 4)
    @backoff.on_exception(backoff.expo, botocore.exceptions.ClientError, max_time=120)
    def download_file_s3(self, s3_file_path, bucket, tmp_dir):
        try:
            file = os.path.join(tmp_dir, os.path.basename(s3_file_path))
            self.s3.Bucket(bucket).download_file(s3_file_path, file, Config=self.config)
        except botocore.exceptions.ClientError:
            print(f"Error downloading '{bucket}/{s3_file_path}'")
            raise  # re-raise last exception
        return file
    
    def get_pc(self, id, tmp_dir):
        s3_pc_path = f"{self.dataset_name}_textures/{id}/pcd.npz"
        npy_data = self.get_file_s3(s3_pc_path, self.bucket_name)
        pc = np.load(io.BytesIO(npy_data))
        # only keep N points randomly
        N_points = self.N_points # TODO set this as config
        pc_size = len(pc['xyz'])
        indices = np.random.choice(pc_size, N_points, replace=False)
        return pc['xyz'][indices, :], pc['color'][indices, :]
    
    

    def get_image(self, id):
        s3_img_path = f"{id}/img/000.png"
        #print(s3_img_path)
        png_data = self.get_file_s3(s3_img_path, self.image_bucket_name)
        image = Image.open(io.BytesIO(png_data)).convert('RGB')
        image = np.asarray(image)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        else:
            image = image.astype(np.float32)
        if self.image_transform is not None:
            image = self.image_transform(image)
        return image
    
    def get_sdf(self,id):
        sdf_path = "ShapeNet_V2_sdfs"
        s3_sdf_path = f"{sdf_path}/{id}/64.npz"
        data = self.get_file_s3(s3_sdf_path, self.sdf_bucket_name)
        data = np.load(io.BytesIO(data))
        sdf_data = torch.tensor(np.array(data['sdf_arr']))
        return sdf_data

    def get_latent(self, id):
        s3_latent_folder = "Wave_Geometry_Net_all_Wavelet_General_Encoder_Down_2_Wavelet_General_Decoder_Up_2_original_4_1024_1_0.25_256_bior6.8_constant_2_2_2_e_r_0_d_r_0_ema_True_all_batched_threshold_use_sample_training_bf16_1.0_1/latent_code/ShapeNet_V2_wavelet_latents"
        s3_latent_path = f"{s3_latent_folder}/{id}/compact_latent_16384_256_bior6.8_constant_1.h5df"
        data = self.get_file_s3(s3_latent_path, self.latent_bucket_name)
        latent_dataset = h5py.File(io.BytesIO(data), 'r')
        latent = torch.tensor(np.array(latent_dataset['pre_quant']))
        return latent