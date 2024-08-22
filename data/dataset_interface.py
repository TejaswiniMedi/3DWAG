import csv
import os
import random
import json
import sys
import numpy as np
import glob
import logging
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, ColorJitter, RandomGrayscale, Grayscale
from skimage.measure import block_reduce 
import io 
import torch
#import cv2
import h5py
from data import dataset_utils
from data.dataset_utils import *
import tempfile
import pickle
    
from cloudpathlib import S3Path
import boto3
from boto3.s3.transfer import TransferConfig
import botocore
from concurrent.futures import ThreadPoolExecutor

def my_collate_fn(batch):
    batch =  list(filter(lambda x : x is not None, batch))    
    return torch.utils.data.dataloader.default_collate(batch)            
#################################################################################################################################################################################################################################################################### 

voxel_bucket_name = "build3d-voxels"

dataset_voxel_mapping = {
                    'ABC': 'ABC_voxels',
                    'BuildingNet': 'BuildingNet_voxels',
                    'Fusion': 'Fusion_voxels',
                    'ModelNet40': 'ModelNet40_voxels',
                    'Objaverse': 'Objaverse_voxels_fixed',
                    'ShapeNet_V2': 'ShapeNet_V2_voxels',
                    'Thingi10K': 'Thingi10K_voxels',
                    'Thingiverse': 'Thingiverse_voxels',
                    'Github': 'Github_voxels',
                    'House3d':'House3d_voxels',
                    'Fg3d': 'Fg3d_voxels',
                    'DeformingThings4D':'DeformingThings4D_voxels',
                    'Coma':'Coma_voxels',
                    'Abo':'Abo_voxels',
                    'Infinigen':'Infinigen_voxels',
                    'Smal':'Smal_voxels',
                    'Smpl':'Smpl_voxels',
                    'Toy4k':'Toy4k_voxels',
                    '3DFuture':'3DFuture_voxels'
                    }

voxel_dataset_mapping = {v: k for k, v in dataset_voxel_mapping.items()}

wavelet_bucket_name = "build3d-wavelets"
wavelet_aug_bucket_name = "build3d-wavelets-aug"

wavelet_dataset_mapping = {
                    '3DFuture':'3DFuture_wavelet_latents',
                    'ABC': 'ABC_wavelet_latents',
                    'Abo':'Abo_wavelet_latents',
                    'BuildingNet': 'BuildingNet_wavelet_latents',
                    'Coma':'Coma_wavelet_latents',
                    'DeformingThings4D':'DeformingThings4D_wavelet_latents',
                    'Fg3d': 'Fg3d_wavelet_latents',
                    'Fusion': 'Fusion_wavelet_latents',
                    'Github': 'Github_wavelet_latents',
                    'House3d':'House3d_wavelet_latents',
                    'Infinigen':'Infinigen_wavelet_latents',
                    'ModelNet40': 'ModelNet40_wavelet_latents',
                    'Objaverse': 'Objaverse_wavelet_latents_fixed',
                    'ShapeNet_V2': 'ShapeNet_V2_wavelet_latents',
                    'Smal':'Smal_wavelet_latents',
                    'Smpl':'Smpl_wavelet_latents',
                    'Thingi10K': 'Thingi10K_wavelet_latents', 
                    'Thingiverse': 'Thingiverse_wavelet_latents',
                    'Toy4k':'Toy4k_wavelet_latents',
                    }

dataset_wavelet_mapping = {v: k for k, v in wavelet_dataset_mapping.items()}

sdf_bucket_name = "build3d-sdfs"

dataset_sdf_mapping = {
                    'ABC': 'ABC_sdfs',
                    'BuildingNet': 'BuildingNet_sdfs',
                    'Fusion': 'Fusion_sdfs',
                    'ModelNet40': 'ModelNet40_sdfs',
                    'Objaverse': 'Objaverse_sdfs_fixed',
                    'ShapeNet_V2': 'ShapeNet_V2_sdfs',
                    'Thingi10K': 'Thingi10K_sdfs',
                    'Thingiverse': 'Thingiverse_sdfs',
                    'Github': 'Github_sdfs',
                    'Coma':'Coma_sdfs',
                    '3DFuture':'3DFuture_sdfs',
                    'Abo':'Abo_sdfs',
                    'DeformingThings4D':'DeformingThings4D_sdfs',
                    'Fg3d':'Fg3d_sdfs',
                    'House3d':'House3d_sdfs',
                    'Infinigen':'Infinigen_sdfs',
                    'Smal':'Smal_sdfs',
                    'Smpl':'Smpl_sdfs',
                    'Toy4k':'Toy4k_sdfs'
                    }


image_bucket_name_mapping = {
                    '3DFuture':'build3d-renders-3dfuture',
                    'ABC': 'build3d-renders-abc',
                    'Abo':'build3d-renders-abo',
                    'BuildingNet': 'build3d-renders-buildingnet',
                    'Coma':'build3d-renders-coma',
                    'DeformingThings4D':'build3d-renders-deformingthings4d',
                    'Fg3d': 'build3d-renders-fg3d',
                    'Fusion': 'build3d-renders-fusion',
                    'Github': 'build3d-renders-github',
                    'House3d':'build3d-renders-house3d',
                    'Infinigen':'build3d-renders-infinigen',
                    'ModelNet40': 'build3d-renders-modelnet40',
                    'Objaverse': 'build3d-renders-objaverse',
                    'ShapeNet_V2': 'build3d-renders-shapnet-v2',
                    'Smal':'build3d-renders-smal',
                    'Smpl':'build3d-renders-smpl',
                    'Thingi10K': 'build3d-renders-thingi10k', 
                    'Thingiverse': 'build3d-renders-thingiverse',
                    'Toy4k':'build3d-renders-toy4k',
                    }

pc_bucket_name = "build3d-pointclouds" 

dataset_pc_mapping = {
                    '3DFuture':'3DFuture_pointclouds',
                    'ABC': 'ABC_pointclouds',
                    'Abo':'Abo_pointclouds',
                    'BuildingNet': 'BuildingNet_pointclouds',
                    'Coma':'Coma_pointclouds',  
                    'DeformingThings4D':'DeformingThings4D_pointclouds',
                    'Fg3d': 'Fg3d_pointclouds',
                    'Fusion': 'Fusion_pointclouds',
                    'Github': 'Github_pointclouds',
                    'House3d':'House3d_pointclouds',
                    'Infinigen':'Infinigen_pointclouds',
                    'ModelNet40': 'ModelNet40_pointclouds',
                    'Objaverse': 'Objaverse_pointclouds',
                    'ShapeNet_V2': 'ShapeNet_V2_pointclouds',
                    'Smal':'Smal_pointclouds',
                    'Smpl':'Smpl_pointclouds',
                    'Thingi10K': 'Thingi10K_pointclouds',
                    'Thingiverse': 'Thingiverse_pointclouds',
                    'Toy4k':'Toy4k_pointclouds',
                    }

pc_dataset_mapping = {v: k for k, v in dataset_pc_mapping.items()}


ifeatures_bucket_name = "image-clips-features"

dataset_ifeatures_mapping = {
                    'ABC': 'ABC_image_features',
                    'BuildingNet': 'BuildingNet_image_features',
                    'Fusion': 'Fusion_image_features',
                    'ModelNet40': 'ModelNet40_image_features',
                    'Objaverse': 'Objaverse_image_features',
                    'ShapeNet_V2': 'ShapeNet_V2_image_features',
                    'Thingi10K': 'Thingi10K_image_features',
                    'Thingiverse': 'Thingiverse_image_features',
                    'Github': 'Github_image_features'
                    }

latent_bucket_name = "build3d-latent-code"

class Dataset_Build3D_S3(Dataset):
    def __init__(self, dataset_folders, split='train', n_px = 224, wavelet_transform=None, image_transform=None, voxel_transform=None, return_reps=["Voxel_32", "Wavelet"], base_rep="Wavelet", exp_name=None,  args=None, over_write=False, testing_cnt=None):
    
        # Attributes
        self.split = split
        self.dataset_folders = dataset_folders
        self.n_px = n_px
        self.return_reps = return_reps
        self.args = args
        
        if "latent" in return_reps:
            self.exp_name = args.exp_name
            self.wavelet_transform = wavelet_transform

        if "sdf_sample" in return_reps:
            self.sdf_points = args.sdf_points
            self.sdf_sample_type = args.sdf_sample_type
            self.sdf_res = args.sdf_res
         
        if "image" in return_reps:
            self.i_transform = get_image_transform(image_transform, n_px=self.n_px)
        
        if "Voxel" in return_reps:
            self.v_transform = get_voxel_transform(voxel_transform)
            
        if "Pointcloud" in return_reps:
            self.num_pc_points = args.num_pc_points
            

        if "Wavelet" in return_reps:
            if hasattr(self.args, 'use_batched_threshold') and self.args.use_batched_threshold:
                dim = 46 # hardcode for now
                self.highs_full_indices_last = create_coordinates(dim, space_range=(0, dim - 1),
                                                             channel_dim=1).detach().cpu().numpy()[:, 1:]
            self.wavelet_transform = wavelet_transform

        self.models = []

        self.file_paths = []

        ### get data file_name + saved path names according to args
        self.args.s3_wavelet_bucket = args.s3_bucket
        wavelet_filename, wavelet_paths_filename = self.get_data_filenames(args)

        for dataset_folder in dataset_folders:
            self.set_s3_config()

            wavelet_paths_list_path = os.path.join(wavelet_dataset_mapping[dataset_folder], wavelet_paths_filename)
            
            if not check_exists(wavelet_paths_list_path, self.args.s3_wavelet_bucket):
                paths = self.list_dataset_folder_s3(wavelet_filename, dataset_folder)

                ### save to avoid reload
                self.upload_save_files_list_path(wavelet_paths_filename, paths, wavelet_paths_list_path)

            ### reload the pickle
            paths = self.load_save_files_list_path(wavelet_paths_list_path)

            print(f"len of {dataset_folder}: {len(paths)}")
            
            # add back sorting
            paths.sort()

            total_len = len(paths)
            split_begin = int(total_len * 0.98)
            if split == "val":
                paths = paths[split_begin:]
                ## add validation count
                if hasattr(self.args, 'val_cnt') and self.args.val_cnt is not None:
                    if hasattr(self.args, 'use_even_val') and self.args.use_even_val:
                        every_n = max(1, int(len(paths) / self.args.val_cnt))
                        print(len(paths), every_n)
                        paths = paths[::every_n]
                    else:
                        paths = paths[:self.args.val_cnt]
            elif split == 'train':
                paths = paths[:split_begin]
                if self.args.train_mode == "finetune":
                    if args.ft_train_datasets is not None:
                        if dataset_folder not in args.ft_train_datasets:
                           paths = [] 
                    else:    
                        paths = refill_and_sample(paths, self.args.ft_train_number)


            self.file_paths = self.file_paths + paths

            print(f"Done Listing folder : {dataset_folder}.....")
    
        ### when have testing cnt
        if testing_cnt is not None:
            self.file_paths = self.file_paths[:testing_cnt]

        print("Size of the dataset: {}".format(len(self.file_paths)))

        ### reset to avoid pickle error
        self.reset_s3_config()
        self.models = self.file_paths
        #print(self.file_paths)
        #raise "err"

    def load_save_files_list_path(self, save_files_list_path):
        file = io.BytesIO()
        self.s3.Bucket(self.args.s3_wavelet_bucket).download_fileobj(save_files_list_path, file)
        file.seek(0)
        with file as fp:
            paths = pickle.load(fp)
        return paths

    def upload_save_files_list_path(self, path_file, paths, save_files_list_path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, path_file)
            with open(tmp_path, 'wb') as fp:
                pickle.dump(paths, fp)
            cp = S3Path(obtain_s3_path(save_files_list_path, self.args.s3_wavelet_bucket))
            cp.upload_from(tmp_path, force_overwrite_to_cloud=True)

    def list_dataset_folder_s3(self, data_file, dataset_folder):
        paths = []
        for object_summary in self.s3.Bucket(self.args.s3_wavelet_bucket).objects.filter(Prefix=dataset_folder):
            if object_summary.key.endswith(data_file):
                paths.append(object_summary.key)
        return paths

    def get_data_filenames(self, args):
        path_file = f'latent_{args.point_num}_{args.resolution}_{args.wavelet}_{args.padding_mode}_h5.pkl'
        data_file = f'latent_{args.point_num}_{args.resolution}_{args.wavelet}_{args.padding_mode}.h5df'

        if args.use_compact_indices:
            path_file = 'compact_' + path_file
            data_file = 'compact_' + data_file

        if hasattr(self.args, 'max_training_level') and self.args.max_training_level != self.args.max_depth:
            index = self.args.max_depth - max(1, self.args.max_training_level)
            path_file = path_file[:-4] + f'_{index}.pkl'
            data_file = data_file[:-5] + f'_{index}.h5df'

        return data_file, path_file

    def reset_s3_config(self):
        self.s3 = None
        self.config = None
        self.boto3_config = None

    def set_s3_config(self):
        if not hasattr(self.args, 'max_concurrency'):
            self.args.max_concurrency = 10000
        if not hasattr(self.args, 'multipart_size'):
            self.args.multipart_size = 1024 * 1024 * 8
        self.boto3_config = botocore.config.Config(max_pool_connections=self.args.max_concurrency,
                                                   s3={'max_queue_size': self.args.max_concurrency},
                                                   connect_timeout=180,
                                                   read_timeout=180,
                                                   retries={'max_attempts': 10})
        self.s3 = boto3.resource('s3', config=self.boto3_config, region_name='us-east-1')
        self.config = TransferConfig(max_concurrency=self.args.max_concurrency, max_io_queue=self.args.max_concurrency,
                                     multipart_threshold=self.args.multipart_size,
                                     multipart_chunksize=self.args.multipart_size,
                                     use_threads=True)

    def compute_mask(self, high_values_arr):
        high_values_last = high_values_arr[:, -7:]  ### get last 7 dimensions
        high_values_max = np.max(np.abs(high_values_last), axis=0)
        high_values_keep = np.abs(high_values_last) > high_values_max[None, :] * self.args.sample_threshold_ratio  ## keep those
        high_values_keep_mask = np.max(high_values_keep, axis=1) > 0  ## keep indices

        return high_values_keep_mask
    
    def load_arr_file(self, file, keys, file_type):
        assert file_type in ['h5df', 'npz']

        ### results
        results = {}
        if file_type == 'h5df':
            arr = h5py.File(file, 'r')

            for key in keys:
                data = np.array(arr[key][:])
                results[key] = data
        elif file_type == 'npz':
            arr = np.load(file)

            for key in keys:
                data = np.array(arr[key])
                results[key] = data
        else:
            raise Exception(f"Unknown file type for {file_type}....")

        return results

    def download_file_s3(self, s3_file_path, bucket, tmp_dir):

        ### set up the s3
        if self.s3 is None:
            self.set_s3_config()

        if self.args.use_local_storage:
            file = os.path.join(tmp_dir, os.path.basename(s3_file_path))
            self.s3.Bucket(bucket).download_file(s3_file_path, file, Config=self.config)
        else:
            file = io.BytesIO()
            self.s3.Bucket(bucket).download_fileobj(s3_file_path, file, Config=self.config)
            file.seek(0)

        return file

    def extract_empty_indices_mask(self, high_indices, high_values):
        high_values_mask = self.compute_mask(high_values_arr=high_values)  ## compute the mask
        # compute empty indices
        high_indices_non_empty = high_indices[high_values_mask, 1:]

        high_indices_empty = remove_duplicate(self.highs_full_indices_last, high_indices_non_empty)
        high_indices_empty_perm = np.random.permutation(high_indices_empty.shape[0])
        high_indices_empty = high_indices_empty[high_indices_empty_perm]
        high_indices_empty = high_indices_empty[:high_values_mask.shape[0]]
        return high_indices_empty, high_values_mask

    def wavelet_aug_apply(self, s3_bucket, s3_file_path):

        if self.wavelet_transform == "all" :
            aug_array = ["","_x_90", "_x_180", "_x_270", "_y_90", "_y_180", "_y_270", "_z_90", "_z_180", "_z_270"]
            aug = random.choice(aug_array)
            file_path = s3_file_path.split(".h5df")[0] + aug + ".h5df"
            s3_bucket = wavelet_aug_bucket_name
        elif self.wavelet_transform == "y_only" :
            aug_array = ["", "_y_90", "_y_180", "_y_270"]
            aug = random.choice(aug_array)
            file_path = s3_file_path.split(".h5df")[0] + aug + ".h5df"
            s3_bucket = wavelet_aug_bucket_name    
        else:
            file_path = s3_file_path
            s3_bucket = s3_bucket
            aug = None

        return s3_bucket, file_path, aug
         
    def __len__(self):
        return len(self.models)
    
    def __getitem__(self, idx):
        model_path = self.models[idx]
        model_id = model_path.split("/")[-2] 
        dataset_name = dataset_wavelet_mapping[model_path.split("/")[-3]]
        #print("input", model_id, dataset_name, model_path)
        data = {}
        tmp_dir =  tempfile.TemporaryDirectory()
        aug = None

        try: 
            return_reps = list(set(self.return_reps))

            if "Wavelet" in return_reps:

                if hasattr(self.args, 'use_dummpy_wavelet') and self.args.use_dummpy_wavelet:
                    data['low'] = np.zeros((1, 46, 46, 46), dtype=np.float32)
                    data['high_indices'] = np.ones((self.args.point_num, 3), dtype=np.int32) * 23
                    data_dim = ((2 ** 3) ** max(1, self.args.max_training_level)) - 1
                    data['high_values'] = np.zeros((self.args.point_num, data_dim), dtype=np.float32)
                else:
                    wavelet_aug_bucket_name, wavelet_file_path, aug = self.wavelet_aug_apply(self.args.s3_wavelet_bucket, model_path)
                    #print(wavelet_aug_bucket_name, wavelet_file_path, wavelet_aug)
                    ## download wavelet file
                    wavelet_file = self.download_file_s3(s3_file_path=wavelet_file_path,
                                                        bucket=wavelet_aug_bucket_name, tmp_dir=tmp_dir)

                    loaded_results = self.load_arr_file(file=wavelet_file, keys=['low', 'compact_indices', 'high_values'],
                                                        file_type='h5df')

                    data['low'] = loaded_results['low']
                    data['high_indices'] = loaded_results['compact_indices'].astype(np.int32)
                    data['high_values'] = loaded_results['high_values']

                    ## compute mask (masking unused values in the diffusion training) + negative samples
                    if hasattr(self.args, 'use_batched_threshold') and self.args.use_batched_threshold:
                        high_indices_empty, high_values_mask = self.extract_empty_indices_mask(
                            high_indices=data['high_indices'],
                            high_values=data['high_values'])

                        data['high_values_mask'] = high_values_mask
                        data['high_indices_empty'] = high_indices_empty

            if "latent" in return_reps:
                wavelet_aug_bucket_name, wavelet_file_path, aug = self.wavelet_aug_apply(self.args.s3_wavelet_bucket, model_path)
                s3_latent_file = os.path.join(self.exp_name, 'latent_code', wavelet_file_path)
                latent_file = self.download_file_s3(s3_file_path=s3_latent_file, bucket=latent_bucket_name, tmp_dir=tmp_dir)
                loaded_results = self.load_arr_file(file=latent_file, keys=['pre_quant', 'z_indice'],
                                                        file_type='h5df')
                data['pre_quant'] = loaded_results['pre_quant']
                data['z_indice'] = loaded_results['z_indice'].astype(np.int32)                                        
                #print(data['pre_quant'].shape, data['z_indice'].shape)
                #raise "err"                                        

            ## voxel processing 
            selected = next((item for item in return_reps if item.startswith('Voxel')), None) 
            if selected:
                folder_name = dataset_voxel_mapping[dataset_name]
                res = selected.split("_")[-1]
                
                s3_voxel_path = os.path.join(folder_name, model_id, "{}.npz".format(res))
                #print(s3_voxel_path)
                
                file = io.BytesIO()
                self.s3.Bucket(voxel_bucket_name).download_fileobj(s3_voxel_path, file)
                file.seek(0)
                
                voxel_data_interface = np.load(file)
                occupancy_arr, color_numpy = voxel_data_interface["occupancy_arr"], voxel_data_interface["color_numpy"]
                voxel_grid = make_a_voxel_grid(occupancy_arr, voxel_resolution=res)

                #print(aug)
                if aug is not None:
                    voxel_grid = np.expand_dims(aug_to_voxel_func(aug, voxel_grid[:,:,:,0]), axis=-1)
                    #print(voxel_grid.shape)

                data["Voxel_{}".format(res)] = voxel_grid.transpose(3,0,1,2).astype(np.float32)
                #print(voxel_grid.shape)
                if np.isnan(data["Voxel_{}".format(res)]).any():
                    raise "nan error in {}".format(model_id)
            
            selected = next((item for item in return_reps if item.startswith('SDF_GRID')), None) 
            if selected:
                sdf_folder_name = dataset_sdf_mapping[dataset_name]
                res = 256 ## hard coded for now: current_rep.split("_")[-1]
                current_res = selected.split("_")[-1]

                s3_sdf_path = os.path.join(sdf_folder_name, model_id, "{}.npz".format(res))
                wavelet_file = self.download_file_s3(s3_file_path=s3_sdf_path,
                                                        bucket=sdf_bucket_name, tmp_dir=tmp_dir)

                loaded_results = self.load_arr_file(file=wavelet_file, keys=['sdf_arr'], file_type='npz')
                
                down_sample = int(256/int(current_res))
                sdf_grid = loaded_results['sdf_arr'][::down_sample, ::down_sample, ::down_sample]
                
                data["SDF_GRID_{}".format(current_res)] = sdf_grid.astype(np.float32)
                
                if np.isnan(data["SDF_GRID_{}".format(current_res)]).any():
                    raise "SDF nan error in {}".format(model_id)
                    
            selected = next((item for item in return_reps if item.startswith('image')), None) 
            if selected:
                image_bucket_name = image_bucket_name_mapping[dataset_name]
            
                try:
                    img_idx = np.random.randint(self.args.max_images_num)
                    s3_img_file = os.path.join(model_id, 'img', f'{str(img_idx).zfill(3)}.png')
                    file = self.download_file_s3(s3_file_path=s3_img_file, bucket=image_bucket_name, tmp_dir=tmp_dir)
                except Exception as e:
                    if "github" in image_bucket_name:
                        img_idx = np.random.randint(self.args.max_images_num)
                        s3_img_file = os.path.join(model_id + ".", 'img', f'{str(img_idx).zfill(3)}.png')
                        file = self.download_file_s3(s3_file_path=s3_img_file, bucket=image_bucket_name, tmp_dir=tmp_dir)
                    else:
                        raise e 
    
                image = Image.open(file).convert('RGB')
                image = self.i_transform(image)
                data['images'] = image
                data['img_idx'] = img_idx
                if np.isnan(data['images']).any():
                    raise "nan error in {}".format(model_id)


            if "Pointcloud" in return_reps:
                folder_name = dataset_pc_mapping[dataset_name]
                
                s3_pc_path = os.path.join(folder_name, model_id, "25000.h5df")
                #print(s3_pc_path, pc_bucket_name)
                
                file = io.BytesIO()
                self.s3.Bucket(pc_bucket_name).download_fileobj(s3_pc_path, file)
                file.seek(0)
                
                with h5py.File(file, 'r') as f:
                    points = f['points'][:]
                    
                total_list = list(range(len(points)))
                random_index = random.sample(total_list, self.num_pc_points)
                
                points = points[random_index].astype(np.float32)
                if aug is not None:
                    points = aug_to_pc_func(aug, points)

                data['Pointcloud'] = normalize_pointcloud(points)

                if np.isnan(data['Pointcloud']).any():
                    raise "nan error in {}".format(model_id)
                #print(data['Pointcloud'].shape)

        

                #data["indices"] = indices
                

            data["idx"] = idx
            data["uid"] = model_id
            data["model_path"] = model_path
            data["dataset_name"] = dataset_name
            return data

        except Exception as e:
            #print(s3_latent_file, latent_bucket_name)
            try:
                new_index = (random.randint(0, self.__len__())) % self.__len__()
                #new_index = (idx + 1) % len(self.models)
                print(f"An error occurred: {e}")
                print("issue in {} dataset name {} model name {} ".format(model_path, dataset_name, model_id))
                return self.__getitem__(new_index) 
            except Exception as e2:
                print(f"An double error occurred: {e2}")
                return None
    
    def get_image_data(self, idx, data, tmp_dir, s3_pre_fix, image_bucket, dataset):
        try:
            if hasattr(self.args, "use_all_views") and self.args.use_all_views:

                if (
                    hasattr(self.args, "testing_views")
                    and self.args.testing_views is not None
                ):
                    img_idx = np.array(self.args.testing_views)
                else:
                    img_idx = np.random.choice(
                        np.arange(self.args.max_images_num),
                        self.args.input_view_cnt,
                        replace=False,
                    )

                images = self.multithread_images_download(
                    image_bucket, img_idx, s3_pre_fix, tmp_dir
                )
                # images = self.single_thread_images_download(image_bucket, img_idx, s3_pre_fix, tmp_dir)
            else:
                if (
                    hasattr(self.args, "testing_views")
                    and self.args.testing_views is not None
                ):
                    img_idx = np.array(self.args.testing_views)[0]
                else:
                    img_idx = np.random.randint(self.args.max_images_num)
                s3_img_file = os.path.join(
                    s3_pre_fix, "img", f"{str(img_idx).zfill(3)}.png"
                )

                img_file = self.download_file_s3(
                    s3_file_path=s3_img_file,
                    bucket=image_bucket,
                    tmp_dir=tmp_dir,
                )

                ## load images
                images = self.transform(dataset_utils.load_image(img_file))
        except Exception as e:
            if self.args.fill_blank_img:
                if hasattr(self.args, "use_all_views") and self.args.use_all_views:
                    img_idx = np.random.choice(
                        np.arange(self.args.max_images_num),
                        self.args.input_view_cnt,
                        replace=False,
                    )
                    images = [
                        self.transform(
                            Image.new(
                                "RGB",
                                (
                                    self.args.render_resolution,
                                    self.args.render_resolution,
                                ),
                                (0, 0, 0),
                            )
                        ).unsqueeze(0)
                        for _ in img_idx
                    ]
                    images = torch.cat(images, dim=0)
                else:
                    img_idx = np.random.randint(self.args.max_images_num)
                    images = self.transform(
                        Image.new(
                            "RGB",
                            (
                                self.args.render_resolution,
                                self.args.render_resolution,
                            ),
                            (0, 0, 0),
                        )
                    )
            else:
                raise e

        data["images"] = images.float()
        data["img_idx"] = img_idx


    def multithread_images_download(
        self, image_bucket, inputs_indices, s3_pre_fix, tmp_dir
    ):
        images = []
        ### multi-thread download
        with ThreadPoolExecutor(max_workers=self.args.input_view_cnt) as executor:
            futures = [
                executor.submit(
                    self.download_file_s3,
                    os.path.join(s3_pre_fix, "img", f"{str(index).zfill(3)}.png"),
                    image_bucket,
                    tmp_dir,
                )
                for index in inputs_indices
            ]
            for future in futures:
                image = dataset_utils.load_image(future.result())
                image = self.transform(image).unsqueeze(0)
                images.append(image)

        images = torch.cat(images, dim=0)
        return images

            
