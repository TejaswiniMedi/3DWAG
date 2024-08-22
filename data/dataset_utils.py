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
from cloudpathlib import S3Path
import numpy as np

def aug_to_pc_func(aug_name, pc):
    if "_" not in aug_name:
        return pc

    #print(aug_name.split("_"))
    axis = aug_name.split("_")[1]
    angle = aug_name.split("_")[2]

    f_map = {
            "x":rotate_point_cloud_x,
            "y": rotate_point_cloud_y,
            "z": rotate_point_cloud_z
        }

    points = f_map[axis](pc, int(angle))   

    return points

def aug_to_voxel_func(aug_name, voxel):
    if "_" not in aug_name:
        return voxel

    #print(aug_name.split("_"))
    axis = aug_name.split("_")[1]
    angle = aug_name.split("_")[2]

    f_map = {
            "x":rotate_sdf_x,
            "y": rotate_sdf_y,
            "z": rotate_sdf_z
        }

    voxel = f_map[axis](voxel, int(angle))   

    return voxel

def rotate_sdf_x(sdf, angle):
    """
    Rotate an SDF grid around the x-axis by 90, 180, 270, or 360 degrees.
    
    :param sdf: 3D numpy array representing the SDF grid.
    :param angle: Angle of rotation (must be 90, 180, 270, or 360).
    :return: Rotated SDF as a 3D numpy array.
    """
    if angle not in [90, 180, 270, 360]:
        raise ValueError("Angle must be 90, 180, 270, or 360.")
    
    # No rotation
    if angle == 360:
        return sdf
    
    # Rotate 90 degrees around the x-axis
    if angle == 90:
        rotated_sdf = np.transpose(sdf, (0, 2, 1))  # Swap y and z axes
        rotated_sdf = np.flip(rotated_sdf, axis=2)  # Reverse the new z-axis
    # Rotate 180 degrees around the x-axis
    elif angle == 180:
        rotated_sdf = np.flip(sdf, axis=1)  # Flip y-axis
        rotated_sdf = np.flip(rotated_sdf, axis=2)  # Flip z-axis
    # Rotate 270 degrees around the x-axis
    elif angle == 270:
        rotated_sdf = np.transpose(sdf, (0, 2, 1))  # Swap y and z axes
        rotated_sdf = np.flip(rotated_sdf, axis=1)  # Reverse the new y-axis
    
    return rotated_sdf

def rotate_sdf_y(sdf, angle):
    """
    Rotate an SDF grid around the y-axis by 90, 180, 270, or 360 degrees.
    
    :param sdf: 3D numpy array representing the SDF grid.
    :param angle: Angle of rotation (must be 90, 180, 270, or 360).
    :return: Rotated SDF as a 3D numpy array.
    """
    if angle not in [90, 180, 270, 360]:
        raise ValueError("Angle must be 90, 180, 270, or 360.")
    
    # No rotation
    if angle == 360:
        return sdf
    
    # Rotate 90 degrees around the y-axis
    if angle == 90:
        rotated_sdf = np.transpose(sdf, (2, 1, 0))  # Swap x and z axes
        rotated_sdf = np.flip(rotated_sdf, axis=2)  # Reverse the new z-axis
    # Rotate 180 degrees around the y-axis
    elif angle == 180:
        rotated_sdf = np.flip(sdf, axis=0)  # Flip x-axis
        rotated_sdf = np.flip(rotated_sdf, axis=2)  # Flip z-axis
    # Rotate 270 degrees around the y-axis
    elif angle == 270:
        rotated_sdf = np.transpose(sdf, (2, 1, 0))  # Swap x and z axes
        rotated_sdf = np.flip(rotated_sdf, axis=0)  # Reverse the new x-axis
    
    return rotated_sdf

def rotate_sdf_z(sdf, angle):
    """
    Rotate an SDF grid around the z-axis by 90, 180, 270, or 360 degrees.
    
    :param sdf: 3D numpy array representing the SDF grid.
    :param angle: Angle of rotation (must be 90, 180, 270, or 360).
    :return: Rotated SDF as a 3D numpy array.
    """
    if angle not in [90, 180, 270, 360]:
        raise ValueError("Angle must be 90, 180, 270, or 360.")
    
    # No rotation
    if angle == 360:
        return sdf
    
    # Rotate 90 degrees around the z-axis
    if angle == 90:
        rotated_sdf = np.transpose(sdf, (1, 0, 2))  # Swap x and y axes
        rotated_sdf = np.flip(rotated_sdf, axis=1)  # Reverse the new y-axis
    # Rotate 180 degrees around the z-axis
    elif angle == 180:
        rotated_sdf = np.flip(sdf, axis=0)  # Flip x-axis
        rotated_sdf = np.flip(rotated_sdf, axis=1)  # Flip y-axis
    # Rotate 270 degrees around the z-axis
    elif angle == 270:
        rotated_sdf = np.transpose(sdf, (1, 0, 2))  # Swap x and y axes
        rotated_sdf = np.flip(rotated_sdf, axis=0)  # Reverse the new x-axis
    
    return rotated_sdf


def rotate_point_cloud_x(points, angle):
    """
    Rotate a point cloud around the x-axis by 90, 180, 270, or 360 degrees.
    
    :param points: 2D numpy array representing the point cloud, shape (n, 3).
    :param angle: Angle of rotation (must be 90, 180, 270, or 360).
    :return: Rotated point cloud as a 2D numpy array.
    """
    rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_a, sin_a],
        [0, -sin_a, cos_a]
    ])
    return points @ rotation_matrix.T

def rotate_point_cloud_y(points, angle):
    """
    Rotate a point cloud around the y-axis by 90, 180, 270, or 360 degrees.
    
    :param points: 2D numpy array representing the point cloud, shape (n, 3).
    :param angle: Angle of rotation (must be 90, 180, 270, or 360).
    :return: Rotated point cloud as a 2D numpy array.
    """
    rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rotation_matrix = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
    return points @ rotation_matrix.T

def rotate_point_cloud_z(points, angle):
    """
    Rotate a point cloud around the z-axis by 90, 180, 270, or 360 degrees.
    
    :param points: 2D numpy array representing the point cloud, shape (n, 3).
    :param angle: Angle of rotation (must be 90, 180, 270, or 360).
    :return: Rotated point cloud as a 2D numpy array.
    """
    rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(rad), np.sin(rad)
    rotation_matrix = np.array([
        [cos_a, sin_a, 0],
        [-sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    return points @ rotation_matrix.T

def unit_variance_2(points):
    #print(points.max())
    centroid = points.mean(axis=0).reshape(1, 3)
    scale = points.flatten().std().reshape(1, 1)
  
    points =  (points - centroid) / scale
    #print(points.max())
    #raise "err"
    return points

def normalize_pointcloud(points):
    # Calculate the centroid of the points
    centroid = points.mean(axis=0)
    
    # Center the points around the origin
    points_centered = points - centroid
    
    # Find the maximum absolute value in the centered points
    max_abs_value = np.max(np.abs(points_centered))
    
    # Scale the points to fit within [-1, 1]
    points_normalized = points_centered / max_abs_value
    
    return points_normalized   


def obtain_s3_path(path, s3_bucket):
    return f's3://{s3_bucket}/{path}'

def check_exists(path, s3_bucket):
    s3_path = obtain_s3_path(path, s3_bucket)
    cp = S3Path(s3_path)
    return cp.is_file()

def remove_duplicate_fast(arr_1, arr_2, dim = 46):
    arr_1 = arr_1[:, 0] * dim * dim + arr_1[:, 1] * dim + arr_1[:, 2]
    arr_2 = arr_2[:, 0] * dim * dim + arr_2[:, 1] * dim + arr_2[:, 2]
    results = np.setdiff1d(arr_1, arr_2, assume_unique=True)
    results_output = []
    for i in range(3):
        results_output.append(results[:, None] % dim)
        results = results // dim
    results = np.concatenate(results_output[::-1], axis = 1) # reverse and concat
    return results

def remove_duplicate(arr_1, arr_2):
    combined = np.concatenate((arr_1, arr_2), axis = 0)
    uniques, counts = np.unique(combined, return_counts=True, axis=0)
    difference = uniques[counts == 1]

    return difference

def create_coordinates(resolution, space_range, channel_dim = 7):
    channels_samples = np.linspace(0, channel_dim - 1, channel_dim)
    dimensions_samples = np.linspace(space_range[0], space_range[1], resolution)
    ch, x, y, z = np.meshgrid(channels_samples, dimensions_samples, dimensions_samples, dimensions_samples)
    ch, x, y, z = ch[:, :, :, :, np.newaxis], x[:, :, :, :, np.newaxis], y[:, :, :, :, np.newaxis], z[:, :, :, :, np.newaxis]
    coordinates = np.concatenate((ch, x, y, z), axis=4)
    coordinates = coordinates.reshape((-1, 4))
    coordinates = torch.from_numpy(coordinates).long()
    return coordinates

def get_voxel_list(str_list):
    
    voxel_reps = []
    for str_ in str_list:
        voxel_reps.append(str_.split("_")[-1])
    
    return voxel_reps

def get_image_transform(image_transform, n_px=224):
    if image_transform == "dino":
        transform =  Compose([
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5]),
        ])
    elif image_transform == "canny_color_and_affine": 
        transform = Compose([
                                Resize(n_px, interpolation=Image.BICUBIC),
                                CenterCrop(n_px),
                                lambda image: image.convert("RGB"),
                                RandomCannyEdgeTransform(100, 200),
                                ColorJitter(0.5, 0.5, 0.5, 0.5), 
                                transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.5,1.2), fill=255),
                                ToTensor(),  
                                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                            ]) 
    elif image_transform == "color_and_affine": 
        transform = Compose([
                                Resize(n_px, interpolation=Image.BICUBIC),
                                CenterCrop(n_px),
                                lambda image: image.convert("RGB"),
                                ColorJitter(0.5, 0.5, 0.5, 0.5), 
                                transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.8,1.2), fill=255),
                                ToTensor(),  
                                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                            ])    
    else:    
        transform = Compose([
                                Resize(n_px, interpolation=Image.BICUBIC),
                                CenterCrop(n_px),
                                lambda image: image.convert("RGB"),
                                ToTensor(),
                                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                            ])
        
    return transform
        

def make_a_grid(occupancy_arr, color_numpy, voxel_resolution=32):
    voxel_resolution = int(voxel_resolution)
    cube_color = np.zeros([voxel_resolution + 1, voxel_resolution +1, voxel_resolution+1, 4]) 
    cube_color[occupancy_arr[:,0],occupancy_arr[:,1],occupancy_arr[:,2], 0:3] = color_numpy
    cube_color[occupancy_arr[:,0],occupancy_arr[:,1],occupancy_arr[:,2], 3] = 1
    return cube_color[:-1,:-1,:-1]

def make_a_voxel_grid(occupancy_arr, voxel_resolution=32):
    voxel_resolution = int(voxel_resolution)
    cube_color = np.zeros([voxel_resolution + 1, voxel_resolution +1, voxel_resolution+1, 1]) 
    cube_color[occupancy_arr[:,0],occupancy_arr[:,1],occupancy_arr[:,2], 0] = 1
    return cube_color[:-1,:-1,:-1]

def my_collate_fn(batch):
    batch =  list(filter(lambda x : x is not None, batch))
    
    if not batch:
        print("batch size is none")
        return None  # or raise an error
    
    return torch.utils.data.dataloader.default_collate(batch)
        
def normalize_coordinates(coords, shape):
    """
    Normalize grid coordinates to lie between -1 and 1.
    """
    norm_coords = 2.0 * (coords / np.array(shape) - 0.5)
    return norm_coords

def sample_fixed_number_near_surface(sdf, num_points=10000, threshold=0.1):
    """
    Sample a fixed number of points from an SDF grid where the absolute SDF value is less than the threshold.
    Args:
    - sdf (numpy.ndarray): The SDF grid.
    - num_points (int): The number of points to sample.
    - threshold (float): A value such that points with an absolute SDF value less than this are considered 'near the surface'.
    Returns:
    - numpy.ndarray: A 2D array where each row is the [x, y, z] coordinate of a sampled point.
    - numpy.ndarray: A 1D array of the SDF values corresponding to the sampled points.
    """
    # Find where the absolute SDF values are less than the threshold.
    near_surface = np.abs(sdf) < threshold

    # Extract the [x, y, z] coordinates of these points.
    coords = np.argwhere(near_surface)

    # If there are fewer than num_points near the surface, you might need to handle this
    # (e.g., by reducing num_points or by selecting all available points).
    if coords.shape[0] < num_points:
        raise ValueError("Not enough points near the surface. Reduce num_points or increase threshold.")

    # Randomly select 'num_points' of them.
    indices = np.random.choice(coords.shape[0], num_points, replace=False)
    selected_coords = coords[indices]
    
    # Normalize the coordinates.
    normalized_coords = normalize_coordinates(selected_coords, sdf.shape)

    # Extract the SDF values for these points.
    sdf_values = sdf[selected_coords[:, 0], selected_coords[:, 1], selected_coords[:, 2]]

    return normalized_coords, sdf_values

def uniform_sample_grid(sdf, num_points=10000):
    """
    Uniformly sample points across the entire SDF grid.
    Returns:
    - numpy.ndarray: A 2D array where each row is the [x, y, z] coordinate of a sampled point.
    - numpy.ndarray: A 1D array of the SDF values corresponding to the sampled points.
    """
    total_points = np.prod(sdf.shape)

    # Randomly select 'num_points' from the grid.
    flat_indices = np.random.choice(total_points, num_points, replace=False)
    
    # Convert flat indices to 3D indices.
    coords = np.unravel_index(flat_indices, sdf.shape)
    sampled_coords = np.vstack(coords).T
    
    # Normalize the coordinates.
    normalized_coords = normalize_coordinates(sampled_coords, sdf.shape)
    
    # Extract the SDF values for these points.
    sdf_values = sdf[sampled_coords[:, 0], sampled_coords[:, 1], sampled_coords[:, 2]]

    return normalized_coords, sdf_values

def mixture_sample_grid(sdf, num_points=10000, threshold=0.05, ratio=0.5):
    
    ### uniform sampling across the grid
    total_points = np.prod(sdf.shape)

    # Randomly select 'num_points' from the grid.
    flat_indices = np.random.choice(total_points, int(num_points*ratio), replace=False)
    
    # Convert flat indices to 3D indices.
    coords = np.unravel_index(flat_indices, sdf.shape)
    sampled_coords = np.vstack(coords).T
    
    
    ### sampling near the surface
    near_surface = np.abs(sdf) < threshold

    # Extract the [x, y, z] coordinates of these points.
    coords_ns = np.argwhere(near_surface)

    # If there are fewer than num_points near the surface, you might need to handle this
    # (e.g., by reducing num_points or by selecting all available points).
    if coords_ns.shape[0] < num_points//2:
        raise ValueError("Not enough points near the surface. Reduce num_points or increase threshold.")

    # Randomly select 'num_points' of them.
    indices_ns = np.random.choice(coords_ns.shape[0], int(num_points*(1-ratio)), replace=False)
    selected_coords_ns = coords_ns[indices_ns]
    
    #print(sampled_coords.shape, selected_coords_ns.shape)
    total_coords = np.concatenate([sampled_coords, selected_coords_ns], axis=0)
    # Normalize the coordinates.
    normalized_coords = normalize_coordinates(total_coords, sdf.shape)
    
    # Extract the SDF values for these points.
    sdf_values = sdf[total_coords[:, 0], total_coords[:, 1], total_coords[:, 2]]

    return normalized_coords, sdf_values

class CannyEdgeTransform():
    def __init__(self, low_threshold, high_threshold):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def __call__(self, x):
        # Convert the input tensor to a numpy array
        #x = x.numpy()
        try:
            x = np.asarray(x)
            x = x.astype(np.uint8)


            # Apply the Canny edge filter
            x = cv2.Canny(x, self.low_threshold, self.high_threshold)
            x = np.stack([x, x, x], axis=-1)

            x = (x - np.min(x)) / (np.max(x) - np.min(x))

            x = 1 - x
            # Convert the edge map into a PIL image
            x = Image.fromarray(np.uint8(x * 255))
            return x
        except:         
            return x
        
class RandomCannyEdgeTransform():

    
    def __init__(self, low_threshold, high_threshold, probability=0.5):
        self.probability = probability
        self.canny = CannyEdgeTransform(low_threshold, high_threshold)

    def __call__(self, x):
        if random.uniform(0, 1) < self.probability:
            x = self.canny(x)
        return x
    

class rotate_simple_axis():
    def __init__(self, axis=2):
        self.axis = axis

    def __call__(self, voxel_grid):
        angle = random.choice([90, 180, 270, 360])
        axis = self.axis
        
        if angle in [0, 360]:
            rotated_voxel_grid = voxel_grid
        elif angle == 90:
            if axis == 0:
                rotated_voxel_grid = np.swapaxes(voxel_grid, 1, 2)[:, ::-1, :]
            elif axis == 1:
                rotated_voxel_grid = np.swapaxes(voxel_grid, 0, 1)[::-1, :, :]
            else: # axis == 2
                rotated_voxel_grid = np.swapaxes(voxel_grid, 0, 2)[:, :, ::-1]
        elif angle == 180:
            if axis == 0:
                rotated_voxel_grid = voxel_grid[:, ::-1, :]
            elif axis == 1:
                rotated_voxel_grid = voxel_grid[:, ::-1, :]
            else: # axis == 2
                rotated_voxel_grid = voxel_grid[:, :, ::-1]
        elif angle == 270:
            if axis == 0:
                rotated_voxel_grid = np.swapaxes(voxel_grid, 1, 2)[:, :, ::-1]
            elif axis == 1:
                rotated_voxel_grid = np.swapaxes(voxel_grid, 0, 1)[:, ::-1, :]
            else: # axis == 2
                rotated_voxel_grid = np.swapaxes(voxel_grid, 0, 2)[::-1, :, :]
        
        return rotated_voxel_grid
    
class IdentityTransform:
    def __call__(self, x):
        return x
    
    
def get_voxel_transform(voxel_transform):
    
    if voxel_transform == "random_rotate_axis":
        transform =  Compose([
            rotate_simple_axis(axis=2)
        ])
    else:
        transform = Compose([
                             IdentityTransform()
                            ])
        
    return transform

def refill_and_sample(paths, num_samples):
    # Make a copy of the original array to use for refilling
    original_paths = paths[:]
    selected_paths = []

    while len(selected_paths) < num_samples:
        # If paths is empty, refill it with the original paths
        if not paths:
            paths = original_paths[:]
        
        # Determine how many samples we need to complete the request
        remaining_samples = num_samples - len(selected_paths)
        
        # Sample either the remaining needed or the remaining paths, whichever is smaller
        sample_size = min(remaining_samples, len(paths))
        sampled = random.sample(paths, sample_size)
        
        # Remove the sampled items from the paths array
        for item in sampled:
            paths.remove(item)
        
        # Add the sampled items to the selected paths
        selected_paths.extend(sampled)

    return selected_paths
    

#if __name__ == "__main__":
  #  x = torch.randn([2048, 3])
   # x = unit_variance_2(x)
    #print(x.shape, x.min())