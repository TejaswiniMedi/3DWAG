import functools
import boto3
import pickle 
#from cloudpathlib import S3Path
import numpy as np
#import matplotlib.pyplot as plt
import yaml

@functools.lru_cache()
def get_s3_client(*args, **kwargs):
    """
    Returns a reusable boto3 client.
    """
    return boto3.client(*args, **kwargs)

def s3_path_exists(bucket_name, s3_path):
    full_path = f"s3://{bucket_name}/{s3_path}"
    path = S3Path(full_path)
    print(path)
    return path.exists()


def download_from_s3(s3_uri, local_file_path):
    # Parse the S3 URI
    bucket_name = s3_uri.split('/')[2]
    s3_key = '/'.join(s3_uri.split('/')[3:])
    # Create a boto3 client
    s3 = boto3.client('s3')
    # Download the file
    s3.download_file(bucket_name, s3_key, local_file_path)   
    
def list_folders_s3(bucket_name, prefix):
    """
    List all folders within a specific folder in an S3 bucket.
    Args:
    bucket_name (str): Name of the S3 bucket.
    prefix (str): Prefix within the bucket to list folders from.
    Returns:
    List[str]: A list of folder names within the specified prefix.
    """
    s3_client = boto3.client('s3')
    paginator = s3_client.get_paginator('list_objects_v2')
    result = []
    # Ensure the prefix ends with a '/'
    if not prefix.endswith('/'):
        prefix += '/'
    limit_pages = 1000
    # Iterate through pages of results
    for page_num, page in enumerate(paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter='/')):
        if page_num >= limit_pages:
            break
        # 'CommonPrefixes' contains the folder paths
        for folder in page.get('CommonPrefixes', []):
            folder_name = folder['Prefix'].split('/')[1]
            result.append(folder_name)
    return result

def get_last_modified_file(bucket_name, prefix):
    # Initialize S3 client
    s3 = boto3.client('s3')
    # Get the list of files
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix) 
    # Check if the bucket is empty
    if 'Contents' not in response:
        return None
    # Extract the files and sort by last modified date
    files = response['Contents']
    last_modified_file = max(files, key=lambda x: x['LastModified'])
    return last_modified_file['Key']

def load_pickle(pkl_filename):
    with open(pkl_filename, 'rb') as pkl_file:
        ids = pickle.load(pkl_file)
    return ids

def plot_pc(points, colors, save_path):
    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Scatter plot with color
    # breakpoint()
    # ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=(-points[:, 2]+1)/10+0.2, cmap='gray', marker='.')
    ax.scatter(points[:, 0], points[:, 2], points[:, 1], c=colors, cmap='gray', marker='.')
    # Title and show
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.5, 0.5])
    # plt.axis('off')
    plt.savefig(save_path, dpi=300)
    plt.close()


def get_model_conf(filepath: str):
    # load params
    with open(filepath, 'r', encoding='utf-8') as stream:
        params = yaml.safe_load(stream)
    return params
