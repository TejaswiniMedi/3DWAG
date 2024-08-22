import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from wavelet_datamodule import WaveletDataModule
from models.vqvae import VQVAE
from VQVAE_trainer import VQVAE_Trainer
from data.utils_args import add_args
from data.utils import get_model_conf
import argparse
import os
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from models import *
from VQVAE_trainer import *
#from datamodule import TextureDataModule
from wavelet_datamodule import WaveletDataModule
from data.utils_args import add_args
from data.utils import get_model_conf
import argparse
import math
import os.path
import wandb
import helper
from models.wavelet_vqvae import VQVAE, VectorQuantizer2

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--params_file', type=str, default="./config/vqvae.yaml")
    parser.add_argument('--dataloader', type=str, choices=['standard', 'ffcv'], default='standard',
                        help='Defines what type of dataloader to use.')
    parser.add_argument('--save_path', type=str, default="./results")
    parser.add_argument('--save_every_n_epochs', type=int, default=1, help='How often to save a new checkpoint')
    parser.add_argument('--run_name', type=str, default="STAGE-1")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--loading_path', type=str, help='If passed, will load and continue training of an existing checkpoint', default=None)
    parser.add_argument('--logging', help='If passed, wandb logger is used', action='store_true')
    parser.add_argument('--wandb_project', type=str, help='Project name for wandb logger', default='vqvae')
    parser.add_argument('--wandb_id', type=str, help='wandb id of the run. Useful for resuming logging of a model', default=None)
    parser.add_argument('--workers', type=int, help='Num of parallel workers', default=1)
    parser.add_argument('--num_nodes', type=int, help='Number of GPU nodes used for evaluation', default=1)
     ## SDF stuff
    parser.add_argument("--sdf_points", type=int, default=20000, help='Number of SDF points')
    parser.add_argument("--sdf_sample_type", type=str, default='mixture', help='uniform, near surface, mix')
    parser.add_argument("--sdf_res", type=int, default=256, help='what is the resolution of SDF')
    parser.add_argument("--greater_or_no", type=helper.bool_flag, default=False, help='use greater or not flag')
    
    ### wavelet setting
    parser.add_argument('--resolution', type=int, default=256, help='resolution')
    parser.add_argument('--max_depth', type=int, default=3, help='max_depth')
    parser.add_argument('--max_training_level', type=int, default=2, help='max_depth')
    parser.add_argument('--point_num', type=int, default=16384, help='point_num')
    parser.add_argument('--keep_level', type=int, default=2, help='keep_level')
    parser.add_argument('--data_keep_level', type=int, default=2, help='data_keep_level')
    parser.add_argument('--wavelet', type=str, default='bior6.8', help='wavelet')
    parser.add_argument('--padding_mode', type=str, default='constant', help='padding_mode')
    parser.add_argument('--use_normalization', help="use min max normalization", action="store_true")
    parser.add_argument('--use_shift_mean', type=helper.bool_flag, default=False,  help="use shift_mean")
    parser.add_argument('--start_stage', type=int, default=0, help='start_stage')
    parser.add_argument('--use_adaptive_stage_update', help="use adaptive_stage_update", action="store_true")
    parser.add_argument('--no_rebalance_loss', type=helper.bool_flag, default=True, help="use no_rebalance_loss")
    parser.add_argument('--use_compact_indices', help="use use_compact_indices", default=True, action="store_true")
    parser.add_argument('--sample_threshold_ratio', type=float, default=0.03125, help='point_num')
    parser.add_argument('--use_batched_threshold', default=True, help="use use_batched_threshold", action="store_true")
    parser.add_argument('--use_sample_training', type=helper.bool_flag, default=False, help="use sample_training")
    parser.add_argument('--use_sample_threshold',  default=True, help="use sample_threshold", action="store_true")
    
    ## Chamfer parameters
    parser.add_argument("--div_hyp", default=1.0, type=float, help="hyperparameter for div hyp")
    
    ### training details
    parser.add_argument('--train_mode', type=str, default="train", help='train or test')
    parser.add_argument('--epochs', type=int, default=300, help="Total epochs")
    parser.add_argument('--checkpoint', type=str, default=None, help="Checkpoint to load")
    parser.add_argument('--use_timestamp',  action='store_true', help='Whether to use timestamp in dump files')
    parser.add_argument('--num_iterations', type=int, default=300000, help='How long the training shoulf go on')    
    
    ## S3 Dataset    
    parser.add_argument('--s3_bucket', type=str, default="build3d-wavelets", help='s3 images_base')
    parser.add_argument('--s3_prefix', type=str, default="dataset", help='s3 prefix')
    parser.add_argument('--use_s3', type=str, default=True, help='using S3 or not')
    parser.add_argument('--use_local_storage', help="use use_local_storage", action="store_true")
    
    parser.add_argument('--use_compile', type=helper.bool_flag, default=True, help='using compile or not')


    return parser.parse_args()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes

def save_3d_visualization(source_data, recon_data, filename):
    """
    Save 3D visualization of the source and reconstructed data (e.g., SDF) as meshes.
    """
    fig = plt.figure(figsize=(12, 6))

    for i, (data, title) in enumerate(zip([source_data, recon_data], ['Source SDF', 'Reconstructed SDF'])):
        # Determine data range
        data_min = data.min()
        data_max = data.max()
        print(f"{title} data range: {data_min} to {data_max}")

        # Extract the surface mesh from the SDF data using marching cubes
        level = (data_min + data_max) / 2  # Example for setting level to the middle of the range

        verts, faces, _, _ = marching_cubes(data, level=level)

        ax = fig.add_subplot(1, 2, i+1, projection='3d')

        # Create a Poly3DCollection from the vertices and faces
        mesh = Poly3DCollection(verts[faces], alpha=0.5, linewidths=0.1, edgecolors='k')

        # Add the mesh to the plot
        ax.add_collection3d(mesh)

        # Set labels and limits
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)

        # Set plot limits
        ax.set_xlim(0, data.shape[0])
        ax.set_ylim(0, data.shape[1])
        ax.set_zlim(0, data.shape[2])

        # Adjust the view angle for better visualization
        ax.view_init(elev=20, azim=30)

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_sdf_as_obj(sdf_batch, filename_prefix):
        print(sdf_batch.shape)
        sdf_batch = sdf_batch.reshape(3,256,256,256)
        # Iterate through the batch of SDFs
        for i, sdf in enumerate(sdf_batch):
            # Generate a filename for each SDF in the batch
            filename = f"{filename_prefix}_{i}.obj"
            sdf = sdf.detach().cpu().numpy()
            # Extract the mesh using the Marching Cubes algorithm
            verts, faces, normals, values = measure.marching_cubes(sdf, level=0)
        
            # Write the vertices and faces to the .obj file
            with open(filename, 'w') as file:
                # Write vertices
                for vert in verts:
                    file.write(f"v {vert[0]} {vert[1]} {vert[2]}\n")
            
                # Write faces (Note: OBJ format starts counting from 1)
                for face in faces:
                    file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")
        
            print(f"Saved SDF as {filename}")
# Example usage (assumes you have source and reconstructed SDF data loaded)
# source_data = np.random.rand(64, 64, 64)  # Replace with your source SDF data
# recon_data = np.random.rand(64, 64, 64)  # Replace with your reconstructed SDF data
# save_3d_visualization(source_data, recon_data, 'sdf_comparison.png')



class CustomVQVAE_Trainer(VQVAE_Trainer):
    def validation_step(self, val_batch, batch_idx):
        data= val_batch
        low = data['low']
        high_indices = data['high_indices']
        high_values = data['high_values']
        high_values_mask = data['high_values_mask']
        high_indices_empty = data['high_indices_empty'] 
        pred,  info , loss_latent, inp = self(low, high_indices, high_values, high_values_mask, high_indices_empty)
        l2 = torch.nn.MSELoss(reduction='mean')
        loss_rec = l2(pred, inp)
        loss = loss_rec + self.beta * loss_latent
        val_iou = self.compute_iou(inp,pred)
        self.log("val_loss", loss, on_epoch=True)
        self.log("reconstruction_val_loss", loss_rec, on_epoch = True)
        self.log("codebook_val_loss", loss_latent, on_epoch=True)
        self.log("val_iou", val_iou, on_epoch=True)
        shape_list = ((256,256,256), (136,136,136),(76,76,76),(46,46,46))
        wavelet_data_pred = WaveletData(shape_list=shape_list, output_stage=2 , max_depth=3, wavelet_volume=pred.contiguous())
        #print(dec.shape)
        low_pred, highs_pred = wavelet_data_pred.convert_low_highs()
        dwt_inverse_module = DWTInverse3d(3, "bior6.8","constant")
        sdf_bernini_pred = dwt_inverse_module((low_pred, highs_pred))
        save_sdf_as_obj(sdf_bernini_pred, f'val_{batch_idx}')
        return loss
    

def main():
    args = parse_args()
    conf = get_model_conf(args.params_file)

    # Configuration params
    gpus = torch.cuda.device_count()
    num_nodes = args.num_nodes
    rank = int(os.getenv('NODE_RANK')) if os.getenv('NODE_RANK') is not None else 0

    workers = int(args.workers)
    seed = int(args.seed)

    # Logging and checkpoints
    log_to_wandb = bool(args.logging)
    project_name = str(args.wandb_project)
    wandb_id = args.wandb_id

    run_name = str(args.run_name)
    save_checkpoint_dir = f'{args.save_path}/{run_name}/'
    load_checkpoint_path = args.loading_path

    if rank == 0:  # prevents from logging multiple times
        logger = WandbLogger(project=project_name, name=run_name, offline=not log_to_wandb, id=wandb_id,
                             resume='must' if load_checkpoint_path else None)
    else:
        logger = WandbLogger(project=project_name, name=run_name, offline=True)

    # Model parameters
    ae_conf = conf['autoencoder']
    q_conf = conf['quantizer']
    l_conf = conf['loss'] if 'loss' in conf.keys() else None
    t_conf = {'lr': 0,  # Placeholder, not used in evaluation
              'betas': (0, 0),  # Placeholder
              'eps': 0,  # Placeholder
              'weight_decay': 0,  # Placeholder
              'warmup_epochs': None,
              'decay_epochs': None,
              }

    # Load model
    model = CustomVQVAE_Trainer.load_from_checkpoint(load_checkpoint_path, strict=False,
                                                     image_size=64, ae_conf=ae_conf, q_conf=q_conf, l_conf=l_conf,
                                                     t_conf=t_conf, init_cb=False, load_loss=True)
    model.save_path = args.save_path  # Set save path for visualizations
    model.eval()

    # Data loading (standard pytorch lightning or ffcv)
    datamodule = WaveletDataModule(args=args)
    datamodule.setup(stage='val')

    # Create a Trainer instance
    trainer = pl.Trainer(strategy="ddp",  # or another strategy if needed
                         accelerator='gpu', num_nodes=num_nodes, devices=gpus, precision='32',
                         callbacks=[ModelCheckpoint(dirpath=save_checkpoint_dir, save_last=True)],
                         logger=logger, deterministic=True)

    # Evaluate the model
    trainer.validate(model, datamodule)

    # Ensure wandb has stopped logging
    wandb.finish()

if __name__ == '__main__':
    main()