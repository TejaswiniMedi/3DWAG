import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from datamodule import TextureDataModule
from models.vqvae import VQVAE
from VQVAE_trainer import VQVAE_Trainer_1
from data.utils_args import add_args
from data.utils import get_model_conf
import argparse
import os
import wandb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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

    return parser.parse_args()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes

def save_3d_visualization(source_data, filename):
    """
    Save 3D visualization of the source data (e.g., SDF) as a mesh.
    """
    fig = plt.figure(figsize=(8, 8))
    
    # Determine data range
    data_min = source_data.min()
    data_max = source_data.max()
    print(f"Source data range: {data_min} to {data_max}")

    # Extract the surface mesh from the SDF data using marching cubes
    level = (data_min + data_max) / 2  # Example for setting level to the middle of the range
    verts, faces, _, _ = marching_cubes(source_data, level=level)

    ax = fig.add_subplot(111, projection='3d')

    # Create a Poly3DCollection from the vertices and faces
    mesh = Poly3DCollection(verts[faces], alpha=0.5, linewidths=0.1, edgecolors='k')

    # Add the mesh to the plot
    ax.add_collection3d(mesh)

    # Set labels and limits
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reconstruction Data Visualization')

    # Set plot limits
    ax.set_xlim(0, source_data.shape[0])
    ax.set_ylim(0, source_data.shape[1])
    ax.set_zlim(0, source_data.shape[2])

    # Adjust the view angle for better visualization
    ax.view_init(elev=20, azim=30)

    # Save the figure
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Example usage (assumes you have source and reconstructed SDF data loaded)
# source_data = np.random.rand(64, 64, 64)  # Replace with your source SDF data
# recon_data = np.random.rand(64, 64, 64)  # Replace with your reconstructed SDF data
# save_3d_visualization(source_data, recon_data, 'sdf_comparison.png')

class CustomVQVAE_Trainer(VQVAE_Trainer_1):
    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        """
        x = batch['sdf']
        preds, usages, loss = self(x)
        
        # Save visualizations during validation
        for i in range(len(x)):
            sdf = x[i].cpu().numpy()
            recon_x = preds[i].cpu().numpy()
            save_path = os.path.join(self.save_path, f"visualization_validation_{batch_idx * len(x) + i}.png")
            save_3d_visualization(sdf, save_path)  # Assuming recon_x[0] is the SDF volume
        
        return {'val_loss': loss}

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        """
        x = batch['sdf']
        preds, usages, loss = self(x)
        
        # Save visualizations during training
        if batch_idx % 1 == 0:  # Save every 100 batches or any other frequency you prefer
            for i in range(len(x)):
                sdf = x[i].cpu().numpy()
                recon_x = preds[i].detach().cpu().numpy()
                save_path = os.path.join(self.save_path, f"visualization_train_{batch_idx * len(x) + i}.png")
                save_3d_visualization(sdf, save_path)  # Assuming recon_x[0] is the SDF volume
        
        return {'loss': loss}

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

    # Data loading (standard pytorch lightning or ffcv)
    datamodule = TextureDataModule(return_sdf=True)
    datamodule.setup(stage='train')

    # Create a Trainer instance
    trainer = pl.Trainer(strategy="ddp",  # or another strategy if needed
                         accelerator='gpu', num_nodes=num_nodes, devices=gpus, precision='32',
                         callbacks=[ModelCheckpoint(dirpath=save_checkpoint_dir, save_last=True)],
                         logger=logger, deterministic=True)

    # Train the model
    trainer.fit(model, datamodule)
    
    # Evaluate the model
    trainer.validate(model, datamodule)

    # Ensure wandb has stopped logging
    wandb.finish()

if __name__ == '__main__':
    main()
