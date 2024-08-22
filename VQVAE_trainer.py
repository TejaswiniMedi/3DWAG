import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models import *
#from datamodule import TextureDataModule
from wavelet_datamodule import WaveletDataModule
from models.wavelet_vqvae import VQVAE, VectorQuantizer2
from torch.nn.utils import clip_grad_norm_
import numpy as np
from sdf_wavelets import *
import numpy as np
from skimage import measure

class VQVAE_Trainer(pl.LightningModule):
    def __init__(self, lr_rate = 4.5e-6,beta=0.25, autoencoder = "Vqvae_encoder", vocab_size=4096,z_channels=32, ch=160, share_quant_resi=4, v_patch_nums=(1, 2, 3, 4),**kwargs):   #, 5, 6, 8, 10, 13, 16)
        super().__init__()
        self.lr = lr_rate
        self.autencoder_name = autoencoder
        self.beta = beta
        if self.autencoder_name == "Vqvae_encoder":
            self.model = VQVAE()

    def forward(self, low, high_indices, high_values, high_values_mask, high_indices_empty):
        return self.model(low, high_indices, high_values, high_values_mask, high_indices_empty)
    
    def training_step(self,batch,batch_idx):
        #sdf = batch['sdf']
        data= batch
        low = data['low']
        high_indices = data['high_indices']
        high_values = data['high_values']
        high_values_mask = data['high_values_mask']
        high_indices_empty = data['high_indices_empty'] 
        pred,  info , loss_latent, inp = self(low, high_indices, high_values, high_values_mask, high_indices_empty)
        #print(pred.min(), pred.max(), inp.min(), inp.max())
        l2 = torch.nn.MSELoss(reduction='mean')
        loss_rec = l2(pred, inp)
        loss = loss_rec + self.beta * loss_latent
        train_iou = self.compute_iou(inp,pred)
        self.log("train_loss", loss)
        self.log("reconstruction_train_loss", loss_rec)
        self.log("codebook_loss", loss_latent)
        self.log("train_iou", train_iou)
        return loss

    
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
        return loss
    
    def test_step(self, test_batch, batch_idx):
        data= test_batch
        low = data['low']
        high_indices = data['high_indices']
        high_values = data['high_values']
        high_values_mask = data['high_values_mask']
        high_indices_empty = data['high_indices_empty'] 
        pred,  info , loss_latent, inp = self(low, high_indices, high_values, high_values_mask, high_indices_empty)
        l2 = torch.nn.MSELoss(reduction='mean')
        loss_rec = l2(pred, inp)
        loss = loss_rec + self.beta * loss_latent
        test_iou = self.compute_iou(inp,pred)
        self.log("val_loss", loss)
        self.log("reconstruction_val_loss", loss_rec)
        self.log("codebook_val_loss", loss_latent)
        self.log("val_iou", test_iou)
        return loss
    
    def configure_optimizers(self):
        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.lr)
        return optimizer
    
    def on_after_backward(self):
        # Specify a very high value for max_norm to avoid actual clipping if undesired
        max_norm = 1e6
        total_norm = clip_grad_norm_(self.parameters(), max_norm=max_norm)
        self.log("gradients_norm", total_norm)

    def normalize_data(self, data):
        min_val = data.min()
        max_val = data.max()
        return (data - min_val) / (max_val - min_val) * 2 - 1
    
    def compute_iou(self, occ1, occ2, mean=True):
        ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.
    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
        mean (bool, optional): If True, returns the mean IoU across all samples. Default is False.
    Returns:
        iou (tensor): IoU values for each sample
    '''
        # Ensure inputs are PyTorch tensors
        #occ1 = torch.tensor(occ1)
        #occ2 = torch.tensor(occ2)
    
        # Put all data in the second dimension
        #print("occ1",occ1.shape)
        if occ1.dim() >= 2:
            occ1 = occ1.view(occ1.size(0), -1)
        if occ2.dim() >= 2:
            occ2 = occ2.view(occ2.size(0), -1)
    
        # Convert to boolean values
        occ1 = (occ1 >= 0.5)
        occ2 = (occ2 >= 0.5)
    
        # Compute IOU
        area_union = (occ1 | occ2).float().sum(dim=-1)
        area_intersect = (occ1 & occ2).float().sum(dim=-1)
    
        iou = area_intersect / area_union
        if mean:
            iou = torch.nanmean(iou)
    
        return iou
    


    def save_sdf_as_obj(self, sdf_batch, filename_prefix):
        # Iterate through the batch of SDFs
        for i, sdf in enumerate(sdf_batch):
            # Generate a filename for each SDF in the batch
            filename = f"{filename_prefix}_{i}.obj"
        
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

# Example usage:
# Assuming `sdf_batch` is a tensor or ndarray of shape (batch_size, x, y, z)
# self.save_sdf_as_obj(sdf_batch, "output/sdf_mesh")

        