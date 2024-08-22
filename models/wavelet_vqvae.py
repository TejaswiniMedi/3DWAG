import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quant import VectorQuantizer2
from .dwt import DWTInverse3d
from .sparse_network import SparseComposer
from data.wavelet_utils import extract_wavelet_coefficients, extract_full_indices, extract_highs_from_values, pad_with_batch_idx
from models.basic_vae import Decoder, Encoder
#from networks.abstract_volume_nn import General_Decoder_Up_2, General_Encoder_Down_2, General_Decoder_Up_1
#from networks.modulating_volume_nn import Modulate_Decoder_Up_2, General_Encoder_Down_2_B, Modulate_Decoder_Up_2_B, Modulate_Decoder_Up_2_F, General_Encoder_Down_2_F, Modulate_Decoder_Up_1_B, General_Encoder_Down_1_B
from models.wavelet_utils import WaveletData
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
####################################################################################################################### ###########################################################################################################################################

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def sum_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))



class VQVAE(nn.Module):
    def __init__(
        self, vocab_size=4096, z_channels=64, ch=128, dropout=0.0,
        beta=0.25,              # commitment loss weight
        using_znorm=False,      # whether to normalize when computing the nearest neighbors
        quant_conv_ks=3,        # quant conv kernel size
        quant_resi=0.5,         # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4,     # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0, # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(1, 2, 3, 11),        #, 4),        #, 5, 6, 8, 6, 8, 10, 13, 16), # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        test_mode=True,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        # ddconfig is copied from https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/vq-f16/config.yaml
        ddconfig = dict(
            dropout=dropout, ch=ch, z_channels=z_channels,
            in_channels=64, ch_mult=(1, 2, 4), num_res_blocks=2,   # from vq-f16/config.yaml above    #ch_mult=(1, 1, 2, 2, 4)
            using_sa=True, using_mid_sa=True,                           # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop('double_z', None)  # only KL-VAE should use double_z=True
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.dwt_sparse_composer = SparseComposer(input_shape=[256, 256, 256],
                                             J=3,
                                             wave='bior6.8', mode='constant',
                                             inverse_dwt_module=None)
        self.dwt_inverse_3d = DWTInverse3d(J=3, wave='bior6.8', mode='constant')
        
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig['ch_mult'])-1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size, Cvae=self.Cvae, using_znorm=using_znorm, beta=beta,
            default_qresi_counts=default_qresi_counts, v_patch_nums=v_patch_nums, quant_resi=quant_resi, share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv3d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        self.post_quant_conv = torch.nn.Conv3d(self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks//2)
        
        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]
    
    # ===================== `forward` is only used in VAE training =====================
    def forward(self, low, high_indices, high_values, high_values_mask=None, high_indices_empty=None, ret_usages=True):   # -> rec_B3HW, idx_N, loss
        data_samples = self.data_process(low, high_indices, high_values, high_values_mask=high_values_mask, high_indices_empty=high_indices_empty)
        #inp = inp.unsqueeze(1)
        inp = data_samples
        #print(inp.shape)
        #h = self.encoder(data_samples)
        VectorQuantizer2.forward
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(self.encoder(inp)), ret_usages=ret_usages)
        #print(self.decoder(self.post_quant_conv(f_hat)).max(),self.decoder(self.post_quant_conv(f_hat)).min())
        return self.decoder(self.post_quant_conv(f_hat)), usages, vq_loss, inp
    
    
    # ===================== `forward` is only used in VAE training =====================
    
    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
    
    def img_to_idxBl(self, low: torch.Tensor, high_indices: torch.Tensor, high_values:torch.Tensor, high_values_mask: torch.Tensor,high_indices_empty: torch.Tensor, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.LongTensor]: 
           # return List[Bl]
        data_samples = self.data_process(low, high_indices, high_values, high_values_mask=high_values_mask, high_indices_empty=high_indices_empty)
        inp = data_samples
        inp = inp.cuda()
        f = self.quant_conv(self.encoder(inp))

        #print("most_important_for_me", self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)[0].shape)
        return self.quantize.f_to_idxBl_or_fhat(f, to_fhat=False, v_patch_nums=v_patch_nums)

    #(low, high_indices, high_values, high_values_mask, high_indices_empty)
    
    def idxBl_to_img(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l ** 0.5)
            ms_h_BChw.append(self.quantize.embedding(idx_Bl).transpose(1, 2).view(B, self.Cvae, pn, pn))
        return self.embed_to_img(ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one)
    
    def embed_to_img(self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(self.post_quant_conv(self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True))).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in self.quantize.embed_to_fhat(ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False)]
    
    def img_to_reconstructed_img(self, x, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None, last_one=False) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(f, to_fhat=True, v_patch_nums=v_patch_nums)
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1) for f_hat in ls_f_hat_BChw]
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.ema_vocab_hit_SV' in state_dict and state_dict['quantize.ema_vocab_hit_SV'].shape[0] != self.quantize.ema_vocab_hit_SV.shape[0]:
            state_dict['quantize.ema_vocab_hit_SV'] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(state_dict=state_dict, strict=strict)   # assign=assign
    
    def data_process(self, low, high_indices, high_values, high_values_mask=None, high_indices_empty=None):
        wavelet_data = WaveletData(shape_list=self.dwt_sparse_composer.shape_list, output_stage=2,
                                   max_depth= 3, low=low,
                                   highs_indices=high_indices, highs_values=high_values)
        data_samples = wavelet_data.convert_wavelet_volume()
        return data_samples