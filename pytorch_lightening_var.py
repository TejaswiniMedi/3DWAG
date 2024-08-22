import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

class VARTrainer(pl.LightningModule):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
        var_opt: AmpOptimizer, label_smooth: float,
    ):
        super(VARTrainer, self).__init__()
        
        self.var = var
        self.vae_local = vae_local
        self.quantize_local: VectorQuantizer2 = vae_local.quantize
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn * pn))
            cur += pn*pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True

    def forward(self, inp_B3HW: FTen, label_B: Union[ITen, FTen]):
        B, V = label_B.shape[0], self.vae_local.vocab_size
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        logits_BLV = self.var(label_B, x_BLCv_wo_first_l)
        return logits_BLV, gt_BL

    def training_step(self, batch, batch_idx):
        inp_B3HW, label_B = batch
        
        # progressive training setup
        prog_si = self.trainer.current_epoch % len(self.patch_nums)
        prog_wp_it = max(min(self.prog_it / self.trainer.max_epochs, 1), 0.01)
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si
        
        # forward pass
        logits_BLV, gt_BL = self(inp_B3HW, label_B)
        
        loss = self.train_loss(logits_BLV.view(-1, self.vae_local.vocab_size), gt_BL.view(-1)).view(len(batch), -1)
        if prog_si >= 0:  # in progressive training
            bg, ed = self.begin_ends[prog_si]
            lw = self.loss_weight[:, :ed].clone()
            lw[:, bg:ed] *= min(max(prog_wp_it, 0), 1)
        else:  # not in progressive training
            lw = self.loss_weight
        loss = loss.mul(lw).sum(dim=-1).mean()
        
        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=True)
        
        # logging
        pred_BL = logits_BLV.data.argmax(dim=-1)
        acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
        self.log('train_loss', loss)
        self.log('train_acc', acc_mean)
        self.prog_it += 1
        
        return loss

    def validation_step(self, batch, batch_idx):
        inp_B3HW, label_B = batch
        logits_BLV, gt_BL = self(inp_B3HW, label_B)
        
        loss_mean = self.val_loss(logits_BLV.view(-1, self.vae_local.vocab_size), gt_BL.view(-1))
        loss_tail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, self.vae_local.vocab_size), gt_BL[:, -self.last_l:].reshape(-1))
        acc_mean = (logits_BLV.data.argmax(dim=-1) == gt_BL).float().mean().item() * 100
        acc_tail = (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).float().mean().item() * 100
        
        metrics = {'val_loss_mean': loss_mean, 'val_loss_tail': loss_tail, 'val_acc_mean': acc_mean, 'val_acc_tail': acc_tail}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        return self.var_opt.optimizer

    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
        config: dict = state.pop('config', None)
        self.prog_it = config.get('prog_it', 0)
        self.last_prog_si = config.get('last_prog_si', -1)
        self.first_prog = config.get('first_prog', True)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)

