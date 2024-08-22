import time
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from functools import partial
import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger
import pytorch_lightning as pl
from utils.amp_sc import AmpOptimizer
from utils.lr_control import filter_params

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
        
        self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.var_opt = var_opt
        
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]* patch_nums[-1]
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
        self.tot = 0
        self.L_mean, self.L_tail, self.acc_mean, self.acc_tail = 0, 0, 0, 0

    def forward(self,label_B, x_BLCv_wo_first_l):
        label_B = None
        self.var_wo_ddp.forward
        return self.var_wo_ddp(label_B,x_BLCv_wo_first_l)

    def validation_step(self, val_batch, batch_idx):
        #print(g_it,stepping, prog_si, prog_wp_it )
        g_it = batch_idx
        prog_wp_it = 16728
        # if progressive training
        stepping = True
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si = -1
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        data = val_batch
        low = data['low']
        high_indices = data['high_indices']
        high_values = data['high_values']
        high_values_mask = data['high_values_mask']
        high_indices_empty = data['high_indices_empty'] 
        label_B = None

        # forward
        B, V = low.shape[0], self.vae_local.vocab_size
        #self.var.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(low, high_indices, high_values, high_values_mask=high_values_mask, high_indices_empty=high_indices_empty)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        #print(gt_BL.shape)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        #print(x_BLCv_wo_first_l.shape)
        with self.var_opt.amp_ctx:
            #self.var_wo_ddp.forward
            logits_BLV = self(label_B, x_BLCv_wo_first_l)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # not in progressive training
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()
        
        # backward
        #grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return loss
        #return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def training_step(
        self, batch, batch_idx
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        
        #print(g_it,stepping, prog_si, prog_wp_it )
        g_it = batch_idx
        prog_wp_it = 16728
        # if progressive training
        stepping = True
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = prog_si = -1
        if self.last_prog_si != prog_si:
            if self.last_prog_si != -1: self.first_prog = False
            self.last_prog_si = prog_si
            self.prog_it = 0
        self.prog_it += 1
        prog_wp = max(min(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog: prog_wp = 1    # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1: prog_si = -1    # max prog, as if no prog
        
        data = batch
        low = data['low']
        high_indices = data['high_indices']
        high_values = data['high_values']
        high_values_mask = data['high_values_mask']
        high_indices_empty = data['high_indices_empty'] 
        label_B = None

        # forward
        B, V = low.shape[0], self.vae_local.vocab_size
        #self.var.require_backward_grad_sync = stepping
        
        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(low, high_indices, high_values, high_values_mask=high_values_mask, high_indices_empty=high_indices_empty)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        #print(gt_BL.shape)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
        #print(x_BLCv_wo_first_l.shape)
        with self.var_opt.amp_ctx:
            #self.var_wo_ddp.forward
            logits_BLV = self(label_B, x_BLCv_wo_first_l)
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            if prog_si >= 0:    # in progressive training
                bg, ed = self.begin_ends[prog_si]
                assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
                lw = self.loss_weight[:, :ed].clone()
                lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
            else:               # not in progressive training
                lw = self.loss_weight
            loss = loss.mul(lw).sum(dim=-1).mean()
        
        # backward
        grad_norm, scale_log2 = self.var_opt.backward_clip_step(loss=loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        
        self.var_wo_ddp.prog_si = self.vae_local.quantize.prog_si = -1
        return loss
    
    def get_config(self):
        return {
            'patch_nums':   self.patch_nums, 'resos': self.resos,
            'label_smooth': self.label_smooth,
            'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
        }
    
    def configure_optimizers(self):
        # build optimizer
        names, paras, para_groups = filter_params(self.var_wo_ddp, nowd_keys={
        'cls_token', 'start_token', 'task_token', 'cfg_uncond',
        'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
        'gamma', 'beta',
        'ada_gss', 'moe_bias',
        'scale_mul',
        })
        opt_clz = {
        'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=True),
        'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=True),
        }['adamw'.lower().strip()]
        opt_kw = dict(lr=1e-4, weight_decay=0)
        print(f'[INIT] optim={opt_clz}, opt_kw={opt_kw}\n')
    
        var_optim = AmpOptimizer(
        mixed_precision=0, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
        grad_clip=2.0, n_gradient_accumulation=1
        )
        del names, paras, para_groups
        print(var_optim)
        return var_optim.optimizer
    
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
