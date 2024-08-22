import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from models import *
#from VQVAE_trainer import *
#from datamodule import TextureDataModule
from pytorch_lightning_var import VARTrainer
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

    parser.add_argument('--params_file', type=str, default= "./config/vqvae.yaml")
    parser.add_argument('--dataloader', type=str, choices=['standard', 'ffcv'], default='standard',
                        help='defines what type of dataloader to use.')
   # parser.add_argument('--dataset_path', type=str, required=True,
              #         help='path to a dataset folder containing two sub-folders (validation / train) or beton files '
                    #         '(train.beton / validation.beton).')
    parser.add_argument('--save_path', type=str, default="./configs")
    parser.add_argument('--save_every_n_epochs', type=int, default=1, help='how often to save a new checkpoint')
    parser.add_argument('--run_name', type=str, default= "STAGE-1")
    parser.add_argument('--loading_path', type=str,
                        help='if passed, will load and continue training of an existing checkpoint', default=None)
    parser.add_argument('--logging', help='if passed, wandb logger is used', action='store_true')
    parser.add_argument('--wandb_project', type=str, help='project name for wandb logger', default='vqvae')
    parser.add_argument('--wandb_id', type=str,
                        help='wandb id of the run. Useful for resuming logging of a model', default=None)
    parser.add_argument('--workers', type=int, help='num of parallel workers', default=1)
    parser.add_argument('--num_nodes', type=int, help='number of gpu nodes used for training', default=1)

    parser.add_argument("--input_type", type=str, default='Wavelet', help='What is the input representation')
    parser.add_argument("--output_type", type=str, default='Wavelet', help='What is the output representation')
    parser.add_argument("--encoder_type", type=str, default='General_Encoder_Down_2', help='what is the encoder')
    parser.add_argument("--decoder_type", type=str, default='General_Decoder_Up_2', help='what is the decoder')
    parser.add_argument('--encoder_num_tran', type=int, default=0, help='encoder res layer')
    parser.add_argument('--decoder_num_tran', type=int, default=0, help='decoder res  layer')
    parser.add_argument('--last_feature_transform', type=str, default=None, help='add_noise, sigmoid, mlp, l2_norm or none')
    parser.add_argument('--reconstruct_loss_type', type=str, default="mean", help='bce or sum (mse) or mean (mse)')
    parser.add_argument('--quantizer_type', type=str, default="ema", help='original vs snikorn vs gumble')
    parser.add_argument('--normalize_latent', type=str, default=None, help='normalize latent')
    
    ### vector quant #####
    parser.add_argument('--e_dim', type=int, default=4, help='Dimension of embedding')
    parser.add_argument('--n_e', type=int, default=1024, help='Number of embeddings')
    parser.add_argument('--beta', type=float, default=0.25, help='beta for code attachment loss')
    parser.add_argument('--sample_mode', type=str, default="bilinear", help='beta for codebook loss')
    parser.add_argument('--padding', type=float, default=0.1, help='beta for codebook loss')
    parser.add_argument('--gamma', type=float, default=1, help='gamma loss')
    parser.add_argument('--grid_size', type=int, default=12, help='12')
    parser.add_argument('--t_loss', type=float, default=1.0, help='texture loss')
    
    
    ### Dataset details
    parser.add_argument('--dataset_name', type=str, default="ShapeNet", help='Dataset path')
    parser.add_argument('--dataset_path', nargs='+' ,  default=["/raid/data/dataset/Shapenet_D2/"], help='Dataset path')
 
    #parser.add_argument('--image_transform', type=str, default=None, help='Image transforms')
    parser.add_argument('--voxel_transform', type=str, default=None, help='Voxel Transforms')
    parser.add_argument("--num_points", type=int, default=2048, help='Number of points')
    parser.add_argument("--num_sdf_points", type=int, default=5000, help='Number of points')
    parser.add_argument('--categories',   nargs='+', default=None, metavar='N')

   
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
    parser.add_argument('--seed', type=int, default=1, help='Seed')
    parser.add_argument('--epochs', type=int, default=300, help="Total epochs")
    parser.add_argument('--checkpoint', type=str, default=None, help="Checkpoint to load")
    parser.add_argument('--use_timestamp',  action='store_true', help='Whether to use timestamp in dump files')
    parser.add_argument('--num_iterations', type=int, default=300000, help='How long the training shoulf go on')    
    
    
    ## SDF stuff
    parser.add_argument("--sdf_points", type=int, default=20000, help='Number of SDF points')
    parser.add_argument("--sdf_sample_type", type=str, default='mixture', help='uniform, near surface, mix')
    parser.add_argument("--sdf_res", type=int, default=256, help='what is the resolution of SDF')
    parser.add_argument("--greater_or_no", type=helper.bool_flag, default=False, help='use greater or not flag')
    
    ## S3 Dataset    
    parser.add_argument('--s3_bucket', type=str, default="build3d-wavelets", help='s3 images_base')
    parser.add_argument('--s3_prefix', type=str, default="dataset", help='s3 prefix')
    parser.add_argument('--use_s3', type=str, default=True, help='using S3 or not')
    parser.add_argument('--use_local_storage', help="use use_local_storage", action="store_true")
    
    parser.add_argument('--use_compile', type=helper.bool_flag, default=True, help='using compile or not')

    return parser.parse_args()


def main():

    # only for A100
    #set_matmul_precision()

    args = parse_args()
    conf = get_model_conf(args.params_file)

    # configuration params (assumes some env variables in case of multi-node setup)
    gpus = torch.cuda.device_count()
    num_nodes = args.num_nodes
    rank = int(os.getenv('NODE_RANK')) if os.getenv('NODE_RANK') is not None else 0
    is_dist = gpus > 1 or num_nodes > 1

    workers = int(args.workers)
    seed = int(args.seed)

    cumulative_batch_size = int(conf['training']['cumulative_bs'])
    batch_size_per_device = cumulative_batch_size // (num_nodes * gpus)

    base_learning_rate = float(conf['training']['base_lr'])
    learning_rate = base_learning_rate * math.sqrt(cumulative_batch_size / 256)

    max_epochs = int(conf['training']['max_epochs'])

    pl.seed_everything(seed, workers=True)

    # logging stuff, checkpointing and resume
    log_to_wandb = bool(args.logging)
    project_name = str(args.wandb_project)
    wandb_id = args.wandb_id

    run_name = str(args.run_name)
    save_checkpoint_dir = f'{args.save_path}{run_name}/'
    save_every_n_epochs = int(args.save_every_n_epochs)

    load_checkpoint_path = args.loading_path
    resume = load_checkpoint_path is not None

    if rank == 0:  # prevents from logging multiple times
        logger = WandbLogger(project=project_name, name=run_name, offline=not log_to_wandb, id=wandb_id,
                             resume='must' if resume else None)
    else:
        logger = WandbLogger(project=project_name, name=run_name, offline=True)

    # model params
    ae_conf = conf['autoencoder']
    q_conf = conf['quantizer']
    l_conf = conf['loss'] if 'loss' in conf.keys() else None
    t_conf = {'lr': learning_rate,
              'betas': conf['training']['betas'],
              'eps': conf['training']['eps'],
              'weight_decay': conf['training']['weight_decay'],
              'warmup_epochs': conf['training']['warmup_epochs'] if 'warmup_epochs' in conf['training'].keys() else None,
              'decay_epochs': conf['training']['decay_epochs'] if 'decay_epochs' in conf['training'].keys() else None,
              }

    # check if using adversarial loss
    use_adversarial = (l_conf is not None
                       and 'adversarial_params' in l_conf.keys()
                       and l_conf['adversarial_params'] is not None)

    # get model
    if resume:
        # image_size: int, ae_conf: dict, q_conf: dict, l_conf: dict, t_conf: dict, init_cb: bool = True,
        #                  load_loss: bool = True
        model = VQVAE.load_from_checkpoint(load_checkpoint_path, strict=False)
                                           #image_size=image_size, ae_conf=ae_conf, q_conf=q_conf, l_conf=l_conf,
                                           #t_conf=t_conf, init_cb=False, load_loss=True)
        print("loaded")
    else:
        #model = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=False, share_quant_resi=4, v_patch_nums=(1,2,3,4)).cuda()
        #model = VQVAE_Trainer()
        model = VARTrainer(
        device='cuda', patch_nums=(1,2,3,11), resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, var=var,
        var_opt=var_optim, label_smooth=args.ls,
    )

    # data loading (standard pytorch lightning or ffcv)
    #datamodule = get_datamodule(args.dataloader, args.dataset_path, image_size, batch_size_per_device,
                                #workers, seed, is_dist, mode='train')

    datamodule = WaveletDataModule(args=args)

    # callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=save_checkpoint_dir, filename='{epoch:02d}', save_last=True,
                                          save_top_k=-1, every_n_epochs=save_every_n_epochs)

    callbacks = [LearningRateMonitor(), checkpoint_callback]

    # trainer
    # set find unused parameters if using vqgan (adversarial training)
    trainer = pl.Trainer(strategy=DDPStrategy(find_unused_parameters=use_adversarial, static_graph=not use_adversarial),
                         accelerator='gpu', num_nodes=num_nodes, devices=gpus, precision='32',
                         callbacks=callbacks, deterministic=True, logger=logger,
                         max_epochs=max_epochs, check_val_every_n_epoch=5)

    print(f"[INFO] workers: {workers}")
    print(f"[INFO] batch size per device: {batch_size_per_device}")
    print(f"[INFO] cumulative batch size (all devices): {cumulative_batch_size}")
    print(f"[INFO] final learning rate: {learning_rate}")

    # check to prevent later error
    if use_adversarial and batch_size_per_device % 4 != 0:
        raise RuntimeError('batch size per device must be divisible by 4! (due to stylegan discriminator forward pass)')

    trainer.fit(model, datamodule, ckpt_path=load_checkpoint_path)

    # ensure wandb has stopped logging
    wandb.finish()


if __name__ == '__main__':
    main()