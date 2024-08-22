import time
import os
import os.path as osp
import jsonargparse
import yaml
import ray
from adsk_ailab_ray.ray_lightning import RayLightningExperiment
from adsk_ailab_ray.tools.aws import aws_s3_sync

from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models import *
from VQVAE_trainer import *
from wavelet_datamodule import WaveletDataModule
from data.utils_args import add_args


class LoggingCallback(pl.callbacks.Callback):
    def __init__(self, ray_base_folder, s3_results_uri, checkpoint_dir):
        self.ray_base_folder = ray_base_folder
        self.s3_results_uri = s3_results_uri
        self.checkpoint_dir = checkpoint_dir

    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
        self.experiment_setup(pl_module)

    def on_test_start(self, trainer, pl_module):
        print("Testing is starting")
        self.experiment_setup(pl_module)

    def experiment_setup(self, pl_module):
        print("Setting up folders.....")
        print(f"Debug folder is {self.ray_base_folder}")
        if not osp.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"Created folder for checkpoints: {self.checkpoint_dir}")
        print(f'Checkpoint will be saved locally to {self.checkpoint_dir}')

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        print(f'Saved checkpoint in {self.checkpoint_dir}')
        aws_s3_sync(source=self.ray_base_folder, destination=self.s3_results_uri)


def run_distributed_training_on_ray_cluster(
        exp_name,
        model_name,
        num_workers,
        num_cpus_per_worker,
        accelerator,
        strategy,
        strategy_kwargs,
        s3_data_uri,
        s3_results_uri,
        batch_size,
        learning_rate,
        max_epochs,
        max_failures,
        ray_base_folder,
        checkpoint_every,
        comet_args,
        data_args,
        run_local_test=False,
        ckpt_path="./checkpoints",
):
    exp_name = exp_name + "_" + time.strftime("%Y%m%d-%H%M%S")

    if run_local_test:
        num_cpus_per_worker = 3
        num_workers = 1
        ray_base_folder = './ray_results'
        torch.set_float16_matmul_precision('high')

    if model_name == 'Vqvae_encoder':
        model_class = VQVAE_Trainer
    else:
        raise NotImplementedError
    model_class_kwargs = {"lr": learning_rate}
    data_class = WaveletDataModule
    data_class_kwargs = {
        "args": data_args,
        "batch_size": batch_size,
        "ray_base_folder": ray_base_folder
    }

    callbacks = []
    checkpoint_dir = osp.join(ray_base_folder, exp_name, 'checkpoints/')
    callbacks.append(
        ModelCheckpoint(dirpath=checkpoint_dir,
                        filename='ckpt_{epoch}_{step}',
                        every_n_train_steps=checkpoint_every,
                        verbose=True)
    )
    callbacks.append(
        LoggingCallback(ray_base_folder=ray_base_folder,
                        s3_results_uri=s3_results_uri,
                        checkpoint_dir=checkpoint_dir)
    )

    trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": accelerator,
        "log_every_n_steps": 1,
        "logger": TensorBoardLogger("logs"),
        "callbacks": callbacks,
        "gradient_clip_val": data_args.get('grad_clipping', 0.5),
    }

    checkpointing_kwargs = {
        "monitor": "val_loss",
        "mode": "max",
        "save_top_k": 1,
    }

    experiment = RayLightningExperiment(
        exp_name=exp_name,
        num_workers=num_workers,
        num_cpus_per_worker=num_cpus_per_worker,
        strategy=strategy,
        strategy_kwargs=strategy_kwargs,
        model_class=model_class,
        model_class_kwargs=model_class_kwargs,
        data_class=data_class,
        data_class_kwargs=data_class_kwargs,
        trainer_kwargs=trainer_kwargs,
        checkpointing_kwargs=checkpointing_kwargs,
        s3_data_uri=s3_data_uri,
        s3_results_uri=s3_results_uri,
        max_failures=max_failures,
        use_gpu=True,
        ckpt_path=ckpt_path,
        comet_experiment_kwargs=comet_args,
        timestamp_exp_name=False,
    )

    if run_local_test:
        result = experiment.fit_dry_run()
        print("Validation Loss: ", result.logged_metrics["val_loss"])
    else:
        result = experiment.fit()

    ray.shutdown()  # Optional: Shut down Ray (used for tests)


if __name__ == "__main__":
    parser = jsonargparse.ArgumentParser()

    parser.add_argument(
        "--config",
        action=jsonargparse.ActionConfigFile,
        help="Path to the main YAML config file.",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="Path to the data YAML config file.",
    )

    parser = add_args(parser)

    args = parser.parse_args()
    print(yaml.dump(args.as_dict(), default_flow_style=False))

    # Manually load additional config file
    #if args.data_config:
    with open(args.data_config, 'r') as file:
        data_config = yaml.safe_load(file)
    #else:
        #data_config = {}

    # Merge data config into args
    dataset_args = data_config
    print(dataset_args)

    # Initialize data module with dataset arguments
    data_module = WaveletDataModule(**dataset_args)

    run_distributed_training_on_ray_cluster(
        exp_name=args.exp_name,
        model_name=args.model_name,
        num_workers=args.num_workers,
        num_cpus_per_worker=args.num_cpus_per_worker,
        accelerator=args.accelerator,
        strategy=args.strategy,
        strategy_kwargs={},
        s3_data_uri=args.s3_data_uri,
        s3_results_uri=args.s3_results_uri,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        max_failures=args.max_failures,
        ray_base_folder=args.ray_base_folder,
        checkpoint_every=args.checkpoint_every,
        comet_args=args.comet,
        data_args=dataset_args,
        ckpt_path=args.ckpt_path,
        run_local_test=args.run_local_test,
    )
