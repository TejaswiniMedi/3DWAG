def add_args(parser):  
    parser.add_argument(
        "--exp_name",
        type=str,
        default="Vqvae",
        help="Experiment name to use for folder naming",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Vqvae_encoder",
        help="Model to use: ",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of GPUs to train on."
    )
    parser.add_argument(
        "--num_cpus_per_worker",
        type=int,
        default=23,
        help="The number of CPUs to use per worker",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto"
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ddp",
        help="The name of distributed strategy used by lightning trainer ('ddp', 'fsdp' or 'deepspeed')",
    )
    parser.add_argument(
        "--s3_data_uri",
        type=str,
        default=None,
        help="S3 URI for the input bucket"
    )
    parser.add_argument(
        "--s3_results_uri",
        type=str,
        default="s3://texturegen-training-output-ue1",
        help="S3 URI for the output bucket",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=3,
        help="Batch size"
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=300,
        help="Max. epochs to train for"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint to recover from"
    )
    parser.add_argument(
        "--max_failures",
        type=int,
        default=0,
        help="The maximum number of attempts to recover a run.",
    )
    parser.add_argument(
        '--ray_base_folder',
        type=str,
        default='/tmp/ray_results/',
        help='local ray base path'
    )
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=50,
        help='checkpoint every'
    )
    parser.add_argument(
        '--run_local_test',
        type=bool,
        default=False,
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=4.5e-6,
    )
    parser.add_argument(
        '--comet',
        type=dict,
        help='comet args'
    )

    parser.add_argument('--validation_every', type=float, default=1, help='Value for gradient clipping')
    parser.add_argument('--grad_clipping', type=float, default=0.5, help='Value for gradient clipping')
    return parser