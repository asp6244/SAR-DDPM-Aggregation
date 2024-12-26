"""
Train SAR-DDPM model.
"""

import argparse
import datetime
from torch.utils.data import DataLoader
import blobfile as bf

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    set_seed,
)
from guided_diffusion.train_util import TrainLoop
from datasets import SynthSARDataset
from parameters import default_args

def main():
    args = create_argparser().parse_args()
    print("Args: " + str(args) + "\n")

    if (args.seed is not None):
        set_seed(args.seed)

    dist_util.setup_dist()
    log_folder = bf.join(
        args.log_path,
        datetime.datetime.now().strftime(f"{args.train_dir.split('/')[1]}_%Y-%m-%d-%H-%M-%S"),
    )
    logger.configure(dir=log_folder, log_suffix="_train", format_strs=["log", "csv"])

    logger.log("Training dataset: " + args.train_dir)
    logger.log("Validation dataset: " + args.val_dir)
    if (args.resume_checkpoint):
        logger.log("Pretrained checkpoint: " + args.resume_checkpoint)
    logger.log("Args: " + str(args) + "\n")

    logger.log("Creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Creating data loaders...")
    train_dataset = SynthSARDataset(args.train_dir, train=True, num_channels=args.in_channels, crop_size=(args.large_size, args.large_size), seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
    val_dataset = SynthSARDataset(args.val_dir, train=False, num_channels=args.in_channels, crop_size=(args.large_size, args.large_size), length=((args.val_samples//args.batch_size)*args.batch_size), seed=args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    logger.log("Training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        train_loader=train_loader,
        val_loader=val_loader,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        in_channels=args.in_channels,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        use_ddim=args.use_ddim,
        learn_sigma=args.learn_sigma,
    ).run_loop()


def create_argparser():
    custom_defaults = dict( # These are overwridden by parameters.py
        # train and eval
        log_interval = 50,
        save_interval = 250,
        val_samples = 20,
        batch_size = 2,
        use_ddim = False,
        microbatch = 1,
        lr_anneal_steps = 0,
        weight_decay = 0.0,
        seed = None,

        # model
        large_size = 256,
        learn_sigma = True,
        in_channels = 3,
        ema_rate = "",
        lr = 1e-4,
        use_fp16 = True,
        fp16_scale_growth = 1e-3,
        num_channels = 192,
        num_heads = 4,
        num_res_blocks = 2,
        resblock_updown = True,
        use_scale_shift_norm = True,
        attention_resolutions = "32,16,8",
        class_cond = False,
        
        # diffusion
        timestep_respacing = "ddim100",
        diffusion_steps = 1000,
        schedule_sampler = "uniform",
        noise_schedule = "linear",
    )
    defaults = sr_model_and_diffusion_defaults()
    defaults.update(custom_defaults)
    defaults.update(default_args())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
