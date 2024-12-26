"""
SAR-DDPM Inference on real SAR images.
"""

import argparse
import os
import re
import datetime
import torch
from torch.utils.data import DataLoader

from guided_diffusion.unet import UNetModel
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from torch.optim import AdamW
from guided_diffusion.fp16_util import MixedPrecisionTrainer
import blobfile as bf
import math
import time
from sewar.full_ref import vifp
import matplotlib.pyplot as plt
import numpy as np
import lpips
lpips_model = lpips.LPIPS(net='alex').cuda()

from tqdm import tqdm

from datasets import SynthSARDataset
from parameters import default_args
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    add_dict_to_argparser,
    set_seed,
)
from guided_diffusion.test_util import evaluate_sar

# Training or testing?
# This will determine which set of arguments to load from ./scripts/parameters.py
test = False

# How many iterations does the first DDPM model use?
num_channels = 100

# Determine where the pretrained SAR-DDPM log folder is
ddpm_log_path = "./logs/DSIFN_2024-12-22-23-05-51/"

# Trained U-Net model path
unet_model_checkpoint = None #"./logs/DSIFN_new_2024-12-26-01-15-16/model_best.pt"

def main():
    args = create_argparser(test=test).parse_args()

    if (args.seed is not None):
        set_seed(args.seed)
    
    # Parse the arguments from the ddpm training logfile
    ddpm_training_log_file = os.path.join(ddpm_log_path, 'log_train.txt')
    if not os.path.exists(ddpm_training_log_file):
        raise FileNotFoundError(f"Log file not found: {ddpm_training_log_file}")
    ddpm_training_args = parse_args_from_log(ddpm_training_log_file)

    if not test: # Train
        log_folder = os.path.join(
            args.log_path,
            datetime.datetime.now().strftime(f"{args.train_dir.split('/')[1]}_new_%Y-%m-%d-%H-%M-%S"),
        )
        logger.configure(dir=log_folder, log_suffix="_train", format_strs=["log", "csv"])
        images_folder = bf.join(logger.get_dir(), "val_images")

        logger.log("Training dataset: " + args.train_dir)
        logger.log("Validation dataset: " + args.val_dir)
    else: # Test
        # Determine where the log folder is
        if not args.training_log_folder:
            log_path = get_most_recent_log_folder(args.log_path)
        else:
            log_path = args.training_log_folder

        # Parse the arguments from the training logfile
        training_log_file = os.path.join(log_path, 'log_train.txt')
        if not os.path.exists(training_log_file):
            raise FileNotFoundError(f"Log file not found: {training_log_file}")
        training_args = parse_args_from_log(training_log_file)

        test_log_folder = os.path.join(
            log_path,
            datetime.datetime.now().strftime(f"test_{args.test_dir.split('/')[1]}_%Y-%m-%d-%H-%M-%S"),
        )
        logger.configure(dir=test_log_folder, log_suffix="_test", format_strs=["log","csv"])
        print(f"Logging to {test_log_folder}")
        images_folder = bf.join(logger.get_dir(), "test_images")

        logger.log("Testing dataset: " + args.test_dir)

    logger.log("Args: " + str(args))
    logger.log("Training args: " + str(ddpm_training_args) + "\n")

    # Get checkpoint path with the largest step index
    if hasattr(args, "test_checkpoint") and args.test_checkpoint:
        test_checkpoint = args.test_checkpoint
    else:
        if hasattr(args, "model_to_use") and args.model_to_use == "BEST":
            test_checkpoint = os.path.join(ddpm_log_path, "model_best.pt")
        else:
            test_checkpoint = os.path.join(ddpm_log_path, "model_max.pt")
        # test_checkpoint = os.path.join(log_path, "model_latest.pt") # TODO: which one?

    dist_util.setup_dist()

    if not bf.exists(images_folder):
        bf.makedirs(images_folder)

    logger.log("Creating model...")
    # Overwrite any training testing arguments with those from the testing arguments
    args_dict = vars(args)
    model, diffusion = sr_create_model_and_diffusion(
        **{k: args_dict[k] if k in args_dict.keys() else (ddpm_training_args[k] if k in ddpm_training_args.keys() else None) for k in sr_model_and_diffusion_defaults().keys()}
    )
    model.to(dist_util.dev())
    
    logger.log("Creating data loader...")
    if not test:
        train_dataset = SynthSARDataset(args.train_dir, train=True, num_channels=ddpm_training_args['in_channels'], crop_size=(ddpm_training_args['large_size'], ddpm_training_args['large_size']), seed=args.seed)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True)
        val_dataset = SynthSARDataset(args.val_dir, train=False, num_channels=ddpm_training_args['in_channels'], crop_size=(ddpm_training_args['large_size'], ddpm_training_args['large_size']), length=((args.val_samples//args.batch_size)*args.batch_size), seed=args.seed)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    else:
        test_dataset = SynthSARDataset(args.test_dir, train=False, num_channels=ddpm_training_args['in_channels'], crop_size=(ddpm_training_args['large_size'], ddpm_training_args['large_size']), length=args.test_length, seed=args.seed)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)


    logger.log("Loading model from checkpoint:" + test_checkpoint)
    dict_load = dist_util.load_state_dict(test_checkpoint, map_location=dist_util.dev())
    model.load_state_dict(dict_load, strict=False)
    if ddpm_training_args['use_fp16']:
        model.convert_to_fp16()

    if ddpm_training_args['large_size'] == 512:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif ddpm_training_args['large_size'] == 256:
        channel_mult = (1, 1, 2, 2, 4, 4)
    elif ddpm_training_args['large_size'] == 64:
        channel_mult = (1, 2, 3, 4)
    else:
        raise ValueError(f"unsupported large size: {ddpm_training_args['large_size']}")

    attention_ds = []
    for res in ddpm_training_args['attention_resolutions'].split(","):
        attention_ds.append(ddpm_training_args['large_size'] // int(res))

    unet_model = UNetModel(
        image_size=ddpm_training_args['large_size'],
        in_channels=ddpm_training_args['in_channels']*(num_channels+1),
        model_channels=ddpm_training_args['num_channels'],
        out_channels=ddpm_training_args['in_channels'],
        num_res_blocks=ddpm_training_args['num_res_blocks'],
        attention_resolutions=tuple(attention_ds),
        dropout=ddpm_training_args['dropout'],
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=ddpm_training_args['use_checkpoint'],
        num_heads=ddpm_training_args['num_heads'],
        num_head_channels=ddpm_training_args['num_head_channels'],
        num_heads_upsample=ddpm_training_args['num_heads_upsample'],
        use_scale_shift_norm=ddpm_training_args['use_scale_shift_norm'],
        resblock_updown=ddpm_training_args['resblock_updown'],
        use_fp16=ddpm_training_args['use_fp16'],
    )

    if unet_model_checkpoint:
        unet_model.load_state_dict(torch.load(unet_model_checkpoint))

    unet_model.to(dist_util.dev())

    logger.log("Beginning testing...")
    model.eval()

    num_steps = int(args.timestep_respacing) if isinstance(args.timestep_respacing, str) else sum(args.timestep_respacing)

    t = torch.tensor([1.0] * args.batch_size, device=dist_util.dev())

    if not test:
        unet_model.train()
        mp_trainer = MixedPrecisionTrainer(
            model=unet_model,
            use_fp16=args.use_fp16,
            fp16_scale_growth=1e-3,
        )
        opt = AdamW(
            mp_trainer.master_params, lr=args.lr, weight_decay=args.weight_decay
        )

        start_time = time.perf_counter()
        net_val_time = 0.0
        best_psnr = 0.0

        step = 0
        e = 0
        # for e in range(50):
        while (step < 8000):
            # avg_psnr, avg_ssim, _, _ = evaluate(val_loader, diffusion, model, dist_util.dev(), images_folder, num_steps, num_channels, unet_model, 1)
            
            progress_bar = tqdm(train_loader, desc=f"PSNR: 00.00/00.00, SSIM: 0.000/0.000", unit='batch')
            for batch_idx, data_tuple in enumerate(progress_bar):
                clean_tensor, noisy_tensor, image_filename = data_tuple
                clean_tensor = clean_tensor.to(dist_util.dev())
                noisy_tensor = noisy_tensor.to(dist_util.dev())
                
                # Otherwise, get the predicted clean image as normal
                model_kwargs = {'noisy': noisy_tensor}

                with torch.no_grad():
                    pred_tensor = diffusion.p_sample_loop(
                                    model,
                                    noisy_tensor.shape,
                                    clip_denoised=True,
                                    model_kwargs=model_kwargs,
                                )
                    
                    pred_tensor = pred_tensor.transpose(0,1).float()
                    pred_tensor = pred_tensor[:, -1*num_channels:]
                    shape = pred_tensor.shape
                    pred_tensor = pred_tensor.contiguous().view(shape[0], shape[1]*shape[2], shape[3], shape[4])

                output = unet_model(torch.cat((noisy_tensor, pred_tensor), dim=1), t)
                
                output = torch.mean(output, dim=1)
                clean_tensor = torch.mean(clean_tensor, dim=1)

                loss = F.mse_loss(output, clean_tensor)

                mp_trainer.zero_grad()
                mp_trainer.backward(loss)
                mp_trainer.optimize(opt)

                output_image = ((output + 1.0)* 127.5).clamp(0, 255.0)
                output_image = torch.round(output_image) / 255.0
                output_image = output_image.contiguous()
                output_image = output_image.detach().cpu().numpy()
                
                # Reformat the images for metrics
                clean_image = ((clean_tensor + 1.0)* 127.5).clamp(0, 255.0)
                clean_image = torch.round(clean_image) / 255.0
                clean_image = clean_image.contiguous()
                clean_image = clean_image.cpu().numpy()

                noisy_image = ((torch.mean(noisy_tensor, dim=1) + 1.0)* 127.5).clamp(0, 255.0)
                noisy_image = torch.round(noisy_image) / 255.0
                noisy_image = noisy_image.contiguous()
                noisy_image = noisy_image.cpu().numpy()

                net_psnr = 0.0
                net_ssim = 0.0
                for b in range(args.batch_size):
                    net_psnr += psnr(clean_image[b], output_image[b])
                    net_ssim += ssim(clean_image[b], output_image[b], data_range=1)
                    
                for b in range(1): # Save a single image every time, but this can be changed
                    clean_image *= 255.0
                    noisy_image *= 255.0
                    output_image *= 255.0
                    
                    # Save clean and predicted clean images
                    if (images_folder is not None):
                        save_filename = os.path.join(images_folder, str(b)+".png")
                        save_test_images(save_filename, noisy_image[b], output_image[b], clean_image[b])
                
                progress_bar.set_description(desc=f"PSNR: {net_psnr/args.batch_size:5.2f}, SSIM: {net_ssim/args.batch_size:5.3f}")

                step += 1
                if (step % args.log_interval == 0):
                    print("")
                    val_time = time.perf_counter()
                    avg_psnr, avg_ssim, _, _ = evaluate(val_loader, diffusion, model, dist_util.dev(), images_folder, num_steps, num_channels, unet_model, 1)
                    net_val_time += time.perf_counter() - val_time
                    
                    logger.log(f"Epoch = {e:3d},  PSNR: {avg_psnr:5.2f},  SSIM: {avg_ssim:5.3f},  Net training time: {(time.perf_counter() - start_time - net_val_time):.1f}s,  Net validation time: {net_val_time:.1f}s")

                    if best_psnr < avg_psnr:
                        best_psnr = avg_psnr
                        logger.log(f"Epoch {e} | New best PSNR: {avg_psnr:5.2f}, SSIM: {avg_ssim:5.3f}")
                        print(f"Epoch {e} | New best PSNR: {avg_psnr:5.2f}, SSIM: {avg_ssim:5.3f}")
                        save(mp_trainer, latest=False)
                    
                    save(mp_trainer, latest=True)

                if (step >= 8000):
                    break

            progress_bar.close()
            e += 1

    else:
        unet_model.convert_to_fp16()
        unet_model.eval()
        avg_psnr, avg_ssim, _, _ = evaluate(test_loader, diffusion, model, dist_util.dev(), images_folder, num_steps, num_channels, unet_model, args.batch_size, test=True, log=True)
        
        # Log average results
        logger.log("\nTesting complete")
        logger.log("Model: " + test_checkpoint)
        logger.log("Training dataset: " + training_args['train_dir'])
        logger.log("Testing dataset: " + args.test_dir)
        logger.log("Cycle spinning: " + str(args.cycle_spinning))


        # SAR Testing
        sar_test_log_folder = os.path.join(
            log_path,
            datetime.datetime.now().strftime(f"SAR_test_{args.test_dir.split('/')[1]}_%Y-%m-%d-%H-%M-%S"),
        )
        logger.configure(dir=sar_test_log_folder, log_suffix="_test", format_strs=["log","csv"])

        logger.log("Training dataset: " + training_args['train_dir'])
        logger.log("Validation dataset: " + training_args['val_dir'])
        logger.log("Testing dataset: " + args.test_dir)
        logger.log("Cycle spinning: " + str(args.cycle_spinning))
        logger.log("DDPM model:" + test_checkpoint)
        logger.log("UNet model:" + unet_model_checkpoint)
        logger.log("Args: " + str(args))
        logger.log("Training args: " + str(training_args) + "\n")

        def sar_model(noisy_tensor):
            # Otherwise, get the predicted clean image as normal
            model_kwargs = {'noisy': noisy_tensor}

            pred_tensor = diffusion.p_sample_loop(
                            model,
                            noisy_tensor.shape,
                            clip_denoised=True,
                            model_kwargs=model_kwargs,
                        )
            
            pred_tensor = pred_tensor.transpose(0,1).float()
            pred_tensor = pred_tensor[:, -1*num_channels:]
            shape = pred_tensor.shape
            pred_tensor = pred_tensor.contiguous().view(shape[0], shape[1]*shape[2], shape[3], shape[4])

            return unet_model(torch.cat((noisy_tensor, pred_tensor), dim=1), t)

        evaluate_sar(sar_model, dist_util.dev(), training_args['in_channels'], training_args['large_size'])


def create_argparser(test=False):
    custom_defaults = dict(        
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
    defaults.update(default_args(test=test))
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def get_most_recent_log_folder(logs_dir):
    folders = [f for f in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, f))]
    if len(folders) == 0:
        raise FileNotFoundError(f"No log folders found in {logs_dir}")
    date_folders = [(folder, datetime.datetime.strptime(folder.split('_')[-1], '%Y-%m-%d-%H-%M-%S')) for folder in folders]
    most_recent_folder = max(date_folders, key=lambda x: x[1])[0]
    return os.path.join(logs_dir, most_recent_folder)


def parse_args_from_log(log_file):
    args_dict = {}
    try:
        with open(log_file, 'r') as file:
            for line in file:
                if line.startswith("Args: Namespace"):
                    args_str = line.strip().lstrip("Args: Namespace(").rstrip(")")
                    
                    # Regex to match key-value pairs, handling quoted values with commas
                    pattern = re.compile(r"(\w+)=('.*?'|\".*?\"|[^,]+)")
                    
                    matches = pattern.findall(args_str)
                    for key, value in matches:
                        # Remove the quotes if the value is a quoted string
                        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                            value = value[1:-1]
                        # Put the value into the correct format
                        try:
                            if value.lower() == 'true':
                                value = True
                            elif value.lower() == 'false':
                                value = False
                            elif value.isdigit():
                                value = int(value)
                            else:
                                try:
                                    value = float(value)
                                except ValueError:
                                    pass  # keep as string if it's neither int nor float
                        except ValueError:
                            pass
                        args_dict[key] = value
                    break
    except Exception as e:
        raise Exception(f"An error occurred while parsing the log file {log_file}: {e}")
    return args_dict


def save_test_images(img_name, *arrays):
    # Convert NumPy arrays to PIL images
    images = [Image.fromarray(array) for array in arrays]

    # Determine the size of the new image
    width, height = images[0].size
    border = 2
    total_width = width * len(images) + border * (len(images) - 1)
    total_height = height

    # Create a new blank image
    new_image = Image.new('L', (total_width, total_height))  # 'L' mode for grayscale

    # Paste the images into the new image with borders
    for i, image in enumerate(images):
        new_image.paste(image, (i * (width + border), 0))

    # Save the final image
    new_image.save(img_name)


def save(mp_trainer, latest=False):
    logger_dir = logger.get_dir()
    savetype = "latest" if latest else "best"

    state_dict = mp_trainer.master_params_to_state_dict(mp_trainer.master_params)
    filename = f"model_{savetype}.pt"

    with bf.BlobFile(bf.join(logger_dir, filename), "wb") as f:
        torch.save(state_dict, f)



def evaluate(loader, diffusion, model, device, images_dir, num_steps, num_channels, unet_model, batch_size, cycle_spinning=False, cycle_width=0, log=False, test=False):
    unet_model.eval()

    t = torch.tensor([1.0] * batch_size, device=dist_util.dev())

    net_psnr = 0.0 # sum PSNR metrics
    net_ssim = 0.0 # sum SSIM metrics
    net_mse = 0.0
    net_lpips = 0.0
    net_vifp = 0.0

    net_time = 0.0 # sum evaluation times
    
    with torch.no_grad():
        _, noisy_tensor, _ = next(iter(loader))
        itr_indexes = list(range(num_steps))

        metric_shape = (len(loader) * noisy_tensor.shape[0], len(itr_indexes))
        all_tensor_psnr = np.empty(metric_shape)
        all_tensor_ssim = np.empty(metric_shape)
        all_tensor_vifp = np.empty(metric_shape)
        all_tensor_lpips = np.empty(metric_shape)

        progress_bar = tqdm(loader, desc=f"[{'Test' if test else 'Validation'}] PSNR: 00.00/00.00, SSIM: 0.000/0.000", unit='batch')


        for batch_idx, data_tuple in enumerate(progress_bar):
            clean_tensor, noisy_tensor, image_filename = data_tuple
            clean_tensor = clean_tensor.to(device)
            noisy_tensor = noisy_tensor.to(device)
            
            # Reformat the images for metrics
            clean_image = ((clean_tensor + 1.0)* 127.5).clamp(0, 255.0)
            clean_image = torch.round(torch.mean(clean_image, dim=1)) / 255.0
            clean_image = clean_image.contiguous()

            batch_size = clean_tensor.shape[0]
            
            batch_start = time.perf_counter()
            
            if (cycle_spinning):
                first = True
                [_, _, num_rows, num_cols] = noisy_tensor.size()
                val_inputv = torch.empty_like(noisy_tensor).to(device)

                # Get number of cycle spins
                N = int(np.ceil(num_rows / cycle_width) * np.ceil(num_cols / cycle_width))

                # For each cycle (in both directions)
                for row in range(0, num_rows, cycle_width):
                    for col in range(0, num_cols, cycle_width):
                        # Execute the cycle spin
                        val_inputv[:,:,:row ,:col ] = noisy_tensor[:,:, num_rows-row:, num_cols-col:]
                        val_inputv[:,:, row:, col:] = noisy_tensor[:,:,:num_rows-row ,:num_cols-col ]
                        val_inputv[:,:, row:,:col ] = noisy_tensor[:,:,:num_rows-row , num_cols-col:]
                        val_inputv[:,:,:row , col:] = noisy_tensor[:,:, num_rows-row:,:num_cols-col ]

                        model_kwargs = {'noisy': val_inputv}

                        # Get the predicted clean image
                        sample = diffusion.p_sample_loop(
                                model,
                                val_inputv.shape,
                                clip_denoised=True,
                                model_kwargs=model_kwargs,
                            )

                        # Unspin the image and add to the averaged image
                        if (first):
                            pred_tensor = (1.0/N)*sample
                            first = False
                        else:
                            pred_tensor[:,:,:, num_rows-row:, num_cols-col:] = pred_tensor[:,:,:, num_rows-row:, num_cols-col:] + (1.0/N)*sample[:,:,:row ,:col ]
                            pred_tensor[:,:,:,:num_rows-row ,:num_cols-col ] = pred_tensor[:,:,:,:num_rows-row ,:num_cols-col ] + (1.0/N)*sample[:,:, row:, col:]
                            pred_tensor[:,:,:,:num_rows-row , num_cols-col:] = pred_tensor[:,:,:,:num_rows-row , num_cols-col:] + (1.0/N)*sample[:,:, row:,:col ]
                            pred_tensor[:,:,:, num_rows-row:,:num_cols-col ] = pred_tensor[:,:,:, num_rows-row:,:num_cols-col ] + (1.0/N)*sample[:,:,:row , col:]

            else:
                # Otherwise, get the predicted clean image as normal
                model_kwargs = {'noisy': noisy_tensor}

                pred_tensor = diffusion.p_sample_loop(
                                model,
                                noisy_tensor.shape,
                                clip_denoised=True,
                                model_kwargs=model_kwargs,
                            )
                
            # Find the elapsed time
            elapsed_time = time.perf_counter() - batch_start
            net_time += elapsed_time

            # if not test:
            #     pred_tensor = torch.unsqueeze(pred_tensor[-1], dim=0)

            pred_tensor = pred_tensor[itr_indexes]
            iterations = pred_tensor.shape[0]

            pred_image = ((pred_tensor + 1.0)* 127.5).clamp(0, 255.0)
            pred_image = torch.round(torch.mean(pred_image, dim=2)) / 255.0
            pred_image = pred_image.contiguous()

            for b in range(batch_size):
                all_tensor_lpips[batch_idx * batch_size + b] = compute_lpips_batch(clean_image[b].repeat(iterations,1,1) * 2.0 - 1.0, pred_image[:,b] * 2.0 - 1.0)

            pred_image_np = pred_image.cpu().numpy()
            clean_image_np = clean_image.cpu().numpy()

            max_psnr_index = [0] * batch_size
            max_psnr = [0.0] * batch_size
            for b in range(batch_size):
                idx = batch_idx * batch_size + b
                for i in range(iterations):
                    all_tensor_psnr[idx,i] = psnr(clean_image_np[b], pred_image_np[i,b])
                    all_tensor_ssim[idx,i] = ssim(clean_image_np[b], pred_image_np[i,b], data_range=1)
                    # if test:
                    #     all_tensor_vifp[idx,i] = vifp(clean_image_np[b], pred_image[i,b])

                max_psnr_index[b] = np.argmax(all_tensor_psnr[idx,:])
                max_psnr[b] = all_tensor_psnr[idx, max_psnr_index[b]]
                
            pred_tensor = pred_tensor.transpose(0,1).float()
            pred_tensor = pred_tensor[:, -1*num_channels:]
            shape = pred_tensor.shape
            pred_tensor = pred_tensor.contiguous().view(shape[0], shape[1]*shape[2], shape[3], shape[4])

            start_time = time.perf_counter()
            output = unet_model(torch.cat((noisy_tensor, pred_tensor), dim=1), t)
            elapsed_time = time.perf_counter() - start_time
            net_time += elapsed_time
            output = torch.mean(output, dim=1)

            output_image = ((output + 1.0)* 127.5).clamp(0, 255.0)
            output_image = torch.round(output_image) / 255.0
            output_image = output_image.contiguous()

            pred_image = output_image
                
            pred_image_np = pred_image.cpu().numpy()
            clean_image_np = clean_image.cpu().numpy()

            img_psnr = [0.0]*batch_size
            img_ssim = [0.0]*batch_size
            img_vifp = [0.0]*batch_size

            for b in range(batch_size):
                img_psnr[b] = psnr(clean_image_np[b], pred_image_np[b])
                img_ssim[b] = ssim(clean_image_np[b], pred_image_np[b], data_range=1)
                # img_vifp[b] = vifp(clean_image_np[b], pred_image_np[b])
            img_lpips = compute_lpips_batch(clean_image * 2.0 - 1.0, pred_image * 2.0 - 1.0)
            
            noisy_image = ((noisy_tensor + 1.0)* 127.5).clamp(0, 255.0)
            noisy_image = torch.round(torch.mean(noisy_image, dim=1)) / 255.0
            noisy_image = noisy_image.contiguous()

            noisy_image_np = noisy_image.cpu().numpy()
            
            clean_image_np *= 255.0
            noisy_image_np *= 255.0
            pred_image_np *= 255.0

            for i in range(batch_size):
                # Save clean and predicted clean images
                if (images_dir is not None):
                    save_filename = os.path.basename(image_filename[i])
                    save_filename = os.path.join(images_dir, save_filename)
                    save_test_images(save_filename, noisy_image_np[i], pred_image_np[i], clean_image_np[i])

                if log:
                    num_digits = int(math.log10(len(loader))) + 1 if len(loader) != 0 else 1

                    status = f"[{(batch_idx+1):>{num_digits}d}/{len(loader)}]  "
                    status += f"PSNR: {img_psnr[i]:5.2f} dB,  "
                    status += f"SSIM: {img_ssim[i]:5.3f},  "
                    status += f"Max PSNR: {max_psnr[i]:5.2f} at {max_psnr_index[i]:2d}  |  "
                    status += f"[{elapsed_time/batch_size:3.1f}s]  {os.path.basename(image_filename[i])}"
                    logger.log(status)
                    
                    logger.logkv('Time', elapsed_time/batch_size)
                    logger.logkv('PSNR', img_psnr[i])
                    logger.logkv('SSIM', img_ssim[i])
                    logger.logkv('LPIPS', img_lpips[i])
                    logger.logkv('VIFP', img_vifp[i])
                    logger.dumpkvs()

            batch_psnr = sum(img_psnr) / batch_size
            batch_ssim = sum(img_ssim) / batch_size
            batch_lpips = sum(img_lpips) / batch_size
            net_psnr += batch_psnr
            net_ssim += batch_ssim
            net_lpips += batch_lpips

            if log:
                batch_vifp = sum(img_vifp) / batch_size
                net_vifp += batch_vifp

            progress_bar.set_description(desc=f"[{'Test' if test else 'Validation'}] PSNR: {batch_psnr:5.2f}/{(net_psnr/(batch_idx+1)):5.2f}, SSIM: {batch_ssim:5.3f}/{(net_ssim/(batch_idx+1)):5.3f}")

        progress_bar.close()

    if not test:
        unet_model.train()

    net_time /= len(loader)
    net_psnr /= len(loader)
    net_ssim /= len(loader)
    net_lpips /= len(loader)
    net_mse /= len(loader)
    
    if log:
        net_vifp /= len(loader)

        logger.log(f"\nAverage elapsed time: {net_time:.3f} s")
        logger.log(f"Average PSNR: {net_psnr:.3f} dB")
        logger.log(f"Average SSIM: {net_ssim:.3f}")
        logger.log(f"Average MSE: {net_mse:2.2e}")
        logger.log(f"Average LPIPS: {net_lpips:.4f}")
        logger.log(f"Average VIFP: {net_vifp:.4f}")

    # Plot the metrics over all DDPM iterations
    psnr_means = np.mean(all_tensor_psnr, axis=0)
    max_psnr_index = np.argmax(psnr_means)
    max_psnr = psnr_means[max_psnr_index]
    if test:
        logger.log(f"Average best PSNR: {max_psnr:5.2f} dB with average index of {max_psnr_index}.")

    ssim_means = np.mean(all_tensor_ssim, axis=0)
    max_ssim_index = np.argmax(ssim_means)
    max_ssim = ssim_means[max_ssim_index]
    
    if test:
        vifp_means = np.mean(all_tensor_vifp, axis=0)
        min_vifp_index = np.argmin(vifp_means)
        min_vifp = vifp_means[min_vifp_index]
    
    lpips_means = np.mean(all_tensor_lpips, axis=0)
    min_lpips_index = np.argmin(lpips_means)
    min_lpips = lpips_means[min_lpips_index]

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(itr_indexes, psnr_means, label='PSNR', marker='o', color='blue', markersize=4)
    ax1.axhline(y=net_psnr, linestyle='--', color='indigo', label=f'AGT PSNR ({net_psnr:5.2f})')
    ax1.scatter(itr_indexes[max_psnr_index], max_psnr, color='blue', s=50, zorder=5, label='Max PSNR')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('PSNR (dB)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis
    ax2 = ax1.twinx()
    ax2.plot(itr_indexes, ssim_means, label='SSIM', marker='o', color='orange', markersize=4)
    ax2.axhline(y=net_ssim, linestyle='--', color='brown', label=f'AGT SSIM ({net_ssim:4.2f})')
    ax2.scatter(itr_indexes[max_ssim_index], max_ssim, color='orange', s=50, zorder=5, label='Max SSIM')
    
    # if test:
    #     ax2.plot(itr_indexes, vifp_means, label='VIFP', marker='o', color='green', markersize=4)
    #     ax2.scatter(itr_indexes[min_vifp_index], min_vifp, color='green', s=50, zorder=5, label='Min VIFP')
    
    ax2.plot(itr_indexes, lpips_means, label='LPIPS', marker='o', color='red', markersize=4)
    ax2.axhline(y=net_lpips, linestyle='--', color='pink', label=f'AGT LPIPS ({net_lpips:4.2f})')
    ax2.scatter(itr_indexes[min_lpips_index], min_lpips, color='red', s=50, zorder=5, label='Min LPIPS')

    ax2.set_ylabel('SSIM / VIFP / LPIPS', color='black')
    ax2.set_ylim(0, 0.9)  # Set the maximum y-value
    ax2.tick_params(axis='y', labelcolor='black')

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.1, 0.7), title="Legend")

    # Adjust plot to fit the legend
    plt.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.75, top=0.9, bottom=0.1)

    # Add title
    plt.title('Change in PSNR and SSIM per DDPM iteration.')

    # Save the plot as a PNG file
    plt.savefig(os.path.join(logger.get_dir(), 'PSNR_plot.png'))
    plt.close(fig)

    return net_psnr, net_ssim, net_time, max_psnr


def compute_lpips_batch(sr_tensors, gt_tensors):
    # Must be pytorch tensors on the GPU between [-1, 1]

    # Ensure the input tensors are of the shape (N, C, H, W) and have 3 channels
    sr_tensors = sr_tensors.unsqueeze(1).repeat(1, 3, 1, 1)
    gt_tensors = gt_tensors.unsqueeze(1).repeat(1, 3, 1, 1)
    
    # Compute LPIPS
    lpips_values = lpips_model(sr_tensors, gt_tensors)
    
    # Return the mean LPIPS value across the batch
    lpips_values = lpips_values.squeeze().tolist()
    if not isinstance(lpips_values, list):
        lpips_values = [lpips_values]
    return lpips_values


def save_test_images(img_name, *arrays):
    # Convert NumPy arrays to PIL images
    images = [Image.fromarray(array) for array in arrays]

    # Determine the size of the new image
    width, height = images[0].size
    border = 2
    total_width = width * len(images) + border * (len(images) - 1)
    total_height = height

    # Create a new blank image
    new_image = Image.new('L', (total_width, total_height))  # 'L' mode for grayscale

    # Paste the images into the new image with borders
    for i, image in enumerate(images):
        new_image.paste(image, (i * (width + border), 0))

    # Save the final image
    new_image.save(img_name)


if __name__ == "__main__":
    main()
