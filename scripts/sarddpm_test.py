"""
SAR-DDPM Inference on real SAR images.
"""

import argparse
import os
import re
from datetime import datetime
import torch
from torch.utils.data import DataLoader

from datasets import SynthSARDataset
from parameters import default_args
from guided_diffusion.test_util import (
    evaluate,
    evaluate_sar,
)
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    add_dict_to_argparser,
    set_seed,
)


def main():
    args = create_argparser().parse_args()

    if (args.seed is not None):
        set_seed(args.seed)

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
        datetime.now().strftime(f"test_{args.test_dir.split('/')[1]}_%Y-%m-%d-%H-%M-%S"),
    )
    logger.configure(dir=test_log_folder, log_suffix="_test", format_strs=["log","csv"])

    logger.log("Training dataset: " + training_args['train_dir'])
    logger.log("Validation dataset: " + training_args['val_dir'])
    logger.log("Testing dataset: " + args.test_dir)
    logger.log("Cycle spinning: " + str(args.cycle_spinning))
    logger.log("Pretrained checkpoint: " + training_args['resume_checkpoint'])

    # Get checkpoint path with the largest step index
    if not args.test_checkpoint:
        if args.model_to_use == "BEST":
            test_checkpoint = os.path.join(log_path, "model_best.pt")
        else:
            test_checkpoint = os.path.join(log_path, "model_max.pt")
        # test_checkpoint = os.path.join(log_path, "model_latest.pt") # Choose one
    else:
        test_checkpoint = args.test_checkpoint

    # Create images folder
    images_folder = os.path.join(test_log_folder, args.images_path)
    os.makedirs(images_folder, exist_ok=True)

    logger.log("Args: " + str(args))
    logger.log("Training args: " + str(training_args) + "\n")

    dist_util.setup_dist()

    logger.log("Creating model...")
    # Overwrite any training testing arguments with those from the testing arguments
    args_dict = vars(args)
    model, diffusion = sr_create_model_and_diffusion(
        **{k: args_dict[k] if k in args_dict.keys() else (training_args[k] if k in training_args.keys() else None) for k in sr_model_and_diffusion_defaults().keys()}
    )
    model.to(dist_util.dev())
    
    logger.log("Creating data loader...")
    test_dataset = SynthSARDataset(args.test_dir, train=False, num_channels=training_args['in_channels'], crop_size=(training_args['large_size'], training_args['large_size']), length=args.test_length, seed=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    logger.log("Loading model from checkpoint:" + test_checkpoint)
    dict_load = dist_util.load_state_dict(test_checkpoint, map_location=dist_util.dev())
    model.load_state_dict(dict_load, strict=False)
    if training_args['use_fp16']:
        model.convert_to_fp16()
            
    logger.log("Beginning testing...")

    avg_psnr, avg_ssim, avg_time, mse, max_psnr = evaluate(test_loader, diffusion, model, dist_util.dev(), images_folder, 
                                            cycle_spinning=args.cycle_spinning, cycle_width=args.cycle_width, log=True, test=True, use_ddim=args.use_ddim, sample_to_use=args.sample_to_use)
        
    # Log average results
    logger.log("\nTesting complete")
    logger.log("Model: " + test_checkpoint)
    logger.log("Training dataset: " + training_args['train_dir'])
    logger.log("Cycle spinning: " + str(args.cycle_spinning))


    # SAR Testing
    if args.sample_to_use == "MAX":
        print("Cannot compute max sample for SAR images.")
        exit(1)
    
    sar_test_log_folder = os.path.join(
        log_path,
        datetime.now().strftime(f"SAR_test_{args.test_dir.split('/')[1]}_%Y-%m-%d-%H-%M-%S"),
    )
    logger.configure(dir=sar_test_log_folder, log_suffix="_test", format_strs=["log","csv"])

    logger.log("Training dataset: " + training_args['train_dir'])
    logger.log("Validation dataset: " + training_args['val_dir'])
    logger.log("Testing dataset: " + args.test_dir)
    logger.log("Cycle spinning: " + str(args.cycle_spinning))
    logger.log("Loading model from checkpoint:" + test_checkpoint)
    logger.log("Args: " + str(args))
    logger.log("Training args: " + str(training_args) + "\n")
    if args.sample_to_use == "LAST":
        logger.log("!! Used the last sample !!\n")
    elif args.sample_to_use == "SWEEP":
        logger.log("!! Averaged the last 8 samples !!\n")
    else:
        exit(1)

    def sar_model(noisy_tensor):
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        
        # Otherwise, get the predicted clean image as normal
        model_kwargs = {'noisy': noisy_tensor}

        pred_tensor = sample_fn(
            model,
            noisy_tensor.shape,
            clip_denoised=True,
            model_kwargs=model_kwargs,
        )
            
        if args.sample_to_use == "LAST":
            pred_tensor = pred_tensor[-1]
        elif args.sample_to_use == "SWEEP":
            pred_tensor = torch.mean(pred_tensor[-8:], dim=0)
        else:
            print("Not a valid 'sample_to_use' string.")
            exit(1)

        return pred_tensor

    evaluate_sar(sar_model, dist_util.dev(), training_args['in_channels'], training_args['large_size'])


def create_argparser():
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, default_args(test=True))
    return parser


def get_most_recent_log_folder(logs_dir):
    folders = [f for f in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, f))]
    if len(folders) == 0:
        raise FileNotFoundError(f"No log folders found in {logs_dir}")
    date_folders = [(folder, datetime.strptime(folder.split('_')[-1], '%Y-%m-%d-%H-%M-%S')) for folder in folders]
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


if __name__ == "__main__":
    main()
