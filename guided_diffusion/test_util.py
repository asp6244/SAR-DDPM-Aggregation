import os
import sys
import time
import math
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from torch.utils.data import DataLoader

from sewar.full_ref import vifp
import lpips
import matplotlib.pyplot as plt

from . import logger

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the path to the other directory
other_dir = os.path.join(current_dir, "..", "scripts")
# Add the other directory to sys.path
sys.path.append(other_dir)
from datasets import SARDataset

# TODO: change these for your own data paths
sen12_sar_list_path = "../sen12/sar/sar_test_samples.txt"
hrsid_sar_list_path = "../HRSID_png/inshore_images/sar_test_samples.txt"

lpips_model = lpips.LPIPS(net='alex').cuda()

def evaluate(loader, diffusion, model, device, images_dir, cycle_spinning=False, cycle_width=0, log=False, test=False, use_ddim=False, sample_to_use="LAST"):
    sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
        )
    
    if log:
        if sample_to_use == "LAST":
            logger.log("!! Used the last sample !!\n")
        elif sample_to_use == "MAX":
            logger.log("!! Used the max sample !!\n")
        elif sample_to_use == "SWEEP":
            logger.log("!! Averaged the last 8 samples !!\n")
        else:
            exit(1)
    
    model.eval()

    net_psnr = 0.0 # sum PSNR metrics
    net_ssim = 0.0 # sum SSIM metrics
    net_mse = 0.0
    net_lpips = 0.0
    net_vifp = 0.0

    net_time = 0.0 # sum evaluation times
    
    with torch.no_grad():
        _, noisy_tensor, _ = next(iter(loader))
        if test:
            # Perform a single pass to warm up the model on the GPU
            test_tensor = torch.empty_like(noisy_tensor).to(device)
            
            test_output = sample_fn(
                model,
                test_tensor.shape,
                clip_denoised=True,
                model_kwargs={'noisy': test_tensor},
                progress=True,
            )
            
            itr_indexes = list(range(len(test_output)))
        else:
            itr_indexes = list(range(100)) # This may need to be changed

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
                        sample = sample_fn(
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

                pred_tensor = sample_fn(
                                model,
                                noisy_tensor.shape,
                                clip_denoised=True,
                                model_kwargs=model_kwargs,
                            )
                
            # Find the elapsed time
            elapsed_time = time.perf_counter() - batch_start
            net_time += elapsed_time

            pred_tensor = pred_tensor[itr_indexes]
            iterations = pred_tensor.shape[0]

            pred_image = ((pred_tensor + 1.0)* 127.5).clamp(0, 255.0)
            pred_image = torch.round(torch.mean(pred_image, dim=2)) / 255.0
            pred_image = pred_image.contiguous()

            for b in range(batch_size):
                all_tensor_lpips[batch_idx * batch_size + b] = compute_lpips_batch(clean_image[b].repeat(iterations,1,1) * 2.0 - 1.0, pred_image[:,b] * 2.0 - 1.0)
                
            batch_mse = F.mse_loss(clean_image, pred_image[-1], reduction='mean').item()

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
            
            if sample_to_use == "LAST":
                pred_image = pred_image[-1]
            elif sample_to_use == "MAX":
                pred_image = pred_image[max_psnr_index, range(batch_size)]
            elif sample_to_use == "SWEEP":
                pred_image = torch.mean(pred_image[-8:], dim=0)
            else:
                exit(1)
                
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

            net_mse += batch_mse
            if log:
                batch_vifp = sum(img_vifp) / batch_size
                net_vifp += batch_vifp

            progress_bar.set_description(desc=f"[{'Test' if test else 'Validation'}] PSNR: {batch_psnr:5.2f}/{(net_psnr/(batch_idx+1)):5.2f}, SSIM: {batch_ssim:5.3f}/{(net_ssim/(batch_idx+1)):5.3f}")

        progress_bar.close()

    if not test:
        model.train()

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
    ax1.scatter(itr_indexes[max_psnr_index], max_psnr, color='blue', s=50, zorder=5, label='Max PSNR')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('PSNR (dB)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis
    ax2 = ax1.twinx()
    ax2.plot(itr_indexes, ssim_means, label='SSIM', marker='o', color='orange', markersize=4)
    ax2.scatter(itr_indexes[max_ssim_index], max_ssim, color='orange', s=50, zorder=5, label='Max SSIM')
    
    # if test:
    #     ax2.plot(itr_indexes, vifp_means, label='VIFP', marker='o', color='green', markersize=4)
    #     ax2.scatter(itr_indexes[min_vifp_index], min_vifp, color='green', s=50, zorder=5, label='Min VIFP')
    
    ax2.plot(itr_indexes, lpips_means, label='LPIPS', marker='o', color='red', markersize=4)
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

    return net_psnr, net_ssim, net_time, net_mse, max_psnr


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


def evaluate_sar(model, device, num_channels, image_size):
    logger.log("Creating data loader...")

    sample_size = 3

    for sar_list_path, dataset_name in zip([sen12_sar_list_path, hrsid_sar_list_path], ["sen12", "hrsid"]):
        logger.log("Creating " + dataset_name + " data loader...")
        test_dataset = SARDataset(sar_list_path, num_channels=num_channels, crop_size=(image_size, image_size))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

        images_folder = os.path.join(logger.get_dir(), dataset_name + "_clean_samples")
        os.makedirs(images_folder, exist_ok=True)
        combined_images_folder = os.path.join(logger.get_dir(), dataset_name + "_clean_samples", "combined_samples")
        os.makedirs(combined_images_folder, exist_ok=True)

        logger.log("Beginning " + dataset_name + " testing...")
        running_enl = 0.0
        running_epd_h = 0.0
        running_epd_v = 0.0
        running_time = 0.0
        enl_cv = 0.0
        epd_cv = 0.0
        progress_bar = tqdm(test_loader, desc=f"[Test - {dataset_name}] ENL: 0000.00/0000.00, EPD-ROA_H: 0.000/0.000, EPD-ROA_V: 0.000/0.000", unit='image')

        # Testing loop
        with torch.no_grad():
            for batch_idx, (noisy_tensor, hm_coords, ht_coords, fnames) in enumerate(progress_bar):
                sample_enl = [0.0] * sample_size
                sample_epd_h = [0.0] * sample_size
                sample_epd_v = [0.0] * sample_size
                sample_time = [0.0] * sample_size
                for i in range(sample_size):
                    # Reformat the images for metrics
                    noisy_image = ((noisy_tensor + 1.0)* 127.5).clamp(0, 255.0)
                    noisy_image = torch.round(torch.mean(noisy_image, dim=1)) / 255.0
                    noisy_image = noisy_image.contiguous()

                    start_time = time.time()
                    pred_tensor = model(noisy_tensor.to(device))
                    sample_time[i] = (time.time() - start_time)

                    # Reformat the images for metrics
                    pred_image = ((pred_tensor + 1.0)* 127.5).clamp(0, 255.0)
                    pred_image = torch.round(torch.mean(pred_image, dim=1)) / 255.0
                    pred_image = pred_image.contiguous().cpu()

                    # Convert to PIL Image for saving
                    pil_image = transforms.ToPILImage()(pred_image)
                    # Save the combined image
                    image_path = os.path.join(images_folder, f'{fnames[0]}_{i}.png')
                    pil_image.save(image_path)

                    # Concatenate input, output, and target images horizontally
                    concatenated_image = torch.cat((noisy_image, pred_image), dim=2)
                    # Convert to PIL Image for saving
                    pil_image = transforms.ToPILImage()(concatenated_image)
                    # Save the combined image
                    image_path = os.path.join(combined_images_folder, f'{fnames[0]}_{i}.png')
                    pil_image.save(image_path)
                    
                    noisy_image, pred_image = noisy_image.numpy(), pred_image.numpy()

                    img_enl = enl(pred_image, hm_coords)
                    img_epd_h, img_epd_v = epd_roa(noisy_image, pred_image, ht_coords)

                    sample_enl[i] = img_enl
                    sample_epd_h[i] = img_epd_h
                    sample_epd_v[i] = img_epd_v

                mean_enl = np.mean(sample_enl)
                mean_epd_h = np.mean(sample_epd_h)
                mean_epd_v = np.mean(sample_epd_v)

                running_enl += mean_enl
                running_epd_h += mean_epd_h
                running_epd_v += mean_epd_v
                running_time += np.mean(sample_time)

                enl_cv += np.std(sample_enl)/mean_enl
                epd_cv += (np.std(sample_epd_h)/mean_epd_h + np.std(sample_epd_v)/mean_epd_v)/2
                
                enl_sample_string = f"{mean_enl:7.1f}["
                epd_h_sample_string = f"{mean_epd_h:5.3f}["
                epd_v_sample_string = f"{mean_epd_v:5.3f}["
                for i in range(sample_size):
                    enl_sample_string += f"{sample_enl[i]:7.1f},"
                    epd_h_sample_string += f"{sample_epd_h[i]:5.3f},"
                    epd_v_sample_string += f"{sample_epd_v[i]:5.3f},"
                enl_sample_string = enl_sample_string[:-1] + "]"
                epd_h_sample_string = epd_h_sample_string[:-1] + "]"
                epd_v_sample_string = epd_v_sample_string[:-1] + "]"

                logger.log(f"[{batch_idx+1:2d}/{len(test_loader):2d}] ENL: {enl_sample_string}, EPD-ROA_H: {epd_h_sample_string}, EPD-ROA_V: {epd_v_sample_string}, | {fnames[0]}")

                progress_bar.set_description(desc=f"[Test - {dataset_name}] ENL: {mean_enl:7.2f}/{(running_enl/(batch_idx+1)):7.2f}, EPD-ROA_H: {mean_epd_h:5.3f}/{(running_epd_h/(batch_idx+1)):5.3f}, EPD-ROA_V: {mean_epd_v:5.3f}/{(running_epd_v/(batch_idx+1)):5.3f}")

        progress_bar.close()
        
        logger.log(f"Testing for {dataset_name} complete!")
        logger.log(f"Average ENL: {(running_enl/(len(test_loader))):5.2f}")
        logger.log(f"Average ENL CV: {(enl_cv/(len(test_loader))):5.3f}")
        logger.log(f"Average EPD (H): {(running_epd_h/(len(test_loader))):5.3f}")
        logger.log(f"Average EPD (V): {(running_epd_v/(len(test_loader))):5.3f}")
        logger.log(f"Average EPD: {((running_epd_h+running_epd_v)/(2*len(test_loader))):5.3f}")
        logger.log(f"Average EPD CV: {(epd_cv/(len(test_loader))):5.3f}")
        logger.log(f"Average time (secs): {(running_time/(len(test_loader))):5.3f}\n")


def enl(pred_image, hm_coords):
    """
    Calculate the Equivalent Number of Looks (ENL) for a specified rectangle in the image.

    Parameters:
        pred_image (numpy.ndarray): A 2D numpy array with values between 0.0 and 255.0.
        hm_coords (tuple): A tuple (x1, y1, x2, y2) defining the rectangle in the image.
                           (x1, y1) is the top-left corner, and (x2, y2) is the bottom-right corner.

    Returns:
        float: The ENL value for the specified rectangle.
    """
    x1, y1, x2, y2 = hm_coords

    # Extract the region of interest (ROI)
    roi = pred_image[:, y1:y2+1, x1:x2+1]/255.0  # Ensure coordinates are inclusive

    # Calculate mean and standard deviation
    mean_value = np.mean(roi)
    std_dev = np.std(roi)

    # Avoid division by zero
    if std_dev == 0:
        raise ValueError("Standard deviation is zero; ENL cannot be calculated.")

    # Calculate ENL
    enl_value = (mean_value / std_dev) ** 2

    return enl_value


def epd_roa(noisy_image, pred_image, ht_coords):
    """
    Calculate the Edge Preservation Degree using the Ratio of Averages (EPD-ROA)
    for a specific region of interest (ROI) defined by ht_coords, for images in the
    format [1, height, width].

    Parameters:
        noisy_image (np.ndarray): Noisy image with shape [1, height, width], values between 0.0 and 1.0.
        pred_image (np.ndarray): Processed image with shape [1, height, width], values between 0.0 and 1.0.
        ht_coords (tuple): Tuple (x1, y1, x2, y2) defining the region of interest (ROI).

    Returns:
        (float, float): Edge Preservation Degree (EPD-ROA) (horizontal, vertical).
    """
    x1, y1, x2, y2 = ht_coords

    # Extract the region of interest (ROI) for the first (and only) channel
    noisy_roi = noisy_image[0, y1:y2+1, x1:x2+1]
    pred_roi = pred_image[0, y1:y2+1, x1:x2+1]

    # Initialize sum variables
    sum_noisy = 0
    sum_pseudoclean = 0
    # Horizontal direction: Process row by row
    for row in range(noisy_roi.shape[0]):
        for col in range(noisy_roi.shape[1] - 1):  # Exclude last pixel
            # Check for division by zero in noisy image
            if noisy_roi[row, col + 1] != 0:
                quotient_noisy = noisy_roi[row, col] / noisy_roi[row, col + 1]
            else:
                quotient_noisy = 0  # or choose another value like 1 depending on how you want to handle it

            # Check for division by zero in pseudoclean image
            if pred_roi[row, col + 1] != 0:
                quotient_pseudoclean = pred_roi[row, col] / pred_roi[row, col + 1]
            else:
                quotient_pseudoclean = 0  # or another value like 1

            sum_noisy += quotient_noisy
            sum_pseudoclean += quotient_pseudoclean
    # Avoid division by zero
    if sum_noisy == 0:
        horizontal_epd = 0
    else:
        horizontal_epd = sum_pseudoclean / sum_noisy
    
    # Initialize sum variables
    sum_noisy = 0
    sum_pseudoclean = 0
    # Vertical direction: Process column by column
    for col in range(noisy_roi.shape[1]):
        for row in range(noisy_roi.shape[0] - 1):  # Exclude last pixel
            # Check for division by zero in noisy image
            if noisy_roi[row + 1, col] != 0:
                quotient_noisy = noisy_roi[row, col] / noisy_roi[row + 1, col]
            else:
                quotient_noisy = 0  # or choose another value like 1

            # Check for division by zero in pseudoclean image
            if pred_roi[row + 1, col] != 0:
                quotient_pseudoclean = pred_roi[row, col] / pred_roi[row + 1, col]
            else:
                quotient_pseudoclean = 0  # or another value like 1

            sum_noisy += quotient_noisy
            sum_pseudoclean += quotient_pseudoclean
    # Avoid division by zero
    if sum_noisy == 0:
        vertical_epd = 0
    else:
        vertical_epd = sum_pseudoclean / sum_noisy

    # Return the ratio of summed quotients (EPD-ROA)
    return (horizontal_epd, vertical_epd)