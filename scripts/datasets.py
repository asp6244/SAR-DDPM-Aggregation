import numpy as np
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import get_worker_info


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results
        
class RandomHorizontalFlip:
    def __call__(self, img, rand_num):
        if rand_num % 2 == 0:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    
class RandomVerticalFlip:
    def __call__(self, img, rand_num):
        if rand_num % 2 == 0:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img

class RandomRotation90:
    def __call__(self, img, rand_num):
        rotations = [0, 90, 180, 270]
        return img.rotate(rotations[rand_num % 4])
    
class RandomCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img, rand_num1, rand_num2):
        width, height = img.size
        crop_width, crop_height = self.crop_size

        if width < crop_width or height < crop_height:
            raise ValueError("Crop size must be smaller than image size")

        # Calculate the number of possible crop positions
        max_x = width - crop_width
        max_y = height - crop_height

        # Use the random number to determine the crop position
        random_x = rand_num1 % (max_x + 1)
        random_y = rand_num2 % (max_y + 1)

        return img.crop((random_x, random_y, random_x + crop_width, random_y + crop_height))


class SynthSARDataset(Dataset):
    def __init__(self, dataset_path, train=False, num_channels=1, crop_size=(256, 256), length=-1, seed=None):
        super().__init__()
        
        self.loaded_rng = False
        self.train = train
        self.num_channels = num_channels
        self.seed = seed

        self.gamma_rng = None

        # Process the SAR dataset
        rng = np.random.default_rng(seed) # Image selection should only depend on seed
        all_images = _list_image_files_recursively(dataset_path)
        if 0 < length < len(all_images):
            self.images_list = rng.choice(all_images, size=length, replace=False).tolist()
        else:
            rng.shuffle(all_images)
            self.images_list = all_images

        if train:
            self.rng_rng = np.random.default_rng(seed)
            self.transform_rng = None
            self.horizontal_flip = RandomHorizontalFlip()
            self.vertical_flip = RandomVerticalFlip()
            self.rotation = RandomRotation90()
            self.crop = RandomCrop(crop_size)
        else:
            self.center_crop = transforms.CenterCrop(size=crop_size)

    def _load_rng(self):
        # Access worker information
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        
        # Set seeds based on a deterministic random number and the worker ID
        rand_nums = self.rng_rng.integers(low=0, high=np.iinfo(np.int32).max, size=2)
        seed_seq = np.random.SeedSequence([self.seed, worker_id, rand_nums]).spawn(2)
        self.gamma_rng = np.random.default_rng(seed_seq[0])
        self.transform_rng = np.random.default_rng(seed_seq[1])
        
        self.loaded_rng = True

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        if self.train and not self.loaded_rng:
            self._load_rng()

        image_filename = self.images_list[idx]
        image = Image.open(image_filename).convert('L')

        if self.train:
            # Apply random crop for training set
            rand_nums = self.transform_rng.integers(low=0, high=np.iinfo(np.int32).max, size=5)

            image = self.horizontal_flip(image, rand_nums[0])
            image = self.vertical_flip(image, rand_nums[1])
            image = self.rotation(image, rand_nums[2])
            image = self.crop(image, rand_nums[3], rand_nums[4])
        else:
            image = self.center_crop(image)
            # Ensure deterministic noise for test and validation
            self.gamma_rng = np.random.default_rng(np.random.SeedSequence([self.seed, idx]))
        
        clean_image = np.float32(image)
        clean_image = clean_image[np.newaxis,:,:]
        
        noisy_array = (clean_image / 255.0)**2
        gamma_noise = self.gamma_rng.gamma(size=noisy_array.shape, shape=1.0, scale=1.0).astype(noisy_array.dtype)
        noisy_array = np.clip(np.sqrt(noisy_array * gamma_noise), 0.0, 1.0)

        clean_array = np.round(clean_image) / 127.5 - 1.0
        noisy_array = np.round(noisy_array * 255.0) / 127.5 - 1.0
        
        if (self.num_channels > 1):
            clean_array = np.repeat(clean_array, self.num_channels, axis=0)
            noisy_array = np.repeat(noisy_array, self.num_channels, axis=0)
        
        return torch.tensor(clean_array), torch.tensor(noisy_array), image_filename


class SARDataset(Dataset):
    def __init__(self, sar_images_list_path, num_channels=1, crop_size=(256, 256)):
        super().__init__()
        
        self.sar_images_list_path = sar_images_list_path
        self.num_channels = num_channels
        self.images_list = []

        # Process the SAR dataset
        with open(sar_images_list_path, "r") as file:
            lines = file.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            self.images_list.append(line.split())

        self.center_crop = transforms.CenterCrop(size=crop_size)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_data = self.images_list[idx]
        image_filename = image_data[0]
        hmx1, hmy1, hmx2, hmy2 = map(int, image_data[1:5])  # Coordinates for the homogeneous box
        htx1, hty1, htx2, hty2 = map(int, image_data[5:9])  # Coordinates for the heterogeneous box
        image_filename = os.path.join(os.path.dirname(self.sar_images_list_path), image_filename)
        
        image = Image.open(image_filename)
        image = self.center_crop(image)
        
        image_array = np.float32(image)
        if len(image_array.shape) == 3:
            image_array = 0.2989 * image_array[:, :, 0] + 0.5870 * image_array[:, :, 1] + 0.1140 * image_array[:, :, 2]
        image_array = image_array[np.newaxis,:,:]

        image_array = np.round(image_array) / 127.5 - 1.0
        
        if (self.num_channels > 1):
            image_array = np.repeat(image_array, self.num_channels, axis=0)
        
        fname = image_filename.split("/")
        fname = fname[-1][:-4]

        return torch.tensor(image_array), (hmx1, hmy1, hmx2, hmy2), (htx1, hty1, htx2, hty2), fname
    