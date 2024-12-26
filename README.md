# Sample Aggregation Methods in Denoising Diffusion Models for SAR Despeckling

Code for the paper Sample Aggregation Methods in Denoising Diffusion Models for SAR Despeckling


## To train and test the old and new SAR-DDPM models:

- Download the weights 64x64 -> 256x256 upsampler from [here](https://github.com/openai/guided-diffusion).

- Create a folder ./weights and place the downloaded weights in the folder.

- Specify the paths to your training, validation, and testing data paths in ./scripts/parameters.py

  - The dataset paths can be specified as the root of a tree structure containing the images.

  - Real SAR images along with the heterogeneous and homogeneous regions for evaluation are specified in text files ./scripts/test_util.py in the following form:
    - [image path] [first homogeneous coordinate x] [first homogeneous coordinate y] [second homogeneous coordinate x] [second homogeneous coordinate y] [first heterogeneous coordinate x] [first heterogeneous coordinate y] [second heterogeneous coordinate x] [second heterogeneous coordinate y]
    - e.g.: ROIs1158_spring/s1_0/ROIs1158_spring_s1_0_p441.png 130 196 156 215 47 204 90 235

- Set up your environment. This may need to be changed depending on your GPU and driver.
```bash
conda create -n SAR-DDPM -y --file dependencies_conda.txt
conda activate SAR-DDPM
conda install conda-forge::cudatoolkit-dev -y
pip install -r dependencies_pip.txt --extra-index-url https://download.pytorch.org/whl/cu113
pip install -e .
```

- Train and evaluate the models
```
# Be sure to edit the parameters found in ./scripts/parameters.py.
# Any command line arguments will override these.

# Train the pretrained U-Net model with the parameters found in the "Train" section of ./scripts/parameters.py.
python ./scripts/sarddpm_train.py

# Test the most recently trained model with the parameters found in the "Test" section of ./scripts/parameters.py.
# The model being evaluated can be changed in the parameters file or with the argument 'training_log_folder'.
python ./scripts/sarddpm_test.py

# Train (and test) the second U-Net used in the aggregation method.
# Modify the four variables at the top of this file, along with the train/test variables from ./scripts/parameters.py.
python ./scripts/sarddpm_retest.py
```

### Acknowledgement:
This code is based on the implementation by [SAR-DDPM](https://github.com/malshaV/SAR_DDPM).
Their code closely follows the DDPM implementation in [guided-diffusion](https://github.com/openai/guided-diffusion).

### Citation:
```
@ARTICLE{paul2024sarddpmagg,
  author={Paul, Alec and Savakis, Andreas},
  journal={Sensors}, 
  title={Sample Aggregation Methods in Denoising Diffusion Models for SAR Despeckling}, 
  year={2025}}
```
