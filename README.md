# SamRobNODDI

SamRobNODDI is a q-space sampling-augmented continuous representation learning framework for robust and generalized neurite orientation dispersion and density imaging (NODDI) parameter estimation from diffusion MRI (dMRI).

The framework combines uniform subsampling (US) and dynamic random subsampling (DRS) during training, represents diffusion signals using spherical harmonics (SH) coefficients, and employs a sampling consistency loss to encourage consistent predictions across different q-space sampling schemes. The released implementation uses a modified ResNet18 backbone.

## Method overview

- **Input representation:** sixth-order SH coefficients are computed separately for the two diffusion shells used by the implementation and concatenated as network input.
- **Sampling augmentation:** the training pipeline uses a fixed uniform-sampling branch and a dynamically randomized-sampling branch.
- **Consistency learning:** the training objective combines supervised losses from the two branches with a consistency term between their predictions.
- **Prediction:** the network estimates three NODDI-derived parameters: intracellular volume fraction (Vic), isotropic volume fraction (Viso), and orientation dispersion (OD).

For the methodological details and experimental settings, please refer to the paper cited below.

## Repository contents

```text
SamRobNODDI/
├── config/                         # Model and training configuration
├── model/                          # Network implementation
├── utils/                           # SH and other utility functions
├── dataset_sh_x.py                 # Dataset loading and SH feature generation
├── dwi_generate_2D_patch_all.py    # Training-data patch preparation
├── dwi_generate_2D_patch_val_test.py # Validation/test patch preparation
├── train.py                        # Training entry point
├── test.py                         # Testing/inference entry point
├── train_util.py                   # Training and evaluation utilities
├── evaluation_mean_std.py          # Evaluation utility
└── install.sh                      # Environment installation script
```

## Requirements and installation

The provided `install.sh` records the package versions used by the original environment, including PyTorch 1.7.1 with CUDA 10.2 and PyTorch Geometric dependencies.

Create and activate a compatible Conda environment, then run:

```bash
bash install.sh
```

The script installs the dependencies explicitly listed in it. Depending on your system and the code paths you use, additional Python packages imported by the project (for example, NumPy, SciPy, DIPY, NiBabel, Matplotlib, pandas, and h5py) may also need to be installed. CUDA, PyTorch, and PyTorch Geometric versions must be mutually compatible. The original installation script is provided as a reference; adjust it for your hardware and software environment if necessary.

## Data preparation

This repository contains code for data preparation and model training/testing; it does **not** include the HCP dataset or preprocessed dMRI data.

The data preparation scripts are:

- `dwi_generate_2D_patch_all.py`
- `dwi_generate_2D_patch_val_test.py`

The dataset loader in `dataset_sh_x.py` expects preprocessed input/target files in the directory structure and naming convention used by the implementation. It also reads the corresponding HCP gradient tables (`bvals` and `bvecs`) and, in the current code, contains an absolute path to the HCP subject directory.

**Before running the scripts, inspect and edit the data paths, subject identifiers, file naming conventions, and output directories in the scripts and dataset loader to match your local dataset organization.** You must have legitimate access to the imaging data and comply with the dataset provider's terms of use.

## Training

The default configuration is:

```text
config/resnet_config.py
```

It specifies a ResNet backbone with 56 input channels and 3 output channels, a batch size of 4, and an initial Adam learning rate of 0.001.

After preparing the training and validation data and updating the paths in `train.py`/`dataset_sh_x.py`, launch training from the repository root:

```bash
python train.py
```

To use another configuration file:

```bash
python train.py --config ./config/resnet_config.py
```

The training script's default data and checkpoint paths are machine-specific. Update them before running.

## Testing

After preparing the test data and updating the paths and checkpoint settings in `test.py`, run:

```bash
python test.py
```

The test script also uses machine-specific default paths. Please check its argument defaults and model-loading logic before running inference.

## Citation

If you use this code in your research, please cite: Taohui Xiao, Jian Cheng, Wenxin Fan, Enqing Dong, and Shanshan Wang, “SamRobNODDI: q-space sampling-augmented continuous representation learning for robust and generalized NODDI,” *Physics in Medicine & Biology*, 70, 175015 (2025). https://doi.org/10.1088/1361-6560/adf9b8.

## Notes

- This repository provides the implementation code, not the HCP data or a guarantee that the scripts will run without adapting local paths and dependencies.
- Please verify data organization, gradient-table correspondence, software compatibility, and checkpoint paths for your own environment.
