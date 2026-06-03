# SP-BBDM for Solar AIA 304 Å to Hα Reconstruction

This repository provides the code and supporting materials for the PASP manuscript:

**[paper title]**

This work studies single-frame cross-band reconstruction from SDO/AIA 304 Å browse images to CHASE/HIS Hα preview images. The proposed method, SP-BBDM, is a two-stage structure-preserving Brownian Bridge diffusion framework. The first stage, denoted as SP-BBDM-grad, introduces a latent gradient consistency loss into a latent BBDM-f4 model. The second stage, denoted as SP-BBDM-REF, applies an ROI-guided residual refinement module to further improve local activity-related structures.

## Repository Structure

```text
.
├── SP-BBDM-grad/
│   └── Code for the first-stage SP-BBDM model with latent gradient loss
│
├── SP-BBDM-REF/
│   └── Code for the ROI-guided residual refinement module
│
├── scripts/
│   └── Training and running scripts for external baselines, including pix2pix and Restormer
│
├── preprocessing/
│   └── Solar image preprocessing scripts, including BBDM solar image preprocessing
│       and ROI extraction scripts for bright-region and filament-region patches
│
├── evaluation/
│   └── Evaluation scripts for full-disk metrics and ROI-specific metrics
│
└── README.md
```

## Folder Description

### `SP-BBDM-grad/`

This folder contains the code for the first-stage SP-BBDM model. This stage is based on a latent BBDM-f4 framework and introduces a latent gradient consistency loss to improve local structural constraints in the latent space.

### `SP-BBDM-REF/`

This folder contains the code for the second-stage ROI-guided residual refinement module. The REF module refines local solar activity structures based on the first-stage reconstructed Hα image, the corresponding AIA 304 Å image, and the ROI mask.

### `scripts/`

This folder contains training and running scripts for external baseline models, including pix2pix and Restormer. These scripts record the preprocessing settings, model configurations, and command-line settings used in the manuscript.

### `preprocessing/`

This folder contains preprocessing scripts for the solar image dataset. It includes the BBDM solar image preprocessing code and the ROI extraction code used to crop bright-region and filament-region patches for local evaluation.

### `evaluation/`

This folder contains the evaluation scripts used in the manuscript. It includes scripts for full-disk PSNR, SSIM, and LPIPS evaluation, as well as ROI-specific metric scripts for Bright-region Intensity Error and Filament Contrast Error.

## Data Sources

The AIA 304 Å images used in this work were obtained from the NASA/SDO AIA/HMI Browse Data service:

* SDO data service: https://sdo.gsfc.nasa.gov/

The Hα images were obtained from the CHASE/HIS RSM-mode preview images provided by the Solar Science Data Center at Nanjing University (SSDC-NJU):

* SSDC-NJU CHASE data service: https://ssdc.nju.edu.cn/NdchaseSatellite

This work uses publicly available browse/preview images rather than the original full-resolution science FITS products.

## Large Files

The trained checkpoints, processed test datasets, and the VQGAN-f4 file are not stored directly in this GitHub repository due to file-size limitations. They are provided through the following external download link:

* Baidu Netdisk: SP-BBDM: https://pan.baidu.com/s/51GOqBO6_LoOVb8atCuN3MA


The external archive contains:

1. `checkpoint/`
   Trained model checkpoints for SP-BBDM-grad and SP-BBDM-REF.

2. `test-data/`
   Processed AIA 304 Å and Hα test images used to reproduce the reported full-disk and ROI evaluation results.

3. `2025-2026-testdata/`
   Additional 2025–2026 temporal hold-out samples used for the failure-case and temporal generalization analysis.

4. `vq-f4/`
   The VQGAN-f4 file used by the latent BBDM-f4 framework.

If the external download link is inaccessible, these large files are available from the corresponding author upon reasonable request.

## Reproducibility

This repository and the external large-file archive provide the materials needed to reproduce the main evaluation results reported in the manuscript, including:

* first-stage SP-BBDM-grad inference;
* second-stage SP-BBDM-REF refinement;
* full-disk PSNR, SSIM, and LPIPS evaluation;
* ROI extraction for bright-region and filament-region patches;
* ROI-specific evaluation, including Bright-region Intensity Error and Filament Contrast Error;
* external baseline settings for pix2pix and Restormer;
* failure-case analysis on the 2025–2026 temporal hold-out samples.

The full training image products and the complete training-pair metadata are not redistributed in this release. The dataset was constructed from public AIA 304 Å browse images and CHASE/HIS Hα preview images through temporal matching, filtering, and preprocessing. Instead of redistributing the full training dataset, this repository provides the original data sources, preprocessing information, processed test data, model checkpoints, and evaluation code so that the reported test-set metrics and failure-case analysis can be reproduced.

## Evaluation

### Full-disk metrics

The full-disk evaluation scripts in `evaluation/` can be used to calculate PSNR, SSIM, and LPIPS between generated Hα images and ground-truth Hα images.

### ROI extraction

The ROI extraction scripts in `preprocessing/` are used to crop bright-region and filament-region patches. ROI locations are detected from the ground-truth Hα images, and the same ROI coordinates are applied to the generated images.

### ROI-specific metrics

The ROI metric scripts in `evaluation/` compute the following local metrics:

* MAE;
* Grad-MAE;
* Bright-region Intensity Error;
* Filament Contrast Error.

These scripts are used to reproduce the ROI evaluation results reported in the manuscript.

## External Baselines

The `scripts/` folder contains the training and running commands for the external baselines used in the manuscript:

* pix2pix;
* Restormer.

The original reference implementations and papers should be cited together with this repository. This repository provides the exact settings used in this study for reproducibility.

## Software and References

The experiments were implemented mainly in Python and PyTorch. The main software packages and reference implementations used in this work include:

* PyTorch;
* LPIPS;
* pix2pix;
* BBDM;
* Restormer;
* NAFNet;
* NumPy;
* OpenCV or scikit-image;
* Matplotlib.

Please cite the corresponding papers and software references if you use this repository or the released materials.

## Contact

For questions about the code, checkpoints, or data access, please contact the corresponding author of the manuscript.
