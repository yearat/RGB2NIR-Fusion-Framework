
# **High-Resolution Near-Infrared Image Generation via End-to-End Fusion of RGB and Low-Resolution NIR Modalities**
> author1 and author2

## Abstract

Near-infrared (NIR) imaging plays a crucial role in applications such as vegetation monitoring, remote sensing, and environmental analysis. However, NIR imaging systems are often expensive and limited in resolution compared to conventional RGB sensors. To address this limitation, we propose an end-to-end deep learning framework that generates high-resolution (HR) NIR images by fusing high-resolution RGB images with low-resolution (LR) NIR inputs. The framework integrates a lightweight image-to-image (I2I) backbone for extracting rich spatial features from the RGB domain, dual feature extractors operating in a multi-scale paradigm, and a learnable fusion module that adaptively combines spatial and spectral information. A decoder network then reconstructs the fused features into the final HR NIR output. Extensive experiments on paired aerial RGB–NIR crop datasets demonstrate that the proposed method significantly outperforms standalone RGB-to-NIR translation and conventional LR-to-HR NIR super-resolution approaches. Even when the NIR input is 32 times smaller in resolution than the RGB image, the framework consistently produces superior results in terms of PSNR and SSIM. Furthermore, results show that 8× and 4× downsampled NIR inputs yield nearly identical performance, confirming the framework’s robustness and efficiency. The modular design of the proposed architecture also enables flexibility for future improvements and provides a practical solution for cost-effective high-quality NIR image generation.

## Setup
The follwoing repository is based on the Pytorch implementation of LYT-Net which can be found here: [https://github.com/albrateanu/LYT-Net]

- Make Conda Environment

```bash
conda  create  -n  LYT_Torch  python=3.9  -y
conda  activate  LYT_Torch
```

- Install Dependencies

```bash

conda  install  pytorch  torchvision  torchaudio  pytorch-cuda=11.8  -c  pytorch  -c  nvidia
pip  install  matplotlib  scikit-learn  scikit-image  opencv-python  yacs  joblib  natsort  h5py  tqdm  tensorboard
pip  install  einops  gdown  addict  future  lmdb  numpy  pyyaml  requests  scipy  yapf  lpips  thop  timm

```

### Prepare Datasets


<summary>

<b>Datasets should be organized as follows:</b>

</summary>

  

```

|--dataset

| |--Canola

| | |--Train

| | | |--RGB

| | | | ...

| | | |--NIR

| | | | ...

| | |--Test

| | | |--RGB

| | | | ...

| | | |--NIR

| | | | ...

```
### Test

You can test the model using the following commands. Pre-trained weights are available at ```trained_weights``` folder. GT Mean evaluation is <b>disabled</b> in our experiments but can be activated by setting the boolean flag ```gt_mean=True``` in the ```compute_psnr()``` method under the ```test.py``` file.


  

```bash

python  test.py

```

  

**Note:** Please modify the dataset and trained weight paths in ```test.py``` as per your requirements.
