# MK-UNet

Official Pytorch implementation of the paper [MK-UNet: Multi-kernel Lightweight CNN for Medical Image Segmentation](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/papers/Rahman_MK-UNet_Multi-kernel_Lightweight_CNN_for_Medical_Image_Segmentation_ICCVW_2025_paper.pdf) published in ICCV 2025 CVAMD
[Md Mostafijur Rahman](https://mostafij-rahman.github.io/), [Radu Marculescu](https://radum.ece.utexas.edu/)
<p>The University of Texas at Austin</p>

[ARXIV](https://arxiv.org/abs/2509.18493) | [PAPER](https://openaccess.thecvf.com/content/ICCV2025W/CVAMD/papers/Rahman_MK-UNet_Multi-kernel_Lightweight_CNN_for_Medical_Image_Segmentation_ICCVW_2025_paper.pdf) | [Code](https://github.com/SLDGroup/MK-UNet)

#### 🔍 **Check out our CVPR 2025 paper! [EffiDec3D](https://github.com/SLDGroup/EffiDec3D)**
#### 🔍 **Check out our CVPR 2024 paper! [EMCAD](https://github.com/SLDGroup/EMCAD)** 
#### 🔍 **Check out our CVPRW 2024 paper! [PP-SAM](https://github.com/SLDGroup/PP-SAM)** 
#### 🔍 **Check out our WACV 2024 paper! [G-CASCADE](https://github.com/SLDGroup/G-CASCADE)**
#### 🔍 **Check out our MIDL 2023 paper! [MERIT](https://github.com/SLDGroup/MERIT)**
#### 🔍 **Check out our WACV 2023 paper! [CASCADE](https://github.com/SLDGroup/CASCADE)**


## Architecture

<p align="center">
<img src="EMCAD_architecture.jpg" width=100% height=40% 
class="center">
</p>

## Quantitative Results

## Qualitative Results

## Usage:
### Recommended environment:
**Please run the following commands.**
```
conda create -n mkunetenv python=3.8
conda activate mkunetenv

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

pip install -r requirements.txt

```

### Data preparation:

- **ClinicDB dataset:**
Download the splited polyp datasets from [Google Drive](https://drive.google.com/drive/folders/1FPJr5f91uUCikxMvkwtZSEnYHemTZq1P?usp=share_link) and move into './data/polyp/' folder.

- **ColonDB dataset:**
Download the splited polyp datasets from [Google Drive](https://drive.google.com/drive/folders/1u4_8dMztnEBUaX-w3XfUR3jXLBhpccPA?usp=share_link) and move into './data/polyp/' folder.

### Training:
```
cd into MK-UNet 
```

### Testing:
```
cd into MK-UNet 
```

## Acknowledgement
We are very grateful for these excellent works [EMCAD](https://github.com/SLDGroup/EMCAD), [CASCADE](https://github.com/SLDGroup/CASCADE), [MERIT](https://github.com/SLDGroup/MERIT), [G-CASCADE](https://github.com/SLDGroup/G-CASCADE), [PP-SAM](https://github.com/SLDGroup/PP-SAM), [PraNet](https://github.com/DengPingFan/PraNet), and [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT), which have provided the basis for our framework.

## Citations

``` 
@inproceedings{rahman2025mk,
  title={Mk-unet: Multi-kernel lightweight cnn for medical image segmentation},
  author={Rahman, Md Mostafijur and Marculescu, Radu},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1042--1051},
  year={2025}
}
```
