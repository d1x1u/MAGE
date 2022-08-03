# MAGE: Multisource Attention Networks with Discriminative Graph and Informative Entities for Classification of Hyperspectral and LiDAR Data

## Description
This is the official PyTorch implementation of the MAGE paper.

This repository will be fully completed after the article is accepted.

Checkpoints are released.

## Prerequisites
- Environment

- Dataset
  - [baiduwangpan](https://pan.baidu.com/s/1aeVm8dCaw9xEBFV6vRE5qQ), extraction code: **d7x3**
  - [Google Drive](https://drive.google.com/file/d/18MBuCG-sHYdNEpbB0J7DxKx1LauYUX-F/view?usp=sharing)
- Checkpoints
  - [baiduwangpan](https://pan.baidu.com/s/1yuZ4PhqgjBB172y_LYpSUg), extraction code: **safz**
  - [Google Drive](https://drive.google.com/drive/folders/1uskMQo5APOito0RNS-rfpaXBvA9ytaXg?usp=sharing)

## command
- Train
```python
python main.py
```
- Evaluation
```python
python test.py
```

## Baseline
- [FusAtNet: Dual Attention Based SpectroSpatial Multimodal Fusion Network for Hyperspectral and LiDAR classification](https://openaccess.thecvf.com/content_CVPRW_2020/html/w6/Mohla_FusAtNet_Dual_Attention_Based_SpectroSpatial_Multimodal_Fusion_Network_for_Hyperspectral_CVPRW_2020_paper.html)
- [S²ENet: Spatial–Spectral Cross-Modal Enhancement Network for Classification of Hyperspectral and LiDAR Data](https://ieeexplore.ieee.org/abstract/document/9583936)
- [Deep Encoder–Decoder Networks for Classification of Hyperspectral and LiDAR Data](https://ieeexplore.ieee.org/abstract/document/9179756)
- [More diverse Means Better](https://ieeexplore.ieee.org/document/9174822/)
- [Learning from labeled and unlabeled data with label propagation](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.14.3864&rep=rep1&type=pdf)

## Data
**Note: Relevant work should be cited when using the dataset to avoid copyright disputes.**
- [MUUFL](https://github.com/GatorSense/MUUFLGulfport/tree/master/MUUFLGulfportSceneLabels)
- [Trento](https://github.com/danfenghong/IEEE_GRSL_EndNet/blob/master/README.md)
- [Houston](https://hyperspectral.ee.uh.edu/?page_id=459)

## Results
| Dataset | OA (%) | AA (%) | Kappa |
| :----: | :----: | :----: | :----: |
| MUUFL  | 95.26 | 96.27 | 93.79 |
| Trento  | 98.93 | 98.45 | 98.57 |
| Houston  | 94.59 | 95.27 | 94.15 |

## Citation
If you find our work helpful, please kindly cite:
```
```
