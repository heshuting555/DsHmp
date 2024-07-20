# Decoupling Static and Hierarchical Motion Perception for Referring Video Segmentation
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-%23EE4C2C.svg?style=&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.7%20|%203.8%20|%203.9-blue.svg?style=&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/decoupling-static-and-hierarchical-motion/referring-video-object-segmentation-on-mevis)](https://paperswithcode.com/sota/referring-video-object-segmentation-on-mevis?p=decoupling-static-and-hierarchical-motion)

**[📄[arXiv]](https://arxiv.org/abs/2404.03645v1)**  &emsp; **[📄[PDF]](https://openaccess.thecvf.com/content/CVPR2024/papers/He_Decoupling_Static_and_Hierarchical_Motion_Perception_for_Referring_Video_Segmentation_CVPR_2024_paper.pdf)** 

This repository contains code for **CVPR2024** paper:
> [Decoupling Static and Hierarchical Motion Perception for Referring Video Segmentation](https://arxiv.org/abs/2404.03645v1)  
> Shuting He,  Henghui Ding  
> CVPR 2024

## Installation:

Please see [INSTALL.md](https://github.com/henghuiding/MeViS/blob/main/INSTALL.md). Then

```
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
```

## Inference

###  1. Val<sup>u</sup> set
Obtain the output masks of Val<sup>u</sup> set:
```
python train_net_dshmp.py \
    --config-file configs/dshmp_swin_tiny.yaml \
    --num-gpus 8 --dist-url auto --eval-only \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [output_dir]
```
Obtain the J&F results on Val<sup>u</sup> set:
```
python tools/eval_mevis.py
```
###  2. Val set
Obtain the output masks of Val set for [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/15094) online evaluation:
```
python train_net_dshmp.py \
    --config-file configs/dshmp_swin_tiny.yaml \
    --num-gpus 8 --dist-url auto --eval-only \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [output_dir] DATASETS.TEST '("mevis_test",)'
```
## Training

Firstly, download the backbone weights (`model_final_86143f.pkl`) and convert it using the script:

```
wget https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_tiny_bs16_50ep/model_final_86143f.pkl
python tools/process_ckpt.py
python tools/get_refer_id.py
```

Then start training:
```
python train_net_dshmp.py \
    --config-file configs/dshmp_swin_tiny.yaml \
    --num-gpus 8 --dist-url auto \
    MODEL.WEIGHTS [path_to_weights] \
    OUTPUT_DIR [path_to_weights]
```

Note: We train on a 3090 machine using 8 cards with 1 sample on each card, taking about 17 hours.

## Models

☁️ [Google Drive](https://drive.google.com/file/d/1YLnRUsANuPVfLo1jrgK05EGUJglrwA9H/view?usp=drive_link)

## Acknowledgement

This project is based on [MeViS](https://github.com/henghuiding/MeViS). Many thanks to the authors for their great works!

## BibTeX
Please consider to cite DsHmp if it helps your research.

```bibtex
@inproceedings{DsHmp,
  title={Decoupling static and hierarchical motion perception for referring video segmentation},
  author={He, Shuting and Ding, Henghui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13332--13341},
  year={2024}
}
```




