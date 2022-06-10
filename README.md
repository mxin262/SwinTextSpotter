
# SwinTextSpotter

<img src="demo/overall.png" width="100%">

This is the pytorch implementation of Paper: SwinTextSpotter: Scene Text Spotting via Better Synergy between Text Detection and Text Recognition (CVPR 2022). The paper is available at [this link](https://arxiv.org/pdf/2203.10209.pdf).

- We use the models pre-trained on ImageNet. The ImageNet pre-trained [SwinTransformer](https://drive.google.com/file/d/1wvzCMLJtEID8hBDu3wLpPv4xm3Es8ELC/view?usp=sharing) backbone is obtained from [SwinT_detectron2](https://github.com/xiaohu2015/SwinT_detectron2).

## Models
[SWINTS-swin-english-pretrain [config]](https://github.com/mxin262/SwinTextSpotter/blob/main/projects/SWINTS/configs/SWINTS-swin-pretrain.yaml) \| [model_Google Drive](https://drive.google.com/file/d/1q3cNhJYPIZ8Sbk0-4i_gnQIF6z09rCKh/view?usp=sharing) \| [model_BaiduYun](https://pan.baidu.com/s/1INNghiHoI_K6m2t9YxVCIw) PW: 954t

[SWINTS-swin-Total-Text [config]](https://github.com/mxin262/SwinTextSpotter/blob/main/projects/SWINTS/configs/SWINTS-swin-finetune-totaltext.yaml) \| [model_Google Drive](https://drive.google.com/file/d/1o6LbT0NayfIzTtJpozAqtz50wrSNnKIJ/view?usp=sharing) \| [model_BaiduYun](https://pan.baidu.com/s/1fLqMa9r-Ea2wIT6I81bwhA) PW: tf0i

[SWINTS-swin-ctw [config]](https://github.com/mxin262/SwinTextSpotter/blob/main/projects/SWINTS/configs/SWINTS-swin-finetune-ctw.yaml) \| [model_Google Drive](https://drive.google.com/file/d/1LC7-JFuQIIYeUt_KaDH61ICGvVkRqkz7/view?usp=sharing) \| [model_BaiduYun](https://pan.baidu.com/s/1q7zZQ1Hnl6QPmwfJXal98Q) PW: 4etq

[SWINTS-swin-icdar2015 [config]](https://github.com/mxin262/SwinTextSpotter/blob/main/projects/SWINTS/configs/SWINTS-swin-finetune-ic15.yaml) \| [model_Google Drive](https://drive.google.com/file/d/15lDht7RtN092DeGggN5qoEyTTUkmIuGs/view?usp=sharing) \| [model_BaiduYun](https://pan.baidu.com/s/1bWTwmIrZOUNqEUqx5cKXng) PW: 3n82

[SWINTS-swin-ReCTS [config]](https://github.com/mxin262/SwinTextSpotter/blob/main/projects/SWINTS/configs/SWINTS-swin-chn_finetune.yaml) \| [model_Google Drive](https://drive.google.com/file/d/1FLW35M18tw4fYSBL1qGzEOkTaD2t6mXT/view?usp=sharing) \| [model_BaiduYun](https://pan.baidu.com/s/1BHsLuwqUs_D_CO54UIaNPQ) PW: a4be

[SWINTS-swin-vintext [config]](https://github.com/mxin262/SwinTextSpotter/blob/main/projects/SWINTS/configs/SWINTS-swin-finetune-vintext.yaml) \| [model_Google Drive](https://drive.google.com/file/d/1IfyPrYFnQOWoY8pPg-GIN5ofuALU15yD/view?usp=sharing) \| [model_BaiduYun](https://pan.baidu.com/s/1c5Xc9_lCun6mazhuxBk7sA) PW: slmp

## Installation
- Python=3.8
- PyTorch=1.8.0, torchvision=0.9.0, cudatoolkit=11.1
- OpenCV for visualization

## Steps
1. Install the repository (we recommend to use [Anaconda](https://www.anaconda.com/) for installation.)
```
conda create -n SWINTS python=3.8 -y
conda activate SWINTS
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install opencv-python
pip install scipy
pip install shapely
pip install rapidfuzz
pip install timm
pip install Polygon3
git clone https://github.com/mxin262/SwinTextSpotter.git
cd SwinTextSpotter
python setup.py build develop
```

2. dataset path
```
datasets
|_ totaltext
|  |_ train_images
|  |_ test_images
|  |_ totaltext_train.json
|  |_ weak_voc_new.txt
|  |_ weak_voc_pair_list.txt
|_ mlt2017
|  |_ train_images
|  |_ annotations/icdar_2017_mlt.json
.......
```
Downloaded images
- ICDAR2017-MLT [[image]](https://rrc.cvc.uab.es/?ch=8&com=downloads)
- Syntext-150k: 
  - Part1: 94,723 [[dataset]](https://universityofadelaide.box.com/s/xyqgqx058jlxiymiorw8fsfmxzf1n03p) 
  - Part2: 54,327 [[dataset]](https://universityofadelaide.box.com/s/e0owoic8xacralf4j5slpgu50xfjoirs)
- ICDAR2015 [[image]](https://rrc.cvc.uab.es/?ch=4&com=downloads)
- ICDAR2013 [[image]](https://rrc.cvc.uab.es/?ch=2&com=downloads)
- Total-Text_train_images [[image]](https://drive.google.com/file/d/1idATPS2Uc0PAwTBcT2ndYNLse3yKtT6G/view?usp=sharing)
- Total-Text_test_images [[image]](https://drive.google.com/file/d/1zd_Z5PwiUEEoO1Y0Sb5vQQmG1uNm5v1b/view?usp=sharing)
- ReCTs [[images&label]](https://pan.baidu.com/s/1JC0_rNbsyz564YakptP6Ow) PW: 2b4q
- LSVT [[images&label]](https://pan.baidu.com/s/1j-zlH8SfmdTtH2OnuT9B7Q) PW: 9uh1
- ArT [[images&label]](https://pan.baidu.com/s/165RtrJVIsJ3QqDjesoX1jQ) PW: 2865
- SynChinese130k [[images]](https://drive.google.com/file/d/1w9BFDTfVgZvpLE003zM694E0we4OWmyP/view?usp=sharing)[[label]](https://drive.google.com/file/d/199sLThD_1e0vtDmpWrAEtUJyleS8DDTv/view?usp=sharing)
- Vintext_images [[image]](https://drive.google.com/file/d/1O8t84JtlQZE9ev4dgHrK3TLfbzRu2z9E/view?usp=sharing)

Downloaded label[[Google Drive]](https://drive.google.com/file/d/1Yx3GRRUogjYYDUrprexGnPerFX3H6r_3/view?usp=sharing) [[BaiduYun]](https://pan.baidu.com/s/14N4waNQG2D0UpAJkjc4qQQ) PW: 46vd

Downloader lexicion[[Google Drive]](https://drive.google.com/file/d/1jNX0NQKtyMC1pnh_IV__0drgNwTnupca/view?usp=sharing) and place it to corresponding dataset.

You can also prepare your custom dataset following the example scripts.
[[example scripts]](https://drive.google.com/file/d/1FE17GXyGPhDk5XI3EpbXwlOv1S8txOx2/view?usp=sharing)

## Totaltext
To evaluate on Total Text, CTW1500, ICDAR2015, first download the zipped annotations with

```
cd datasets
mkdir evaluation
cd evaluation
wget -O gt_ctw1500.zip https://cloudstor.aarnet.edu.au/plus/s/xU3yeM3GnidiSTr/download
wget -O gt_totaltext.zip https://cloudstor.aarnet.edu.au/plus/s/SFHvin8BLUM4cNd/download
wget -O gt_icdar2015.zip https://drive.google.com/file/d/1wrq_-qIyb_8dhYVlDzLZTTajQzbic82Z/view?usp=sharing
wget -O gt_vintext.zip https://drive.google.com/file/d/11lNH0uKfWJ7Wc74PGshWCOgSxgEnUPEV/view?usp=sharing
```

3. Pretrain SWINTS (e.g., with Swin-Transformer backbone)

```
python projects/SWINTS/train_net.py \
  --num-gpus 8 \
  --config-file projects/SWINTS/configs/SWINTS-swin-pretrain.yaml
```

4. Fine-tune model on the mixed real dataset

```
python projects/SWINTS/train_net.py \
  --num-gpus 8 \
  --config-file projects/SWINTS/configs/SWINTS-swin-mixtrain.yaml
```

5. Fine-tune model

```
python projects/SWINTS/train_net.py \
  --num-gpus 8 \
  --config-file projects/SWINTS/configs/SWINTS-swin-finetune-totaltext.yaml
```

6. Evaluate SWINTS (e.g., with Swin-Transformer backbone)
```
python projects/SWINTS/train_net.py \
  --config-file projects/SWINTS/configs/SWINTS-swin-finetune-totaltext.yaml \
  --eval-only MODEL.WEIGHTS ./output/model_final.pth
```

7. Visualize the detection and recognition results (e.g., with ResNet50 backbone)
```
python demo/demo.py \
  --config-file projects/SWINTS/configs/SWINTS-swin-finetune-totaltext.yaml \
  --input input1.jpg \
  --output ./output \
  --confidence-threshold 0.4 \
  --opts MODEL.WEIGHTS ./output/model_final.pth
```

## Example results:

<img src="demo/results.png" width="100%">

## Acknowlegement
[Adelaidet](https://github.com/aim-uofa/AdelaiDet), [Detectron2](https://github.com/facebookresearch/detectron2), [ISTR](https://github.com/hujiecpp/ISTR), [SwinT_detectron2](https://github.com/xiaohu2015/SwinT_detectron2), [Focal-Transformer](https://github.com/microsoft/Focal-Transformer) and [MaskTextSpotterV3](https://github.com/MhLiao/MaskTextSpotterV3).

## Citation

If our paper helps your research, please cite it in your publications:

```BibText
@article{huang2022swints,
  title = {SwinTextSpotter: Scene Text Spotting via Better Synergy between Text Detection and Text Recognition},
  author = {Mingxin Huang and YuLiang liu and Zhenghao Peng and Chongyu Liu and Dahua Lin and Shenggao Zhu and Nicholas Yuan and Kai Ding and Lianwen Jin},
  journal={arXiv preprint arXiv:2203.10209},
  year = {2022}
}
```

# Copyright

For commercial purpose usage, please contact Dr. Lianwen Jin: eelwjin@scut.edu.cn

Copyright 2019, Deep Learning and Vision Computing Lab, South China China University of Technology. http://www.dlvc-lab.net
