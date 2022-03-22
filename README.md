
# SwinTextSpotter

<img src="demo/overall.png" width="100%">

This is the pytorch implementation of Paper: SwinTextSpotter: Scene Text Spotting via Better Synergy between Text Detection and Text Recognition (CVPR 2022). The paper is available at [this link](https://arxiv.org/pdf/2203.10209.pdf).

- We use the models pre-trained on ImageNet. The ImageNet pre-trained [SwinTransformer](https://drive.google.com/drive/u/1/folders/19UaSgR4OwqA-BhCs_wG7i6E-OXC5NR__) backbone is obtained from [SwinT_detectron2](https://github.com/xiaohu2015/SwinT_detectron2).

#### Installation
- Python=3.8
- PyTorch=1.8.0, torchvision=0.9.0, cudatoolkit=11.1
- OpenCV for visualization

#### Steps
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
|  |_ annotations/train.json
|_ mlt2017
|  |_ train_images
|  |_ annotations/train.json
.......
```
Downloaded images
- ICDAR2017-MLT [[image]](https://rrc.cvc.uab.es/?ch=8&com=downloads)
- Syntext-150k: 
  - Part1: 94,723 [[dataset]](https://universityofadelaide.box.com/s/xyqgqx058jlxiymiorw8fsfmxzf1n03p) 
  - Part2: 54,327 [[dataset]](https://universityofadelaide.box.com/s/e0owoic8xacralf4j5slpgu50xfjoirs)
- ICDAR2015 [[image]](https://rrc.cvc.uab.es/?ch=4&com=downloads)
- ICDAR2013 [[image]](https://rrc.cvc.uab.es/?ch=2&com=downloads)
- Total-Text [[image]](https://drive.google.com/file/d/1bC68CzsSVTusZVvOkk7imSZSbgD1MqK2/view?usp=sharing)
- ReCTs [[images&label]](https://rrc.cvc.uab.es/?ch=12&com=downloads)
- LSVT [[images&label]](https://drive.google.com/file/d/1E9RMFiRaRW4WdzA9Py7OimfzA82-Bwik/view?usp=sharing)(8.2G)
- ArT [[images&label]](https://drive.google.com/file/d/1ss_3oYVYexSmhx7AP4cahl8Emd49Wrh8/view?usp=sharing)
- SynChinese130k [[images&label]](https://drive.google.com/file/d/1w9BFDTfVgZvpLE003zM694E0we4OWmyP/view?usp=sharing)

Downloaded label[[Google Drive]](https://drive.google.com/file/d/1wd_Z8UPNXRtnzU_qZCukKhxa_CDO5eaO/view?usp=sharing) [[BaiduYun]](https://pan.baidu.com/s/18GM7kwT-cuW01vDl4zutoQ) PW: a8gj

3. Pretrain SWINTS (e.g., with Swin-Transformer backbone)

```
python projects/SWINTS/train_net.py --num-gpus 8 --config-file projects/SWINTS/configs/SWINTS-swin-pretrain.yaml
```

4. Fine-tune model on the mixed real dataset

```
python projects/SWINTS/train_net.py --num-gpus 8 --config-file projects/SWINTS/configs/SWINTS-swin-mixtrain.yaml
```

5. Fine-tune model

```
python projects/SWINTS/train_net.py --num-gpus 8 --config-file projects/SWINTS/configs/SWINTS-swin-finetune-totaltext.yaml
```

6. Evaluate SWINTS (e.g., with Swin-Transformer backbone)
```
python projects/SWINTS/train_net.py --config-file projects/SWINTS/configs/SWINTS-swin-finetune-totaltext.yaml --eval-only MODEL.WEIGHTS ./output/model_final.pth
```

7. Visualize the detection and recognition results (e.g., with ResNet50 backbone)
```
python demo/demo.py --config-file projects/SWINTS/configs/SWINTS-swin-finetune-totaltext.yaml --input input1.jpg --output ./output --confidence-threshold 0.4 --opts MODEL.WEIGHTS ./output/model_final.pth
```

### Example results:

<img src="demo/results.png" width="100%">

## Acknowlegement
Part of the codes are built on top of [Detectron2](https://github.com/facebookresearch/detectron2), [ISTR](https://github.com/hujiecpp/ISTR), [SwinT_detectron2](https://github.com/xiaohu2015/SwinT_detectron2), [Focal-Transformer](https://github.com/microsoft/Focal-Transformer) and [MaskTextSpotterV3](https://github.com/MhLiao/MaskTextSpotterV3).

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
