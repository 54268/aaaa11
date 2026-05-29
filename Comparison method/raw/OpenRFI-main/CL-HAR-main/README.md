

CL-HAR is an open-source PyTorch library of contrastive learning on wearable-sensor-based human activity recognition (HAR). For more information, please refer to our KDD-2022 paper ["What Makes Good Contrastive Learning on Small-Scale Wearable-based Tasks?"](https://arxiv.org/abs/2202.05998 ).

For more of our results, please refer to [results.md](results.md)


To install required packages, run the following code. The current Pytorch version is 1.8.

```
conda create -n CL-HAR python=3.8.3
conda activate CL-HAR
pip install -r requirements.txt
```

For the version of torch and torchvision, we found torch==1.8.0 and torchvision==0.9.0 work fine on cuda 11.6 (Tesla V100). The version of these packages should be subject to the cuda version on your device



To train contrastive models on UCIHAR dataset, run the following script.
```
'na', 'noise', 'scale', 'negate', 'perm', 'shuffle', 't_flip', 't_warp', 'resample', 'rotation', 'perm_jit', 'jit_scal', 'hfc', 'lfc', 'p_shift', 'ap_p', 'ap_f', 'same', 'repeat'
p

python main.py --framework simclr  --backbone RoInformer --dataset RF_mini_10 --aug1 noise_RF --aug2 perm_jit_RF --n_epoch 120 --batch_size 256 --lr 0.0001 --lr_cls 0.03

```



- UCIHAR [link](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
- SHAR [link](http://www.sal.disco.unimib.it/technologies/unimib-shar/)
- HHAR [link](http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition)
- RF 


- random 
- subject
- subject_large



Refer to ```models/backbones.py```
- FCN
- DeepConvLSTM
- LSTM
- AE
- CAE
- Transformer

To obtain supervised learning baselines of the encoder networks, you may use ```main_supervised_baseline.py``` 
<br>To train an encoder network under supervised setting, you can run the following code:
```angular2html
python main_supervised_baseline.py --batch_size 128 --lr 1e-4 --dataset RF_mini_10 --backbone RoInformer

```

Refer to ```models/frameworks.py```. For sub-modules (projectors, predictors) in the frameworks, refer to ```models/backbones.py```
- TS-TCC 
- SimSiam
- BYOL
- SimCLR
- NNCLR


![contrastive_models](figures/contrastive_models.png)


![backbone_networks](figures/backbone_networks.png)


![backbones](figures/backbones.png)


- NTXent ```models/loss.py```
- Cosine Similarity


Refer to ```augmentations.py```
- 
  - noise
  - scale
  - negate
  - perm
  - shuffle
  - t\_flip
  - t\_warp
  - resample
  - rotation
  - perm\_jit
  - jit\_scal

- 
  - hfc
  - lfc
  - p\_shift
  - ap\_p
  - ap\_f


- logger
- t-SNE
- MDS



If you find any of the codes helpful, kindly cite our paper.

> ```
>@misc{qian2022makes,
>      title={What Makes Good Contrastive Learning on Small-Scale Wearable-based Tasks?},
>      author={Hangwei Qian and Tian Tian and Chunyan Miao},
>      year={2022},
>      eprint={2202.05998},
>      archivePrefix={arXiv},
>      primaryClass={cs.LG}
>}
> ```



Part of the augmentation transformation functions are adapted from
- https://github.com/emadeldeen24/TS-TCC
- https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
- https://github.com/LijieFan/AdvCL/blob/main/fr_util.py

Part of the contrastive models are adapted from 
- https://github.com/lucidrains/byol-pytorch
- https://github.com/lightly-ai/lightly
- https://github.com/emadeldeen24/TS-TCC

Loggers used in the repo are adapted from 
- https://github.com/emadeldeen24/TS-TCC
- https://github.com/fastnlp/fitlog
