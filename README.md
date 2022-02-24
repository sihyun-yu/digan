## DIGAN (ICLR 2022)

Official PyTorch implementation of **["Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks"](https://openreview.net/forum?id=Czsdv-S4-w9)** by 
[Sihyun Yu*](https://sihyun-yu.github.io/)<sup>,1</sup>, 
[Jihoon Tack*](https://jihoontack.github.io/)<sup>,1</sup>, 
[Sangwoo Mo*](https://sites.google.com/view/sangwoomo/)<sup>,1</sup>, 
[Hyunsu Kim](https://www.linkedin.com/in/blandocs/)<sup>2</sup>, 
[Junho Kim](https://github.com/taki0112)<sup>2</sup>, 
[Jung-Woo Ha](https://aidljwha.wordpress.com/)<sup>2</sup>, 
[Jinwoo Shin](https://alinlab.kaist.ac.kr/shin.html)<sup>1</sup>.  
<sup>1</sup>KAIST, <sup>2</sup>NAVER AI Lab (KAIST-NAVER Hypercreative AI Center)

**TL;DR**: We make video generation scalable leveraging implicit neural representations.  
[paper](https://openreview.net/forum?id=Czsdv-S4-w9) | [project page](https://sihyun-yu.github.io/digan/)

<p align="center">
    <img src=figures/method_overview.png width="900"> 
</p>

Illustration of the (a) generator and (b) discriminator of DIGAN. The generator creates a video INR weight from random content and motion vectors, which produces an image that corresponds to the input 2D grids {(x, y)} and time t. Two discriminators determine the reality of each image and motion (from a pair of images and their time difference), respectively.

### 1. Environment setup
```
conda create -n digan python=3.8
conda activate digan

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install hydra-core==1.0.6
pip install tqdm scipy scikit-learn av ninja
pip install click gitpython requests psutil einops tensorboardX
```

### 2. Dataset 
One should organize the video dataset as follows:

#### UCF-101
```
UCF-101
|-- train
    |-- class1
        |-- video1.avi
        |-- video2.avi
        |-- ...
    |-- class2
        |-- video1.avi
        |-- video2.avi
        |-- ...
    |-- ...
```

#### Other video datasets (Sky Time lapse, TaiChi-HD, Kinetics-food)
```
Video dataset
|-- train
    |-- video1
        |-- frame00000.png
        |-- frame00001.png
        |-- ...
    |-- video2
        |-- frame00000.png
        |-- frame00001.png
        |-- ...
    |-- ...
|-- val
    |-- video1
        |-- frame00000.png
        |-- frame00001.png
        |-- ...
    |-- ...
```
#### Dataset download
- Link: [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php), [Sky Time lapse](https://github.com/weixiong-ur/mdgan), [TaiChi-HD](https://github.com/AliaksandrSiarohin/first-order-model)
- For Kinetics-food dataset, read [prepare_data/README.md](./prepare_data/README.md)

### 3. Training
To train the model, navigate to the project directory and run:
```
python src/infra/launch.py hydra.run.dir=. +experiment_name=<EXP_NAME> +dataset.name=<DATASET>
```
You may change training options via modifying `configs/main.yml` and `configs/digan.yml`.\
Also the dataset list is as follows, `<DATASET>`: {`UCF-101`,`sky`,`taichi`,`kinetics`}

### 4. Evaluation (FVD and KVD)
```
python src/scripts/compute_fvd_kvd.py --network_pkl <MODEL_PATH> --data_path <DATA_PATH>
```

### 5. Video generation
Genrate and visualize videos (as gif and mp4):
```
python src/scripts/generate_videos.py --network_pkl <MODEL_PATH> --outdir <OUTPUT_PATH>
```

### 6. Results
Generated video results of DIGAN on TaiChi (top) and Sky (bottom) datasets.\
More generated video results are available at the following [site](https://sihyun-yu.github.io/digan/).\
One can download the pretrained checkpoints from the following [link](https://drive.google.com/drive/folders/1zrzyBMrqy7V_o4gGGLo_m2aErfshaFjz).

<p align="center">
    <img src=figures/taichi.gif width="500" height="500" />
</p>

<p align="center">
    <img src=figures/sky.gif width="500" height="500" />
</p>

### Citation
```
@inproceedings{
    yu2022digan,
    title={Generating Videos with Dynamics-aware Implicit Generative Adversarial Networks},
    author={Yu, Sihyun and Tack, Jihoon and Mo, Sangwoo and Kim, Hyunsu and Kim, Junho and Ha, Jung-Woo and Shin, Jinwoo},
    booktitle={International Conference on Learning Representations},
    year={2022},
}
```

### Reference
This code is mainly built upon [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada-pytorch) and [INR-GAN](https://github.com/universome/inr-gan) repositories.\
We also used the code from following repositories: [DiffAug](https://github.com/mit-han-lab/data-efficient-gans), [VideoGPT](https://github.com/wilson1yan/VideoGPT), [MDGAN](https://github.com/weixiong-ur/mdgan)

### Lisence
```
Copyright 2022-present NAVER Corp.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
