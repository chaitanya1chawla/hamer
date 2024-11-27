# HaMeR: Hand Mesh Recovery

Code repository for the paper:
**Reconstructing Hands in 3D with Transformers**

[Georgios Pavlakos](https://geopavlakos.github.io/), [Dandan Shan](https://ddshan.github.io/), [Ilija Radosavovic](https://people.eecs.berkeley.edu/~ilija/), [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/), [David Fouhey](https://cs.nyu.edu/~fouhey/), [Jitendra Malik](http://people.eecs.berkeley.edu/~malik/)

[![arXiv](https://img.shields.io/badge/arXiv-2312.05251-00ff00.svg)](https://arxiv.org/pdf/2312.05251.pdf)  [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://geopavlakos.github.io/hamer/)     [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rQbQzegFWGVOm1n1d-S6koOWDo7F2ucu?usp=sharing)  [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/geopavlakos/HaMeR)

![teaser](assets/teaser.jpg)

## News

- [2024/06] HaMeR received the 2nd place award in the Ego-Pose Hands task of the Ego-Exo4D Challenge! Please check the [validation report](https://www.cs.utexas.edu/~pavlakos/hamer/resources/egoexo4d_challenge.pdf).
- [2024/05] We have released the evaluation pipeline!
- [2024/05] We have released the HInt dataset annotations! Please check [here](https://github.com/ddshan/hint).
- [2023/12] Original release!

## Pull docker image
I have pushed a pre-compiled docker image if you don't want to go through the hassle of making your own container.
```bash
docker pull chaitanya1chawla/hamer_container:hamer_image
docker run -it --gpus all chaitanya1chawla/hamer_container:hamer_image
```

## Installation with Docker [recommended]

Setup docker container with cuda image:
```bash
# Pull image and run container:
docker pull nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu22.04  # devel image because base image doesn't support cuda/nvcc etc.
docker run -it --gpus all nvcr.io/nvidia/cuda:11.7.0-devel-ubuntu22.04

# Inside container -

# Setup environment:
apt-get update && apt-get upgrade
apt install python3
apt install python3-pip
apt-get install ffmpeg libsm6 libxext6
python3 -m pip install numpy matplotlib scikit-learn scikit-image opencv-python opencv-contrib-python
python3 -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
apt-get install libglfw3-dev libgles2-mesa-dev


# Clone detectron2:
apt install git
python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

 pip install "numpy<2"
```

First you need to clone the repo:

```
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
pip install -e .[all]

cd third-party
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e third-party/ViTPose
```

You also need to download the trained models:

```bash
bash fetch_demo_data.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section.  We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.


## Local Installation

First you need to clone the repo:

```
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
```

We recommend creating a virtual environment for HaMeR. You can use venv:

```bash
python3.10 -m venv .hamer
source .hamer/bin/activate
```

or alternatively conda:

```bash
conda create --name hamer python=3.10
conda activate hamer
```

Then, you can install the rest of the dependencies. This is for CUDA 11.7, but you can adapt accordingly:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all] # install detectron2 separately
pip install -v -e third-party/ViTPose
```

Install Detectron2:

```bash
# Create conda env
conda create --name detectron2 python==3.9 -y
conda activate detectron2

# Install torch
pip install torch torchvision

# Install gcc and g++ with conda
conda install -c conda-forge pybind11
conda install -c conda-forge gxx
conda install -c anaconda gcc_linux-64
conda upgrade -c conda-forge --all

# I had to add a version to the gcc install, and used conda-forge:
conda install -c conda-forge  gcc_linux-64=13.2.0

# Install detectron2 (specific version)
pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.6'
```

You also need to download the trained models:

```bash
bash fetch_demo_data.sh
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section.  We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.

### Docker Compose

If you wish to use HaMeR with Docker, you can use the following command:

```
docker compose -f ./docker/docker-compose.yml up -d
```

After the image is built successfully, enter the container and run the steps as above:

```
docker compose -f ./docker/docker-compose.yml exec hamer-dev /bin/bash
```

Continue with the installation steps:

```bash
bash fetch_demo_data.sh
```

## Demo

```bash
python demo.py \
    --img_folder example_data --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame
```

## HInt Dataset

We have released the annotations for the HInt dataset. Please follow the instructions [here](https://github.com/ddshan/hint)

## Training

First, download the training data to `./hamer_training_data/` by running:

```
bash fetch_training_data.sh
```

Then you can start training using the following command:

```
python train.py exp_name=hamer data=mix_all experiment=hamer_vit_transformer trainer=gpu launcher=local
```

Checkpoints and logs will be saved to `./logs/`.

## Evaluation

Download the [evaluation metadata](https://www.dropbox.com/scl/fi/7ip2vnnu355e2kqbyn1bc/hamer_evaluation_data.tar.gz?rlkey=nb4x10uc8mj2qlfq934t5mdlh) to `./hamer_evaluation_data/`. Additionally, download the FreiHAND, HO-3D, and HInt dataset images and update the corresponding paths in  `hamer/configs/datasets_eval.yaml`.

Run evaluation on multiple datasets as follows, results are stored in `results/eval_regression.csv`.

```bash
python eval.py --dataset 'FREIHAND-VAL,HO3D-VAL,NEWDAYS-TEST-ALL,NEWDAYS-TEST-VIS,NEWDAYS-TEST-OCC,EPICK-TEST-ALL,EPICK-TEST-VIS,EPICK-TEST-OCC,EGO4D-TEST-ALL,EGO4D-TEST-VIS,EGO4D-TEST-OCC'
```

Results for HInt are stored in `results/eval_regression.csv`. For [FreiHAND](https://github.com/lmb-freiburg/freihand) and [HO-3D](https://codalab.lisn.upsaclay.fr/competitions/4318) you get as output a `.json` file that can be used for evaluation using their corresponding evaluation processes.

## Acknowledgements

Parts of the code are taken or adapted from the following repos:

- [4DHumans](https://github.com/shubham-goel/4D-Humans)
- [SLAHMR](https://github.com/vye16/slahmr)
- [ProHMR](https://github.com/nkolot/ProHMR)
- [SPIN](https://github.com/nkolot/SPIN)
- [SMPLify-X](https://github.com/vchoutas/smplify-x)
- [HMR](https://github.com/akanazawa/hmr)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [Detectron2](https://github.com/facebookresearch/detectron2)

Additionally, we thank [StabilityAI](https://stability.ai/) for a generous compute grant that enabled this work.

## Citing

If you find this code useful for your research, please consider citing the following paper:

```bibtex
@inproceedings{pavlakos2024reconstructing,
    title={Reconstructing Hands in 3{D} with Transformers},
    author={Pavlakos, Georgios and Shan, Dandan and Radosavovic, Ilija and Kanazawa, Angjoo and Fouhey, David and Malik, Jitendra},
    booktitle={CVPR},
    year={2024}
}
```
