# Hamer Teleop

## Preliminaries

- Ubuntu 20.04
- ROS Noetic
- CUDA 11.8

## Installation
First you need to clone the repo:
```
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
```

We recommend creating a conda environment for HaMeR. You can use venv:
```bash
conda create --name hamer python=3.10
conda activate hamer
```

Then, you can install the rest of the dependencies. This is for CUDA 11.8, but you can adapt accordingly:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -e .[all]
```

``` bash
cd third-party
git clone https://github.com/ViTAE-Transformer/ViTPose.git
pip install -v -e third-party/ViTPose
```

Besides these files, you also need to download the MANO model. Please visit the [MANO website](https://mano.is.tue.mpg.de) and register to get access to the downloads section.  We only require the right hand model. You need to put `MANO_RIGHT.pkl` under the `_DATA/data/mano` folder.

```bash
pip install pyrealsense2
pip install cvzone
pip install mediapipe
pip install rospkg
cd third-party
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d && pip install -e .

echo 'source /opt/ros/noetic/setup.sh' >> ~/.bashrc
```

## Run

```bash
roscore
python move_gripper.py
```

## Acknowledgements
This code is modifed by [HaMer](https://github.com/geopavlakos/hamer). The code is maintained by [Jinzhou Li](https://github.com/kingchou007), [Tianhao Wu](https://github.com/tianhaowuhz), [Wei Wei](https://github.com/v-wewei), and [Hongwei Fan](https://github.com/hwfan).
