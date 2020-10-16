# DD2430


## Requirements
* pytest
* numpy
* pytorch

## Install requirements
Run the following command in the terminal to install requirements.
`pip install -r requirements.txt`
If you want more information about the available GPU install pycuda by:
`pip install pycuda`
*Note* pycuda requiers [CUDA](https://developer.nvidia.com/cuda-downloads) and [Visual Studio](https://visualstudio.microsoft.com/)

## Run tests 
Make sure that you are in the root folder for the project and run the command.
`python -m pytest`

## Run the network
In the main method in U_net.py you can select which tests that should be preformed. Set generate_augmented_data to true in order to generate data, note if you already have the data, set it to false.
* Run `python setup.py install --user`
* Run `python src/network/Network_driver.py`

## Tensorboard
* You need tensorflow installed `pip install tensorflow`
* Start tensorboard with `tensorboard --logdir=runs`

## Github tips
* `git checkout -b branchname` create a branch
* Push your changes to the branch and create a pull-requst via this github page
* If the tests passes, merge the and delete the branch


## Preprocessing

- [Download dataset](#download-dataset)
- Dataset Types
    - [Grayscale Channels](#grayscale-channels)
    - [RGB Channels](#rgb-channels)
    - [Grayscale BigImage](#grayscale-big-image)
    - [RGB BigImage](#rgb-big-image)
    - [Grayscale FrontPage](#grayscale-frontpage)
    - [RGB FrontPage](#rgb-frontpage)
- Inspecting Datasets
    - [Inspect RGB BigImage](#inspect-rgb-big-image)
    - [Inspect RGB Channels](#inspect-rgb-channels)

### Download dataset

```bash
OPEN_REVIEW_USERNAME=...
OPEN_REVIEW_PASSWORD=...

python3 src/Tools/open_review_dataset.py\
    --username=${OPEN_REVIEW_USERNAME?}\
    --password=${OPEN_REVIEW_PASSWORD?}\
    --base_path=data\
    --num_threads=20\
    --mode=download   
```

### Grayscale Channels

```bash
python3 src/Tools/open_review_dataset.py \
  --base_path=data\
  --image_width=256\
  --image_height=256\
  --num_pages=8\
  --mode=gs-channels\
  --num_processes=6
```

### RGB Channels

```bash
python3 src/Tools/open_review_dataset.py \
  --base_path=data\
  --image_width=256\
  --image_height=256\
  --num_pages=8\
  --mode=rgb-channels\
  --num_processes=6
```

### Grayscale Big Image

```bash
python3 src/Tools/open_review_dataset.py \
  --base_path=data\
  --image_width=256\
  --image_height=256\
  --num_pages=8\
  --mode=gs-bigimage\
  --num_processes=6
```
### RGB Big Image

```bash
python3 src/Tools/open_review_dataset.py \
  --base_path=data\
  --image_width=256\
  --image_height=256\
  --num_pages=8\
  --mode=rgb-bigimage\
  --num_processes=6
```
### Grayscale Frontpage

```bash
python3 src/Tools/open_review_dataset.py \
  --base_path=data\
  --image_width=256\
  --image_height=256\
  --mode=gs-frontpage\
  --num_processes=6
```
### RGB Frontpage

```bash
python3 src/Tools/open_review_dataset.py \
  --base_path=data\
  --image_width=256\
  --image_height=256\
  --mode=rgb-frontpage\
  --num_processes=6
```

### Inspect RGB Big Image
```bash
IMAGE_PATH=...

python3 src/Tools/open_review_dataset.py \
  --inspect=data/papers/$IMAGE_PATH\
  --mode=rgb-bigimage

xd $IMAGE_PATH.png
```

### Inspect RGB Channels
```bash
IMAGE_PATH=...

python3 src/Tools/open_review_dataset.py \
  --inspect=data/papers/$IMAGE_PATH\
  --mode=rgb-channels

xd $IMAGE_PATH.png
```
