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

### GSChannels

```bash
python3 src/Tools/open_review_dataset.py \
  --base_path=data\
  --image_width=256\
  --image_height=256\
  --num_pages=8\
  --mode=gschannels
```

### RGBChannels

```bash
python3 src/Tools/open_review_dataset.py \
  --base_path=data\
  --image_width=256\
  --image_height=256\
  --num_pages=8\
  --mode=rgbchannels
```
