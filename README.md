# Dilated feature distillation attention network for efficient image super-resolution [paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320325005163)

## Environment in our experiments
[python 3.8]

[Ubuntu 20.04]

[BasicSR 1.4.2](https://github.com/XPixelGroup/BasicSR)

[PyTorch 1.13.0, Torchvision 0.14.0, Cuda 11.7](https://pytorch.org/get-started/previous-versions/)

### Installation
```
git clone https://github.com/Supereeeee/DFDAN.git
pip install -r requirements.txt
python setup.py develop
```

## How To Test
· Refer to ./options/test for the configuration file of the model to be tested and prepare the testing data.  

· The pre-trained models have been palced in ./experiments/pretrained_models/  

· Then run the follwing codes for testing:  

```
python basicsr/test.py -opt options/test/test_DFDAN_x2.yml
python basicsr/test.py -opt options/test/test_DFDAN_x3.yml
python basicsr/test.py -opt options/test/test_DFDAN_x4.yml
```
The testing results will be saved in the ./results folder.

## How To Train
· Refer to ./options/train for the configuration file of the model to train.  

· Preparation of training data can refer to this page. All datasets can be downloaded at the official website.  

· Note that the default training dataset is based on lmdb, refer to [docs in BasicSR](https://github.com/XPixelGroup/BasicSR/blob/master/docs/DatasetPreparation.md) to learn how to generate the training datasets.  

· The training command is like following:
```
python basicsr/train.py -opt options/train/train_DFDAN_x2.yml
python basicsr/train.py -opt options/train/train_DFDAN_x3.yml
python basicsr/train.py -opt options/train/train_DFDAN_x4.yml
```
For more training commands and details, please check the docs in [BasicSR](https://github.com/XPixelGroup/BasicSR)  

## Model Complexity
· The network structure of DFDAN is palced at ./basicsr/archs/DFDAN_arch.py

· We adopt thop tool to calculate model complexity, see ./basicsr/archs/model_complexity.py

## Inference time
· We test the inference time on multiple benchmark datasets on a 140W fully powered 3060 laptop. 

· You can run ./inference/inference_DFDAN.py on your decive.


## Acknowledgement
This code is based on [BasicSR](https://github.com/XPixelGroup/BasicSR) toolbox. Thanks for the awesome work.

## Contact
If you have any question, please email quanwei1277@163.com.
