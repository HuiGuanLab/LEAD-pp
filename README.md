# LEAD++: Unsupervised Fine-grained Visual Recognition with Multi-context Enhanced Entropy-based Adaptive Distillation

<img src="framework.png"> 
This work is an extension of our previous method, LEAD, which focuses on entropy-based adaptive distillation for fine-grained visual representation learning.
The official implementation of LEAD is available at: https://github.com/HuiGuanLab/LEAD.

## Datasets
| Dataset | Download Link |
| -- | -- |
| CUB-200-2011 | https://www.vision.caltech.edu/datasets/cub_200_2011/ |
| Stanford Cars | https://huggingface.co/datasets/tanganke/stanford_cars/ |
| FGVC Aircraft | https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ |
| Stanford Dogs | http://vision.stanford.edu/aditya86/ImageNetDogs/ |

All datasets are expected to be processed and organized in a unified **ImageFolder format**.Please download the datasets and arrange them following this structure.
For the **CUB-200-2011** and **FGVC Aircraft** dataset, you can use the following command to convert it into the desired ImageFolder format:

```
python aircraft_organize.py --ds /path/to/fgvc-aircraft-2013b --out /path/to/aircraft --link none
python bird_organize.py -----
```

All datasets are expected to be processed into the following structure:
```
LEAD++
├── bird/
│   ├── images/ 
		├── 001.Black_footed_Albatross
		├── 002.Laysan_Albatross
		……
	├── images.txt
	├── train_test_split.txt
├── car/
│   ├── train/ 
		├── Acura Integra Type R 2001
		├── Acura RL Sedan 2012
		……
    ├── test/
├── aircraft/
│   ├── train/ 
		├── 707-320
		├── 727-200
		……
    ├── test/
……
```

## Environments
- **Ubuntu** 22.04
- **CUDA** 12.4

Use the following instructions to create the corresponding conda environment. Besides, you should download the ResNet50 pre-trained model by clicking [here](https://download.pytorch.org/models/resnet50-19c8e357.pth) and save it in this folder.

```sh
conda create --name LEAD python=3.9.1
conda activate LEAD
pip install -r requirements.txt
```

## Mutual-information–based Localization Preprocessing
Before training, we need to generate cropped images by following the steps below.
Please make sure you are **in the LEAD++ root** directory before running the commands.
```
cd DDT
chmod +x ./run_ddt.sh
```
```
./run_ddt.sh $task $dataset $pretrained $cuda_device
```
`$task` is the task name (bird, car, aircraft and others).

`$dataset` is the dataset path for unsupervised pre-training.

`$pretrained` indicates whether to use pretrained weights for the model.In our implementation, we use pretrained ResNet-50 weights.

`$cuda_device` is the ID of used GPU.








