# Open-World Fine-Grained Fashion Retrieval with LLM-based Commonsense Knowledge Infusion
This is a repository contains the implementation of our submission 590.

## Table of Contents

* [Environments](#environments)
* [Datasets](#datasets)
* [Configuration](#configuration)
* [Training](#training)
* [Evaluation](#evaluation)

## Environments
- **Ubuntu** 22.04
- **CUDA** 12.2
- **Python** 3.9

Install other required packages by
```sh
pip install -r requirements.txt
```

## Datasets
We conduct experiments on three fashion related datasets, i.e., FashionAI, DARN, and DeepFashion. Please download and put them in the corresponding folders.

### Configuration

The behavior of our codes is controlled by configuration files under the `config` directory. 

```sh
config
│── FashionAI
│   ├── FashionAI.yaml
│   ├── train.yaml
├── DARN
│   ├── DARN.yaml
│   ├── train.yaml
└── DeepFashion
    ├── DeepFashion.yaml
    ├── train.yaml
```

Each dataset is configured by two types of configuration files. One is `<Dataset>.yaml` that specifies basic dataset information such as path to the training data and annotation files. The other two set some training options as needed.

If the above `data` directory is placed at the same level with `main.py`, no changes are needed to the configuration files. Otherwise, be sure to correctly configure relevant path to the data according to your working environment.

## Training

Run the following script that uses default settings:

```python
python main.py --cfg config/<Dataset>/<Dataset>.yaml config/<Dataset>/train.yaml
```

## Evaluation

Run the following script to test on the trained models:

```python
python main.py --cfg config/<Dataset>/<Dataset>.yaml config/<Dataset>/train.yaml --resume runs/<Dataset>_s2/model_best.pth.tar --test TEST
```
