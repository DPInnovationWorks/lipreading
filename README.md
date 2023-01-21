# HUST Lipreading

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/zero0kiriyu/lipreading.git
cd lipreading

# [OPTIONAL] create conda environment
conda create -n lipreading python=3.9
conda activate lipreading

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt

conda install -c conda-forge pyturbojpeg
```

Train model with default configuration

```bash
# train on CPU
python lipreading/train.py trainer=cpu

# train on GPU
python lipreading/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python lipreading/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python lipreading/train.py trainer.max_epochs=20 datamodule.batch_size=64
```

# grid训练

1. 正常版本训练
```
python lipreading/train.py experiment=grid_lipnet_ctc
```

2. 下毒版本训练
```
python lipreading/train.py experiment=grid_lipnet_ctc_poison
```
