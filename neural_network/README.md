# FaceBaker Deep Neural Network

This module contains everything for training and running FaceBaker neural network, including PCA computation.

## Setup

The easiest way to set everything up is to install [Conda](https://docs.conda.io/en/latest/).

You can use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) (which is a compact and handy Conda distributive) or [Anaconda](https://docs.anaconda.com/anaconda/install/) (which is heavy and contains a ton of packages that you won't need for this project).

After Conda is installed, open the Conda Prompt, navigate to the FaceBaker/neural_network directory and run:

```
conda create --name FaceBaker -y python=3.7
conda activate FaceBaker
conda install -y --file requirements.txt
```