# natural-wm
Code for the project: Natural constraints explain working memory capacity limitations in sensory-cognitive models

The paper is now on bioRxiv:
Xie, Y., Duan, Y., Cheng, A., Jiang, P., Cueva, C.J. and Yang, G.R., 2023. Natural constraints explain working memory capacity limitations in sensory-cognitive models. bioRxiv, pp.2023-03.

### general pipeline
In general our pipeline consist of three part. 

1. training a model, this step initialize a network (or a part of network models) and train them with specific training objectives. Such as pre-training of a CNN to perform image classification, pre-training of a CNN with unsupervised learning objectives, or training of a RNN to perform working memory tasks. This step will save a list of models at different stages during the training.

This step can be run in parallel on the cluster.
```bash
# train models, define experiments, write configurations in root/configs/experiments.py
main.py -t exp_name
```

1. evaluating a model, this step load a existing model then evaluate and save its performance or neural activities for later analyses. For example, this is used when we load a existing model and evalutate it with working memory tasks but with different dataset.

This step can be run in parallel on the cluster.
```bash
# evaluate models, write evaluation code in root/configs/exp_eval.py
main.py -e exp_name
```

3. analyzing a model, this step is use to analyze model performance and make figures. (for some older experiments, the evaluation step is merged with this step.)

This step can not be run in parallel on the cluster right now.
```bash
# analyze models, write analysis and plotting code in root/configs/exp_analysis.py
main.py -a exp_name
```

to do the above on the cluster, ustilizing paralell computation, just add `-c` to each of the above command in the end

On Openmind, start an interactive session and run the following command.
```bash
main.py -t exp_name -c
main.py -e exp_name -c
main.py -a exp_name -c
```

### some example use cases

```bash
# for example, train a ResNet on CIFAR dataset with image classification objective
# this provide pretrained CNN for many other experiments
main.py -t classification_pretrain

# train, evaluate and analyze model with Delayed-Match-to-Sample task with CIFAR-10 images with different delay steps.
main.py -t dms_gen_novel_images_CIFAR10_var_delay
main.py -e dms_gen_novel_images_CIFAR10_var_delay
main.py -a dms_gen_novel_images_CIFAR10_var_delay

# train and analyze delayed-match-to-sampel with luck vogel dataset
main.py -t luck_vogel_dms
main.py -a luck_vogel_dms

# train a LSTM model to do delayed-match-to-sample
# task with one-hot vectors as input
main.py -t dms_gen_novel_low_dim
# perform analysis on this experiments
main.py -a dms_gen_novel_low_dim

# first pre-train a CNN for visual area
main.py -t mem_pres_pretrain1
# then connect this pre-trained ResNet with LSTM to perform Delayed-Match-to-Sample task
main.py -t dms_gen_novel_images_MNIST

# to do the above on the cluster
main.py -t exp_name -c
main.py -a exp_name -c

# use -m to check if all jobs of a particular training 
# experiment are complete, if not complete submit jobs that were 
# not complete again
main.py -t exp_name -c -m
```

# environment
```bash
conda create --name wm python=3.10
conda activate wm

# install pytorch
# linux
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
# mac
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

# install other packages
conda install tqdm seaborn imageio sympy scikit-learn scipy pandas scikit-image
conda install -c conda-forge adjusttext
pip install wandb

# log in at wandb
wandb login

# install the latest version ofneurogym
# following instructions on https://neurogym.github.io/ 
# or simply run the following command (install neurogym one folder up) 
cd ..
git clone https://github.com/YudiXie/neurogym.git
cd neurogym
pip install -e .

# install visual-prior, a package for loading taakonomy models
cd ..
git clone https://github.com/alexsax/visual-prior
cd visual-prior
pip install -e .
```

# Notes about data

1. Keep all the data files within flexiblewm/data folder.
2. do dataset preparation in prepare_dataset.py, so that we run this file once when a new dataset is added. But we don’t want to run the data preparation every time we run some other experiments.
2. do model preparation in prepare_model.py
