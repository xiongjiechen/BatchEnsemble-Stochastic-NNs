# Batch-Ensemble-Stochastic-Neural-Networks
This repository provides the code to reproduce the experimental results in the paper **Batch-Ensemble Stochastic Neural Networks for Out-of-Distribution Detection**.

## Prerequisites

### Python packages

To install the required python packages, run the following command:

```
pip install -r requirements.txt
```


### Datasets
Datasets used in this repository are the [Two-Moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) dataset, [CIFAR10](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.222.9220&rep=rep1&type=pdf) dataset, [FashionMNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist) dataset, [SVHN](http://ufldl.stanford.edu/housenumbers/) dataset, [NotMNIST](https://www.kaggle.com/datasets/lubaroli/notmnist) dataset, and [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.

### Experiment results

- To reproduce the experiment results on two-moons dataset, run all code cells in ```./SNN_two_moons.ipynb```.
- To reproduce the FashionMNIST vs MNIST & NotMNIST experiment results, run the python script ```train_snn_fm_be.py```.
- To reproduce the SVHN vs CIFAR-10 experiment results, run the python script ```train_snn_svhn_vs_cifar.py```.

