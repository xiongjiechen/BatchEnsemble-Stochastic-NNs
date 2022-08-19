import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from matplotlib import pyplot as plt
import time

from utils.datasets import (
    get_CIFAR10,
    get_SVHN,
    get_FashionMNIST,
    get_MNIST,
    get_notMNIST,
)


def prepare_ood_datasets(true_dataset, ood_dataset):
    # Preprocess OoD dataset same as true dataset
    ood_dataset.transform = true_dataset.transform

    datasets = [true_dataset, ood_dataset]

    anomaly_targets = torch.cat(
        (torch.zeros(len(true_dataset)), torch.ones(len(ood_dataset)))
    )

    concat_datasets = torch.utils.data.ConcatDataset(datasets)

    dataloader = torch.utils.data.DataLoader(
        concat_datasets, batch_size=500, shuffle=False, num_workers=4, pin_memory=False
    )

    return dataloader, anomaly_targets


def loop_over_dataloader(model, dataloader):
    model.eval()

    with torch.no_grad():
        scores = []
        accuracies = []
        for i, (data, target) in enumerate(dataloader):
            tic = time.time()
            data = data.cuda()
            target = target.cuda()

            output = model(data)
            toc = time.time()
            # print(toc-tic)
            model.logit = model.logit * model.logit.max(-1,keepdims=True)[0].exp()
            normalized_output = F.softmax(model.logit, dim=1)
            entropy = (-normalized_output * (torch.log(normalized_output + 1e-20))).sum(-1, keepdim=True)#-model.logit.exp().sum(-1,keepdims=True)#output.max(1)#

            kernel_distance, pred = output.max(1)

            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())


            scores.append(entropy.cpu().numpy())
            # scores.append(-kernel_distance.cpu().numpy())

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies

def loop_over_dataloader_snn(model, dataloader):
    model.eval()

    with torch.no_grad():
        scores = []
        scores_entropy = []
        accuracies = []
        for data, target in dataloader:
            data = data.cuda()
            target = target.cuda()

            output = model(data)

            model.logit = model.logit * model.logit.max(-1,keepdims=True)[0].exp()
            normalized_output = F.softmax(model.logit, dim=1)
            entropy = (-normalized_output * (torch.log(normalized_output + 1e-20))).sum(-1, keepdim=True)#-model.logit.exp().sum(-1,keepdims=True)#
            kernel_distance, pred = output.max(1)

            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())

            scores_entropy.append(entropy.cpu().numpy())
            scores.append(-kernel_distance.cpu().numpy())

    scores = np.concatenate(scores_entropy)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies

def loop_over_dataloader_snn_be(model, dataloader, num_model = 4):
    model.eval()

    with torch.no_grad():
        scores = []
        scores_entropy = []
        accuracies = []
        for i, (data, target) in enumerate(dataloader):
            tic = time.time()

            data = data.cuda()
            target = target.cuda()

            data_repeat = torch.cat([data for i in range(num_model)], dim=0)

            output_repeat = model(data_repeat)
            toc = time.time()
            #print(toc - tic)

            model.logit = model.logit * model.logit.max(-1,keepdims=True)[0].exp()
            normalized_output = F.softmax(model.logit, dim=-1)
            normalized_output = normalized_output.reshape([num_model, data.shape[0], -1])
            normalized_output = normalized_output.mean(0)
            entropy = (-normalized_output * (torch.log(normalized_output + 1e-20))).sum(-1, keepdim=True)#-model.logit.exp().sum(-1,keepdims=True)#

            output = output_repeat.reshape([num_model, data.shape[0], -1]).mean(0)
            kernel_distance, pred = output.max(1)

            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())


            scores_entropy.append(entropy.cpu().numpy())
            scores.append(-kernel_distance.cpu().numpy())

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies

def loop_over_dataloader_snn_be_no_temper(model, dataloader, num_model = 4):
    model.eval()

    with torch.no_grad():
        scores = []
        scores_entropy = []
        accuracies = []
        for i, (data, target) in enumerate(dataloader):
            tic = time.time()

            data = data.cuda()
            target = target.cuda()

            data_repeat = torch.cat([data for i in range(num_model)], dim=0)

            output_repeat = model(data_repeat)
            toc = time.time()
            #print(toc - tic)

            # model.logit = model.logit * model.logit.max(-1,keepdims=True)[0].exp()
            normalized_output = F.softmax(model.logit, dim=-1)
            normalized_output = normalized_output.reshape([num_model, data.shape[0], -1])
            normalized_output = normalized_output.mean(0)
            entropy = (-normalized_output * (torch.log(normalized_output + 1e-20))).sum(-1, keepdim=True)#-model.logit.exp().sum(-1,keepdims=True)#

            output = output_repeat.reshape([num_model, data.shape[0], -1]).mean(0)
            kernel_distance, pred = output.max(1)

            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())


            scores_entropy.append(entropy.cpu().numpy())
            scores.append(-kernel_distance.cpu().numpy())

    scores = np.concatenate(scores)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies

def loop_over_dataloader_conformal(model, dataloader, held_out_dataloader, qhat, num_model = 4):
    model.eval()

    with torch.no_grad():
        scores = []
        scores_entropy = []
        accuracies = []
        for i, (data, target) in enumerate(dataloader):

            data = data.cuda()
            target = target.cuda()

            output = model(data)

            # entropy = (-output * (torch.log(output + 1e-20))).sum(-1, keepdim=True)#-model.logit.exp().sum(-1,keepdims=True)#

            kernel_distance, pred = output.max(1)

            accuracy = pred.eq(target)
            accuracies.append(accuracy.cpu().numpy())

            scores_entropy.append((-output * (torch.log(output + 1e-20))).sum(-1, keepdim=True).cpu().numpy())
            scores.append(-kernel_distance.cpu().numpy())

    scores = np.concatenate(scores_entropy)
    accuracies = np.concatenate(accuracies)

    return scores, accuracies

def get_auroc_ood(true_dataset, ood_dataset, model):
    dataloader, anomaly_targets = prepare_ood_datasets(true_dataset, ood_dataset)

    scores, accuracies = loop_over_dataloader(model, dataloader)

    accuracy = np.mean(accuracies[: len(true_dataset)])
    roc_auc = roc_auc_score(anomaly_targets, scores)


    in_n_samples = (anomaly_targets==0).sum()
    ood_n_samples = anomaly_targets.shape[0] - in_n_samples
    in_scores = scores[:in_n_samples]
    ood_scores = scores[in_n_samples:]
    quantile_tpr95 = torch.quantile(torch.tensor(in_scores), 0.95).cpu().numpy()
    fpr95 = (ood_scores < quantile_tpr95).sum() /ood_n_samples
    #print('FPR95:',fpr95.item())
    return accuracy, roc_auc, fpr95

def get_auroc_ood_snn(true_dataset, ood_dataset, model):
    dataloader, anomaly_targets = prepare_ood_datasets(true_dataset, ood_dataset)

    scores, accuracies = loop_over_dataloader_snn(model, dataloader)

    accuracy = np.mean(accuracies[: len(true_dataset)])
    roc_auc = roc_auc_score(anomaly_targets, scores)


    in_n_samples = (anomaly_targets==0).sum()
    ood_n_samples = anomaly_targets.shape[0] - in_n_samples
    in_scores = scores[:in_n_samples]
    ood_scores = scores[in_n_samples:]
    quantile_tpr95 = torch.quantile(torch.tensor(in_scores), 0.95).cpu().numpy()
    fpr95 = (ood_scores < quantile_tpr95).sum() /ood_n_samples
    #print('FPR95:',fpr95.item())
    return accuracy, roc_auc, fpr95

def get_auroc_ood_snn_be(true_dataset, ood_dataset, model, num_models=4):
    dataloader, anomaly_targets = prepare_ood_datasets(true_dataset, ood_dataset)

    scores, accuracies = loop_over_dataloader_snn_be(model, dataloader, num_model=num_models)
    accuracy = np.mean(accuracies[: len(true_dataset)])
    roc_auc = roc_auc_score(anomaly_targets, scores)
    auprc=average_precision_score(anomaly_targets, scores)

    in_n_samples = (anomaly_targets==0).sum()
    ood_n_samples = anomaly_targets.shape[0] - in_n_samples
    in_scores = scores[:in_n_samples]
    ood_scores = scores[in_n_samples:]
    quantile_tpr95 = torch.quantile(torch.tensor(in_scores), 0.95).cpu().numpy()
    fpr95 = (ood_scores < quantile_tpr95).sum() /ood_n_samples
    #print('FPR95:',fpr95.item())
    return accuracy, roc_auc, fpr95, auprc

def get_auroc_ood_conformal(true_dataset, ood_dataset, held_out, model):
    dataloader, anomaly_targets = prepare_ood_datasets(true_dataset, ood_dataset)
    held_out_dataloader = torch.utils.data.DataLoader(
        held_out, batch_size=500, shuffle=False, num_workers=0
    )
    for i, (data, target) in enumerate(held_out_dataloader):
        n = data.shape[0]
        alpha = 0.1
        data = data.cuda()
        target = target.cuda()
        conformal_scores = 1-model(data)[[i for i in range(data.shape[0])], target]
        qhat = torch.quantile(conformal_scores, np.ceil((n+1)*(1-alpha))/n )
        break
    scores, accuracies = loop_over_dataloader_conformal(model, dataloader, held_out_dataloader, qhat)
    accuracy = np.mean(accuracies[: len(true_dataset)])
    roc_auc = roc_auc_score(anomaly_targets, scores)


    in_n_samples = (anomaly_targets==0).sum()
    ood_n_samples = anomaly_targets.shape[0] - in_n_samples
    in_scores = scores[:in_n_samples]
    ood_scores = scores[in_n_samples:]
    quantile_tpr95 = torch.quantile(torch.tensor(in_scores), 0.95).cpu().numpy()
    fpr95 = (ood_scores < quantile_tpr95).sum() /ood_n_samples
    #print('FPR95:',fpr95.item())
    return accuracy, roc_auc, fpr95

def get_auroc_classification_be(dataset, model):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=500, shuffle=False, num_workers=4, pin_memory=False
    )

    scores, accuracies = loop_over_dataloader(model, dataloader)

    accuracy = np.mean(accuracies)
    roc_auc = roc_auc_score(1 - accuracies, scores)
    auprc = average_precision_score(1 - accuracies, scores)

    return accuracy, roc_auc, auprc


def get_cifar_svhn_ood_be(model):
    _, _, _, cifar_test_dataset = get_CIFAR10()
    _, _, _, svhn_test_dataset = get_SVHN()

    return get_auroc_ood_snn_be(cifar_test_dataset, svhn_test_dataset, model)


def get_fashionmnist_mnist_ood(model):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, mnist_test_dataset = get_MNIST()

    return get_auroc_ood(fashionmnist_test_dataset, mnist_test_dataset, model)


def get_fashionmnist_notmnist_ood(model):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, notmnist_test_dataset = get_notMNIST()

    return get_auroc_ood(fashionmnist_test_dataset, notmnist_test_dataset, model)

def get_cifar_svhn_ood_snn(model):
    _, _, _, cifar_test_dataset = get_CIFAR10()
    _, _, _, svhn_test_dataset = get_SVHN()

    return get_auroc_ood_snn(cifar_test_dataset, svhn_test_dataset, model)


def get_fashionmnist_mnist_ood_snn(model):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, mnist_test_dataset = get_MNIST()

    return get_auroc_ood_snn(fashionmnist_test_dataset, mnist_test_dataset, model)


def get_fashionmnist_notmnist_ood_snn(model):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, notmnist_test_dataset = get_notMNIST()

    return get_auroc_ood_snn(fashionmnist_test_dataset, notmnist_test_dataset, model)

def get_cifar_svhn_ood_snn_be(model, num_model):
    _, _, _, cifar_test_dataset = get_CIFAR10()
    _, _, _, svhn_test_dataset = get_SVHN()

    return get_auroc_ood_snn_be(cifar_test_dataset, svhn_test_dataset, model, num_models=num_model)


def get_fashionmnist_mnist_ood_snn_be(model, num_model):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, mnist_test_dataset = get_MNIST()

    return get_auroc_ood_snn_be(fashionmnist_test_dataset, mnist_test_dataset, model, num_models=num_model)


def get_fashionmnist_notmnist_ood_snn_be(model, num_model):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, notmnist_test_dataset = get_notMNIST()

    return get_auroc_ood_snn_be(fashionmnist_test_dataset, notmnist_test_dataset, model, num_models=num_model)

def get_cifar_svhn_ood_conformal(model, num_model):
    _, _, _, cifar_test_dataset = get_CIFAR10()
    _, _, _, svhn_test_dataset = get_SVHN()

    return get_auroc_ood_conformal(cifar_test_dataset, svhn_test_dataset, svhn_test_dataset, model)


def get_fashionmnist_mnist_ood_conformal(model):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, mnist_test_dataset = get_MNIST()

    return get_auroc_ood_conformal(fashionmnist_test_dataset, mnist_test_dataset, fashionmnist_test_dataset, model)


def get_fashionmnist_notmnist_ood_conformal(model):
    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()
    _, _, _, mnist_test_dataset = get_MNIST()
    _, _, _, notmnist_test_dataset = get_notMNIST()

    return get_auroc_ood_conformal(fashionmnist_test_dataset, notmnist_test_dataset, fashionmnist_test_dataset,model)