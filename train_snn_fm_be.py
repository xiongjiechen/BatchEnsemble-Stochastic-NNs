import random
import numpy as np
import pickle
import torch
import torch.utils.data
from torch.nn import functional as F

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Loss

from ignite.contrib.handlers.tqdm_logger import ProgressBar

from utils.evaluate_ood import (
    get_fashionmnist_mnist_ood,
    get_fashionmnist_notmnist_ood,
    get_fashionmnist_mnist_ood_snn,
    get_fashionmnist_notmnist_ood_snn,
    get_fashionmnist_mnist_ood_snn_be,
    get_fashionmnist_notmnist_ood_snn_be
)
from utils.datasets import FastFashionMNIST, get_FashionMNIST
from utils.network_fm import CNN_DUQ, SNN, likelihood_loss, loss_proto, SNN_BE
from utils.utils import setup_seed, accuracy_confidence_data, plot_bar, plot_ece, calculate_ece
from networks.batchensemble_layers import *

def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 200):
        optim_factor = 3
    elif(epoch > 160):
        optim_factor = 2
    elif(epoch > 80):
        optim_factor = 1

    return init*math.pow(0.1, optim_factor)

def train_model(l_gradient_penalty, length_scale, final_model, num_models = 4):
    dataset = FastFashionMNIST("data/", train=True, download=True)
    test_dataset = FastFashionMNIST("data/", train=False, download=True)

    idx = list(range(60000))
    random.shuffle(idx)

    if final_model:
        train_dataset = dataset
        val_dataset = test_dataset
    else:
        train_dataset = torch.utils.data.Subset(dataset, indices=idx[:55000])
        val_dataset = torch.utils.data.Subset(dataset, indices=idx[55000:])

    num_classes = 10
    embedding_size = 256
    learnable_length_scale = False
    gamma = 0.999

    model = SNN_BE(
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
        num_models=num_models,
        gp=l_gradient_penalty
    )
    model = model.cuda()
    my_list = ['alpha', 'gamma']

    params_multi_tmp = list(filter(lambda kv: (my_list[0] in kv[0]) or (my_list[1] in kv[0]), model.named_parameters()))
    param_core_tmp = list(
        filter(lambda kv: (my_list[0] not in kv[0]) and (my_list[1] not in kv[0]), model.named_parameters()))
    params_multi = [param for name, param in params_multi_tmp]
    param_core = [param for name, param in param_core_tmp]
    optimizer =torch.optim.SGD([
                {'params': param_core,'weight_decay': 1e-4},
                {'params': params_multi, 'weight_decay': 0.0}
            ], lr=0.05, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40,60,80], gamma=0.1)
    # optimizer = torch.optim.SGD(
    #     model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4
    # )


    def output_transform_bce(output):
        y_pred, y, _, _ = output
        return y_pred, y

    def output_transform_acc(output):
        y_pred, y, _, _ = output
        return y_pred, torch.argmax(y, dim=1)

    def output_transform_gp(output):
        y_pred, y, x, y_pred_sum = output
        return x, y_pred_sum

    def tile(a, dim, n_tile, device='cuda'):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

    def calc_gradient_penalty(x, y_pred_sum):
        gradients = torch.autograd.grad(
            outputs=y_pred_sum,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred_sum),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        # L2 norm
        grad_norm = gradients.norm(2, dim=1)

        # Two sided penalty
        gradient_penalty = ((grad_norm - 1) ** 2).mean()

        return gradient_penalty

    def output_transform_ece(output):
        y_pred, y, x, y_pred_sum = output
        y_pred = y_pred / y_pred.sum(-1, keepdims=True)
        return y_pred, y

    def calc_ece(y_pred, y):
        acc_conf_recorder = []
        target = torch.where(y!=0)[1]
        acc_conf_recorder.append(np.concatenate(((y_pred.argmax(-1)==target).cpu().float().detach().numpy().reshape(-1,1),
                                  y_pred.max(-1,keepdim=True)[0].cpu().float().detach().numpy()),axis=-1))
        acc_conf_recorder, intervals, acc_conf_interval_raw, acc_conf_interval, acc_conf_interval_mean = accuracy_confidence_data(
            acc_conf_recorder)
        ece = calculate_ece(acc_conf_recorder.shape[0], acc_conf_interval_raw)
        return torch.tensor(ece*100, requires_grad=True).to('cuda')

    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        y = F.one_hot(y, num_classes=10).float()

        x, y = x.cuda(), y.cuda()

        # x = torch.cat([x for i in range(num_models)], dim=0)
        # y = torch.cat([y for i in range(num_models)], dim=0)
        x.requires_grad_(True)

        y_pred = model(x, y)
        y_pred_normalized = F.softmax(model.logit, dim=-1)#y_pred / y_pred.sum(-1, keepdims=True)
        loss = F.binary_cross_entropy(y_pred, y)

        loss +=  l_gradient_penalty * calc_gradient_penalty(x, y_pred.sum(1)) + model.loss_proto

        x.requires_grad_(False)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            model.update_embeddings(x, y)

        return loss.item()

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        y = F.one_hot(y, num_classes=10).float()

        x, y = x.cuda(), y.cuda()

        x.requires_grad_(True)

        y_pred = model(x)
        y_pred_normalized = y_pred / y_pred.sum(-1, keepdims=True)
        return y_pred, y, x, y_pred.sum(1)

    trainer = Engine(step)
    evaluator = Engine(eval_step)

    metric = Accuracy(output_transform=output_transform_acc)
    metric.attach(evaluator, "accuracy")

    metric = Loss(F.binary_cross_entropy, output_transform=output_transform_bce)
    metric.attach(evaluator, "bce")

    metric = Loss(calc_gradient_penalty, output_transform=output_transform_gp)
    metric.attach(evaluator, "gradient_penalty")

    metric = Loss(calc_ece, output_transform=output_transform_ece)
    metric.attach(evaluator, "ece")

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[10, 20], gamma=0.2
    )

    dl_train = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=0, drop_last=True
    )

    dl_val = torch.utils.data.DataLoader(
        val_dataset, batch_size=2000, shuffle=False, num_workers=0
    )

    dl_test = torch.utils.data.DataLoader(
        test_dataset, batch_size=2000, shuffle=False, num_workers=0
    )

    pbar = ProgressBar()
    pbar.attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        scheduler.step()

        if trainer.state.epoch % 5 == 0:
            evaluator.run(dl_val)
            _, roc_auc_mnist, fpr95_mnist, auprc_mnist = get_fashionmnist_mnist_ood_snn_be(model,num_model=num_models)
            _, roc_auc_notmnist, fpr95_notmnist, auprc_notmnist = get_fashionmnist_notmnist_ood_snn_be(model,num_model=num_models)

            metrics = evaluator.state.metrics

            print(
                f"Validation Results - Epoch: {trainer.state.epoch} "
                f"Acc: {metrics['accuracy']:.4f} "
                f"BCE: {metrics['bce']:.4f} "
                f"GP: {metrics['gradient_penalty']:.4f} "
                f"ECE: {metrics['ece']:.4f} "
                f"AUROC MNIST: {roc_auc_mnist:.4f} "
                f"AUPRC MNIST: {auprc_mnist:.4f} "
                f"FPR95 MNIST: {fpr95_mnist:.4f} "
                f"AUROC NotMNIST: {roc_auc_notmnist:.4f} "
                f"AUPRC NotMNIST: {auprc_notmnist:.4f} "
                f"FPR95 NotMNIST: {fpr95_notmnist:.4f} "
            )
            print(f"Sigma: {model.sigma}, GP: {model.gp}")

    trainer.run(dl_train, max_epochs=30)

    evaluator.run(dl_val)
    val_accuracy = evaluator.state.metrics["accuracy"]

    evaluator.run(dl_test)
    test_accuracy = evaluator.state.metrics["accuracy"]
    test_ece = evaluator.state.metrics["ece"]

    return model, val_accuracy, test_accuracy, test_ece

if __name__ == "__main__":

    seed = 12
    setup_seed(seed)
    num_models = 4

    _, _, _, fashionmnist_test_dataset = get_FashionMNIST()

    # Finding gradient penalty coefficient
    l_gradient_penalties = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0][1:]
    length_scales = [1.0]

    repetition = 5  # Increase for multiple repetitions
    final_model = False  # set true for final model to train on full train set

    results = {}

    for l_gradient_penalty in l_gradient_penalties:
        for length_scale in length_scales:
            val_accuracies = []
            test_accuracies = []
            test_eces = []
            roc_aucs_mnist = []
            roc_aucs_notmnist = []
            fpr95s_mnist = []
            fpr95s_notmnist = []
            auprcs_mnist = []
            auprcs_notmnist = []

            for _ in range(repetition):
                print(" ### NEW MODEL ### ")
                model, val_accuracy, test_accuracy, test_ece = train_model(
                    l_gradient_penalty, length_scale, final_model, num_models=num_models
                )
                accuracy, roc_auc_mnist, fpr95_mnist, auprc_mnist = get_fashionmnist_mnist_ood_snn_be(model,num_model=num_models)
                _, roc_auc_notmnist, fpr95_notmnist, auprc_notmnist = get_fashionmnist_notmnist_ood_snn_be(model,num_model=num_models)

                val_accuracies.append(val_accuracy)
                test_accuracies.append(test_accuracy)
                test_eces.append(test_ece)
                roc_aucs_mnist.append(roc_auc_mnist)
                roc_aucs_notmnist.append(roc_auc_notmnist)
                fpr95s_mnist.append(fpr95_mnist)
                fpr95s_notmnist.append(fpr95_notmnist)
                auprcs_mnist.append(auprc_mnist)
                auprcs_notmnist.append(auprc_notmnist)

            results[f"lgp{l_gradient_penalty}_ls{length_scale}"] = [
                (np.mean(val_accuracies), np.std(val_accuracies)),
                (np.mean(test_accuracies), np.std(test_accuracies)),
                (np.mean(test_eces), np.std(test_eces)),
                (np.mean(roc_aucs_mnist), np.std(roc_aucs_mnist)),
                (np.mean(fpr95s_mnist), np.std(fpr95s_mnist)),
                (np.mean(roc_aucs_notmnist), np.std(roc_aucs_notmnist)),
                (np.mean(fpr95s_notmnist), np.std(fpr95s_notmnist)),
                (np.mean(auprcs_notmnist), np.std(auprcs_notmnist))
            ]
            print(results[f"lgp{l_gradient_penalty}_ls{length_scale}"])

    print(results)
    with open('kernel_results.pkl', 'wb') as f:
        pickle.dump(results, f)
