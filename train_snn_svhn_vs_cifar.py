import argparse
import json
import pathlib
import random

import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard.writer import SummaryWriter

from torchvision.models import resnet18

from ignite.engine import Events, Engine
from ignite.metrics import Accuracy, Average, Loss
from ignite.contrib.handlers import ProgressBar

from utils.wide_resnet import WideResNet
from utils.resnet_duq import *
from utils.datasets import all_datasets
from utils.evaluate_ood import get_cifar_svhn_ood_be, get_auroc_classification_be
from utils.utils import setup_seed, accuracy_confidence_data, plot_bar, plot_ece, calculate_ece

def main(
    architecture,
    batch_size,
    length_scale,
    centroid_size,
    learning_rate,
    l_gradient_penalty,
    gamma,
    weight_decay,
    final_model,
    output_dir,
):
    writer = SummaryWriter(log_dir=f"runs/{output_dir}")

    ds = all_datasets["CIFAR10"]()
    input_size, num_classes, dataset, test_dataset = ds

    # Split up training set
    idx = list(range(len(dataset)))
    random.shuffle(idx)

    if final_model:
        train_dataset = dataset
        val_dataset = test_dataset
    else:
        val_size = int(len(dataset) * 0.8)
        train_dataset = torch.utils.data.Subset(dataset, idx[:val_size])
        val_dataset = torch.utils.data.Subset(dataset, idx[val_size:])

        val_dataset.transform = (
            test_dataset.transform
        )  # Test time preprocessing for validation

    model_output_size = 512
    epochs = 100
    milestones = [25, 50, 75]
    feature_extractor = ResNet_BE()

    if centroid_size is None:
        centroid_size = model_output_size

    model = SNN_BE_ResNet(
        num_classes,
        centroid_size,
        False,
        length_scale,
        gamma,
        feature_extractor,
    )
    model = model.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.2
    )

    def output_transform_bce(output):
        y_pred, y, _, _ = output
        return y_pred, y

    def output_transform_acc(output):
        y_pred, y, _, _ = output
        return y_pred, torch.argmax(y, dim=1)

    def output_transform_gp(output):
        y_pred, y, x, y_pred_sum = output
        return x, y_pred_sum

    def calc_gradients_input(x, y_pred):
        gradients = torch.autograd.grad(
            outputs=y_pred,
            inputs=x,
            grad_outputs=torch.ones_like(y_pred),
            create_graph=True,
        )[0]

        gradients = gradients.flatten(start_dim=1)

        return gradients

    def calc_gradient_penalty(x, y_pred):
        gradients = calc_gradients_input(x, y_pred)

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
        y_pred_normalized = F.softmax(model.logit, dim=-1)
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

    metric = Average()
    metric.attach(trainer, "loss")

    metric = Accuracy(output_transform=output_transform_acc)
    metric.attach(evaluator, "accuracy")

    metric = Loss(F.binary_cross_entropy, output_transform=output_transform_bce)
    metric.attach(evaluator, "bce")

    metric = Loss(calc_gradient_penalty, output_transform=output_transform_gp)
    metric.attach(evaluator, "gradient_penalty")

    metric = Loss(calc_ece, output_transform=output_transform_ece)
    metric.attach(evaluator, "ece")

    pbar = ProgressBar(dynamic_ncols=True)
    pbar.attach(trainer)

    kwargs = {"num_workers": 4, "pin_memory": True}

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(trainer):
        metrics = trainer.state.metrics
        loss = metrics["loss"]

        print(f"Train - Epoch: {trainer.state.epoch} Loss: {loss:.2f}")

        writer.add_scalar("Loss/train", loss, trainer.state.epoch)

        if trainer.state.epoch > (epochs - 5):
            accuracy, auroc, fpr95, auprc = get_cifar_svhn_ood_be(model)
            print(f"Test Accuracy: {accuracy}, AUROC: {auroc}, FPR95: {fpr95}, AUPRC: {auprc}")
            writer.add_scalar("OoD/test_accuracy", accuracy, trainer.state.epoch)
            writer.add_scalar("OoD/roc_auc", auroc, trainer.state.epoch)

            accuracy, auroc, auprc = get_auroc_classification_be(val_dataset, model)
            print(f"AUROC - uncertainty: {auroc}", f"FPR95 - uncertainty: {fpr95}", f"AUPRC - uncertainty: {auprc}")
            writer.add_scalar("OoD/val_accuracy", accuracy, trainer.state.epoch)
            writer.add_scalar("OoD/roc_auc_classification", auroc, trainer.state.epoch)

        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        acc = metrics["accuracy"]
        bce = metrics["bce"]
        GP = metrics["gradient_penalty"]
        loss = bce + l_gradient_penalty * GP+ model.loss_proto

        print(
            (
                f"Valid - Epoch: {trainer.state.epoch} "
                f"Acc: {acc:.4f} "
                f"Loss: {loss:.2f} "
                f"BCE: {bce:.2f} "
                f"GP: {GP:.2f} "
            )
        )

        writer.add_scalar("Loss/valid", loss, trainer.state.epoch)
        writer.add_scalar("BCE/valid", bce, trainer.state.epoch)
        writer.add_scalar("GP/valid", GP, trainer.state.epoch)
        writer.add_scalar("Accuracy/valid", acc, trainer.state.epoch)

        scheduler.step()

    trainer.run(train_loader, max_epochs=epochs)
    evaluator.run(test_loader)
    acc = evaluator.state.metrics["accuracy"]

    print(f"Test - Accuracy {acc:.4f}")

    torch.save(model.state_dict(), f"runs/{output_dir}/model.pt")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--architecture",
        default="ResNet18",
        choices=["ResNet18", "WRN"],
        help="Pick an architecture (default: ResNet18)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size to use for training (default: 128)",
    )

    parser.add_argument(
        "--centroid_size",
        type=int,
        default=None,
        help="Size to use for centroids (default: same as model output)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.05,
        help="Learning rate (default: 0.05)",
    )

    parser.add_argument(
        "--l_gradient_penalty",
        type=float,
        default=0.75,
        help="Weight for gradient penalty (default: 0.75)",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="Decay factor for exponential average (default: 0.999)",
    )

    parser.add_argument(
        "--length_scale",
        type=float,
        default=0.1,
        help="Length scale of RBF kernel (default: 0.1)",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=5e-4, help="Weight decay (default: 5e-4)"
    )

    parser.add_argument(
        "--output_dir", type=str, default="results", help="set output folder"
    )

    # Below setting cannot be used for model selection,
    # because the validation set equals the test set.
    parser.add_argument(
        "--final_model",
        action="store_true",
        default=False,
        help="Use entire training set for final model",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    print("input args:\n", json.dumps(kwargs, indent=4, separators=(",", ":")))

    pathlib.Path("./runs/" + args.output_dir).mkdir(exist_ok=True)

    main(**kwargs)
