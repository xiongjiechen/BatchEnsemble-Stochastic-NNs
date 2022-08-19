from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
import numpy as np
import torch.nn.functional as F
import random

def dataset_loader(args):
    if args.dataset == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                               transforms.Normalize((0.1307,), (0.3081,))]),
                           download=True),
            batch_size=args.batch_size, shuffle=True, drop_last=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                                transforms.Normalize((0.1307,),
                                                                                                     (0.3081,))])),
            batch_size=args.batch_size, shuffle=True, drop_last=True)
    elif args.dataset == 'FashionMNIST':
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.2861,), (0.3530,))]
        transform = transforms.Compose(transform_list)
        train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data_fashion', train=True, download=True,
                                                                         transform=transform),
                                                   batch_size=args.batch_size, shuffle=True, drop_last=True)

        test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data_fashion', train=False,
                                                                        transform=transform),
                                                  batch_size=args.batch_size, shuffle=True, drop_last=True)
    if args.ood_dataset == 'MNIST':
        ood_train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                               transforms.Normalize((0.2861,), (0.3530,))]),
                           download=True),
            batch_size=args.batch_size, shuffle=True, drop_last=True)

        ood_test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                                transforms.Normalize((0.2861,),
                                                                                                     (0.3530,))])),
            batch_size=args.batch_size, shuffle=True, drop_last=True)

    elif args.ood_dataset == 'FashionMNIST':
        transform_list = [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        transform = transforms.Compose(transform_list)
        ood_train_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data_fashion', train=True, download=True,
                                                                             transform=transform),
                                                       batch_size=args.batch_size, shuffle=True, drop_last=True)

        ood_test_loader = torch.utils.data.DataLoader(datasets.FashionMNIST('./data_fashion', train=False,
                                                                            transform=transform),
                                                      batch_size=args.batch_size, shuffle=True, drop_last=True)
    return train_loader, test_loader, ood_train_loader, ood_test_loader


def kde(sample, base, sigma=1.0, n_ensemble=1):
    # sample : batch x feature
    # base: class x num_sample x feature
    # print(sample.shape,base.shape)
    # print(sample[:,None,None,:].shape, base[None,:,:,:].shape)
    d = sample[:, None, None, :] - base[None, :, :, :]

    n_samples, n_class, n_prototype, prototype_dim = d.shape
    if n_ensemble > 1:
        d = d.reshape([n_samples, n_class, n_ensemble, n_prototype // n_ensemble, prototype_dim])
    d = d ** 2
    d = d.sum(dim=-1)
    kernel = -d
    return kernel.mean(dim=-1)


def hausdorff_distance(sample, base, sigma=1.0, n_ensemble=1):
    d = sample[:, None, None, :] - base[None, :, :, :]
    n_samples, n_class, n_prototype, prototype_dim = d.shape
    if n_ensemble > 1:
        d = d.reshape([n_samples, n_class, n_ensemble, n_prototype // n_ensemble, prototype_dim])
    d = d ** 2
    d = d.sum(dim=-1)
    hausdorff_d = torch.max(d, dim=-1)[0]
    kernel = -hausdorff_d
    return kernel


plt.style.use('seaborn-whitegrid')

def calculate_ece(n_samples, acc_conf_interval_raw):
    ece = 0
    for acc_conf_i in acc_conf_interval_raw:
        if acc_conf_i.shape[0]!=0:
            #print(acc_conf_i)
            ece += acc_conf_i.shape[0]/n_samples*np.abs(acc_conf_i[:,0].mean()-acc_conf_i[:,1].mean())
    return ece

def accuracy_confidence_data(acc_conf_recorder,length=0.1):
    acc_conf_recorder = np.array(acc_conf_recorder).reshape(-1,2)
    intervals = np.arange(0,1.1,length)
    index_conf = []
    for i in range(len(intervals)-1):
        index_conf.append( np.where( ( acc_conf_recorder[:,1]<=intervals[i+1]) &
                                   (acc_conf_recorder[:,1]>intervals[i]) )[0] )
    acc_conf_interval_raw = []
    for i in range(len(index_conf)):
        acc_conf_interval_raw.append(acc_conf_recorder[index_conf[i]])

    lambda_x=lambda i,x: np.array([[0,(i+0.5)/10]]) if x.shape[0]==0 else x
    acc_conf_interval=[lambda_x(i,x) for i,x in enumerate(acc_conf_interval_raw)]
    acc_conf_interval_mean = [x.mean(0) for x in acc_conf_interval]
    acc_conf_interval_mean = np.array(acc_conf_interval_mean)
    return acc_conf_recorder, intervals, acc_conf_interval_raw, acc_conf_interval,acc_conf_interval_mean

def plot_bar(acc_conf_recorder, intervals, acc_conf_interval_raw, acc_conf_interval_mean, sigma,regularizer, distance, dataset='CIFAR10',save=False):
    plt.grid(linestyle='--', linewidth = 3)
    bar=plt.bar(intervals[1:]-0.05,acc_conf_interval_mean[:,0],
            width=0.1,align='center', edgecolor='black', color='green', linewidth=5)
    plt.plot(np.arange(0,1.02,0.02),np.arange(0,1.02,0.02),linestyle='--', linewidth=5,color='red',  dashes=(3, 2), alpha=0.5)

    n_samples=[x.shape[0] for x in acc_conf_interval_raw]
    for i,rect in enumerate(np.array(bar)):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2.0, height+0.015, '{:.1f}%'.format(100*n_samples[i]/acc_conf_recorder.shape[0]), ha='center', va='bottom', fontsize=12, color='darkblue')
    ece = calculate_ece(acc_conf_recorder.shape[0], acc_conf_interval_raw)
    plt.xlabel('Confidence',fontsize=20)
    plt.ylabel('Accuracy',fontsize=20)
    plt.xlim(0,1.0)
    plt.ylim(0,1.1)
    plt.title('Test Accuracy: {:.3f}%\n ECE: {:.3f}%'.format(100*acc_conf_recorder[:,0].mean(), 100*ece))
    fig = plt.gcf()
    plt.show()
    if save:
        fig.savefig('{}_{}_{}_ece_tempering_{:.1f}.jpg'.format(regularizer, distance,dataset,sigma), bbox_inches='tight')
    #return ece

def plot_ece(model, plot_dataset_loader, dataset, method='softmax', save=False, device='cuda',args=None, adaptive_logit=False, threshold=-0.25):
    acc_conf_recorder = []
    plt.figure()
    for (data, target) in plot_dataset_loader:
        test_feature = model(data.to(device),method=args.distance)
        test_feature_logit, sigma_ = model.classify(test_feature,sampling = True,
                                            n_samples=args.sampling_centroids, method=method, n_ensemble = args.n_ensemble)
        test_feature_logit = test_feature_logit.clone().reshape(test_feature_logit.shape[0],
                                                                test_feature_logit.shape[1], -1)
        if adaptive_logit:
            far_index = torch.where(test_feature_logit.detach().clone().max(dim=1)[0] < threshold)
            test_feature_logit[far_index[0], :, far_index[1]] = (test_feature_logit.detach().clone().max(dim=1)[0]).exp()[far_index[0], far_index[1]][:,None]\
                                                                * test_feature_logit.clone()[far_index[0], :, far_index[1]]
            #test_feature_logit = test_feature_logit.detach().clone().max(dim=1)[0].exp()[:,None,:]*test_feature_logit.clone()#

        pred_ensemble = F.softmax(test_feature_logit,dim=1).reshape([args.batch_size, args.num_class,-1])
        pred = pred_ensemble.mean(-1)
        acc_conf_recorder.append(np.concatenate(((pred.cpu().argmax(-1)==target).float().detach().numpy().reshape(-1,1),
                                  pred.max(-1,keepdim=True)[0].cpu().float().detach().numpy()),axis=-1))
    acc_conf_recorder, intervals, acc_conf_interval_raw, acc_conf_interval,acc_conf_interval_mean = accuracy_confidence_data(acc_conf_recorder)
    plot_bar(acc_conf_recorder, intervals, acc_conf_interval_raw, acc_conf_interval_mean, 1.0, args.centroid_reg, args.distance, dataset= dataset, save=save)


def plot_kernel_distance(model, dataset_loader, OOD_dataset_loader, dataset, OOD_dataset, save=False, method='softmax', args=None, adaptive_logit=False, threshold=-0.25):
    acc_conf_recorder = []
    plt.figure()
    for (data, target) in dataset_loader:
        test_feature = model(data.to(args.device), method=args.distance)
        test_feature_logit, sigma_ = model.classify(test_feature, sampling=True,
                                            n_samples=args.sampling_centroids, method=method, n_ensemble=args.n_ensemble)
        test_feature_logit = test_feature_logit.clone().reshape(test_feature_logit.shape[0],
                                                                test_feature_logit.shape[1], -1)
        if adaptive_logit:
            far_index = torch.where(test_feature_logit.detach().clone().max(dim=1)[0] < threshold)
            test_feature_logit[far_index[0], :, far_index[1]] = test_feature_logit.detach().clone().max(dim=1)[0].exp()[
                                                                    far_index[0], far_index[1]][:, None] \
                                                                * test_feature_logit.clone()[far_index[0], :,
                                                                  far_index[1]]
            # test_feature_logit = test_feature_logit.detach().clone().max(dim=1)[0].exp()[:,None,:]*test_feature_logit.clone()#

        pred_ensemble = F.softmax(test_feature_logit, dim=1).reshape([args.batch_size, args.num_class, -1])
        pred = pred_ensemble.mean(-1)

        acc_conf_recorder.append(
            np.concatenate(((pred.cpu().argmax(-1) == target).float().detach().numpy().reshape(-1, 1),
                            pred.max(-1, keepdim=True)[0].cpu().float().detach().numpy()), axis=-1))
    acc_conf_recorder, intervals, acc_conf_interval_raw, acc_conf_interval, acc_conf_interval_mean = accuracy_confidence_data(
        acc_conf_recorder)
    plt.hist(acc_conf_recorder[:, 1], width=0.007, align='mid', edgecolor='None', color='orange', linewidth=5,
             label=dataset + ' probability of predicted class', bins=100)
    plt.xlim([0, 1])
    plt.xticks(np.arange(0.2, 1.2, 0.2))
    quantile_tpr95 = torch.quantile(torch.tensor(acc_conf_recorder[:, 1]), 0.05).cpu().numpy()

    acc_conf_recorder = []
    for (data, target) in OOD_dataset_loader:
        #         test_feature = model(data.to(device),method=method)
        #         test_feature_logit = model.classify(test_feature,sigma=sigma,sampling = sampling,
        #                                             n_samples=n_sample_prototype, method=method)
        #         if method=='softmax':
        #             test_feature_logit=F.softmax(test_feature_logit, dim=-1)
        test_feature = model(data.to(args.device), method=args.distance)
        test_feature_logit, sigma_ = model.classify(test_feature, sampling=True,
                                            n_samples=args.sampling_centroids, method=method, n_ensemble=args.n_ensemble)
        test_feature_logit = test_feature_logit.clone().reshape(test_feature_logit.shape[0],
                                                                test_feature_logit.shape[1], -1)
        if adaptive_logit:
            far_index = torch.where(test_feature_logit.detach().clone().max(dim=1)[0] < threshold)
            test_feature_logit[far_index[0], :, far_index[1]] = test_feature_logit.detach().clone().max(dim=1)[0].exp()[far_index[0], far_index[1]][:,None]\
                                                                * test_feature_logit.clone()[far_index[0], :, far_index[1]]
            #test_feature_logit = test_feature_logit.detach().clone().max(dim=1)[0].exp()[:,None,:]*test_feature_logit.clone()#

        pred_ensemble = F.softmax(test_feature_logit, dim=1).reshape([args.batch_size, args.num_class, -1])
        pred = pred_ensemble.mean(-1)
        acc_conf_recorder.append(
            np.concatenate(((pred.cpu().argmax(-1) == target).float().detach().numpy().reshape(-1, 1),
                            pred.max(-1, keepdim=True)[0].cpu().float().detach().numpy()), axis=-1))
    acc_conf_recorder, intervals, acc_conf_interval_raw, acc_conf_interval, acc_conf_interval_mean = accuracy_confidence_data(
        acc_conf_recorder)

    acc_conf_recorder = np.array(acc_conf_recorder).reshape([-1,2])
    fpr95=(acc_conf_recorder[:, 1]>quantile_tpr95).sum()/acc_conf_recorder.shape[0]
    print("Kernel distance FPR95:{:.3f}%".format(fpr95*100))

    plt.hist(acc_conf_recorder[:, 1], width=0.007, align='mid', edgecolor='None', color='royalblue', linewidth=5,
             label='OOD_' + OOD_dataset + ' probability of predicted class', bins=100)
    plt.xlim([0, 1])
    plt.xticks(np.arange(0.2, 1.2, 0.2))

    plt.legend(fontsize=20)
    fig = plt.gcf()
    plt.show()
    if save:
        fig.savefig('{}_{}_{}_vs_{}_conf_tempering_{:.1f}.jpg'.format(args.centroid_reg, method, dataset, OOD_dataset, 1.0),
                    bbox_inches='tight')

def plot_distance(model, dataset_loader, OOD_dataset_loader, dataset, OOD_dataset, save=False, method='softmax', args=None, width=0.2, adaptive_logit=False):
    acc_logit_recorder = []
    plt.figure()
    for (data, target) in dataset_loader:
        #         test_feature = model(data.to(device),method=method)
        #         test_feature_logit = model.classify(test_feature,sigma=sigma,sampling = sampling,
        #                                             n_samples=n_sample_prototype, method=method)
        #         if method=='softmax':
        #             test_feature_logit=F.softmax(test_feature_logit, dim=-1)
        test_feature = model(data.to(args.device), method=args.distance)
        test_feature_logit, sigma_ = model.classify(test_feature, sampling=True,
                                            n_samples=args.sampling_centroids, method=method, n_ensemble=args.n_ensemble)
        test_feature_logit = test_feature_logit.clone().reshape(test_feature_logit.shape[0], test_feature_logit.shape[1], -1)
        logit = test_feature_logit.mean(-1)

        acc_logit_recorder.append(
            np.concatenate(((logit.cpu().argmax(-1) == target).float().detach().numpy().reshape(-1, 1),
                            logit.max(-1, keepdim=True)[0].cpu().float().detach().numpy()), axis=-1))
    acc_logit_recorder, intervals, acc_conf_interval_raw, acc_conf_interval, acc_conf_interval_mean = accuracy_confidence_data(
        acc_logit_recorder)
    plt.hist(-acc_logit_recorder[:, 1], width=width, align='mid', edgecolor='None', color='orange', linewidth=5,
             label=dataset + ' logit', bins=100)

    acc_logit_recorder = []
    for (data, target) in OOD_dataset_loader:
        test_feature = model(data.to(args.device), method=args.distance)
        test_feature_logit, sigma_ = model.classify(test_feature, sampling=True,
                                            n_samples=args.sampling_centroids, method=method, n_ensemble=args.n_ensemble)
        test_feature_logit = test_feature_logit.clone().reshape(test_feature_logit.shape[0],
                                                                test_feature_logit.shape[1], -1)
        logit = test_feature_logit.mean(-1)

        acc_logit_recorder.append(
            np.concatenate(((logit.cpu().argmax(-1) == target).float().detach().numpy().reshape(-1, 1),
                            logit.max(-1, keepdim=True)[0].cpu().float().detach().numpy()), axis=-1))
    acc_logit_recorder, intervals, acc_conf_interval_raw, acc_conf_interval, acc_conf_interval_mean = accuracy_confidence_data(
        acc_logit_recorder)
    plt.hist(-acc_logit_recorder[:, 1], width=width, align='mid', edgecolor='None', color='royalblue', linewidth=5,
             label='OOD_' + OOD_dataset + ' logit', bins=100)
    plt.legend(fontsize=20)
    fig = plt.gcf()
    plt.show()
    if save:
        fig.savefig('{}_{}_{}_vs_{}_tempering_{:.1f}_logit.jpg'.format(args.centroid_reg, method, dataset, OOD_dataset, 1.0),
                    bbox_inches='tight')

def plot_exp_distance(model, dataset_loader, OOD_dataset_loader, dataset, OOD_dataset, save=False, method='softmax', args=None, width=0.0035, adaptive_logit=False):
    acc_logit_recorder = []
    plt.figure()
    for (data, target) in dataset_loader:
        #         test_feature = model(data.to(device),method=method)
        #         test_feature_logit = model.classify(test_feature,sigma=sigma,sampling = sampling,
        #                                             n_samples=n_sample_prototype, method=method)
        #         if method=='softmax':
        #             test_feature_logit=F.softmax(test_feature_logit, dim=-1)
        test_feature = model(data.to(args.device), method=args.distance)
        test_feature_logit, sigma_ = model.classify(test_feature, sampling=True,
                                            n_samples=args.sampling_centroids, method=method, n_ensemble=args.n_ensemble)
        test_feature_logit = test_feature_logit.clone().reshape(test_feature_logit.shape[0],
                                                                test_feature_logit.shape[1], -1)
        logit = test_feature_logit.mean(-1)

        acc_logit_recorder.append(
            np.concatenate(((logit.cpu().argmax(-1) == target).float().detach().numpy().reshape(-1, 1),
                            logit.max(-1, keepdim=True)[0].cpu().float().detach().numpy()), axis=-1))
    acc_logit_recorder, intervals, acc_conf_interval_raw, acc_conf_interval, acc_conf_interval_mean = accuracy_confidence_data(
        acc_logit_recorder)
    plt.hist(np.exp(acc_logit_recorder[:, 1]), width=width, align='mid', edgecolor='None', color='orange', linewidth=5,
             label=dataset + ' exp logit', bins=100)
    quantile_tpr95 = torch.quantile(torch.tensor(np.exp(acc_logit_recorder[:, 1])), 0.05).cpu().numpy()

    acc_logit_recorder = []
    for (data, target) in OOD_dataset_loader:
        test_feature = model(data.to(args.device), method=args.distance)
        test_feature_logit, sigma_ = model.classify(test_feature, sampling=True,
                                            n_samples=args.sampling_centroids, method=method, n_ensemble=args.n_ensemble)
        test_feature_logit = test_feature_logit.clone().reshape(test_feature_logit.shape[0],
                                                                test_feature_logit.shape[1], -1)
        logit = test_feature_logit.mean(-1)

        acc_logit_recorder.append(
            np.concatenate(((logit.cpu().argmax(-1) == target).float().detach().numpy().reshape(-1, 1),
                            logit.max(-1, keepdim=True)[0].cpu().float().detach().numpy()), axis=-1))
    acc_logit_recorder, intervals, acc_conf_interval_raw, acc_conf_interval, acc_conf_interval_mean = accuracy_confidence_data(
        acc_logit_recorder)

    acc_logit_recorder = np.array(acc_logit_recorder).reshape([-1,2])
    fpr95=(np.exp(acc_logit_recorder[:, 1])>quantile_tpr95).sum()/acc_logit_recorder.shape[0]
    print("Exp distance FPR95:{:.3f}%".format(fpr95*100))
    plt.hist(np.exp(acc_logit_recorder[:, 1]), width=width, align='mid', edgecolor='None', color='royalblue', linewidth=5,
             label='OOD_' + OOD_dataset + ' exp logit', bins=100)
    plt.legend(fontsize=20)
    fig = plt.gcf()
    plt.show()
    if save:
        fig.savefig('{}_{}_{}_vs_{}_tempering_{:.1f}_exp_logit.jpg'.format(args.centroid_reg, method, dataset, OOD_dataset, 1.0),
                    bbox_inches='tight')

def plot_entropy(model, dataset_loader, OOD_dataset_loader, dataset, OOD_dataset, save=False, method='softmax', args=None, adaptive_logit=False, threshold=-0.25):
    acc_entropy_recorder = []
    plt.figure()
    for (data, target) in dataset_loader:
        test_feature = model(data.to(args.device), method=args.distance)
        test_feature_logit, sigma_ = model.classify(test_feature, sampling=True,
                                            n_samples=args.sampling_centroids, method=method, n_ensemble=args.n_ensemble)
        test_feature_logit = test_feature_logit.clone().reshape(test_feature_logit.shape[0],
                                                                test_feature_logit.shape[1], -1)
        if adaptive_logit:
            far_index = torch.where(test_feature_logit.detach().clone().max(dim=1)[0] < threshold)
            test_feature_logit[far_index[0], :, far_index[1]] = test_feature_logit.detach().clone().max(dim=1)[0].exp()[far_index[0], far_index[1]][:,None]\
                                                                * test_feature_logit.clone()[far_index[0], :, far_index[1]]
            #test_feature_logit = test_feature_logit.detach().clone().max(dim=1)[0].exp()[:,None,:]*test_feature_logit.clone()#

        pred_ensemble = F.softmax(test_feature_logit, dim=1).reshape([args.batch_size, args.num_class, -1])
        pred = pred_ensemble.mean(-1)

        acc_entropy_recorder.append( (-pred*( torch.log(pred+1e-20) )).sum(-1, keepdim=True).cpu().float().detach().numpy() )
    acc_entropy_recorder = np.array(acc_entropy_recorder).reshape([-1])
    plt.hist(acc_entropy_recorder, width=0.015, align='mid', edgecolor='None', color='orange', linewidth=5,
             label=dataset + ' entropy', bins=100)
    quantile_tpr95 = torch.quantile(torch.tensor(acc_entropy_recorder), 0.95).cpu().numpy()
    acc_entropy_recorder = []
    for (data, target) in OOD_dataset_loader:
        test_feature = model(data.to(args.device), method=args.distance)
        test_feature_logit, sigma_ = model.classify(test_feature, sampling=True,
                                            n_samples=args.sampling_centroids, method=method, n_ensemble=args.n_ensemble)
        test_feature_logit = test_feature_logit.clone().reshape(test_feature_logit.shape[0],
                                                                test_feature_logit.shape[1], -1)
        if adaptive_logit:
            far_index = torch.where(test_feature_logit.detach().clone().max(dim=1)[0] < threshold)
            test_feature_logit[far_index[0], :, far_index[1]] = test_feature_logit.detach().clone().max(dim=1)[0].exp()[far_index[0], far_index[1]][:,None]\
                                                                * test_feature_logit.clone()[far_index[0], :, far_index[1]]
            #test_feature_logit = test_feature_logit.detach().clone().max(dim=1)[0].exp()[:,None,:]*test_feature_logit.clone()#

        pred_ensemble = F.softmax(test_feature_logit, dim=1).reshape([args.batch_size, args.num_class, -1])
        pred = pred_ensemble.mean(-1)

        acc_entropy_recorder.append( (-pred*( torch.log(pred+1e-20) )).sum(-1, keepdim=True).cpu().float().detach().numpy() )

    acc_entropy_recorder = np.array(acc_entropy_recorder).reshape([-1])
    fpr95=(acc_entropy_recorder<quantile_tpr95).sum()/acc_entropy_recorder.shape[0]
    print("Entropy FPR95:{:.3f}%".format(fpr95*100))
    plt.hist(acc_entropy_recorder, width=0.015, align='mid', edgecolor='None', color='royalblue', linewidth=5,
             label='OOD_' + OOD_dataset + ' entropy', bins=100)
    plt.legend(fontsize=20)
    fig = plt.gcf()
    plt.show()
    if save:
        fig.savefig('{}_{}_{}_vs_{}_entropy_tempering_{:.1f}.jpg'.format(args.centroid_reg, method, dataset, OOD_dataset, 1.0),
                    bbox_inches='tight')

def plot_mutual_information(model, dataset_loader, OOD_dataset_loader, dataset, OOD_dataset, save=False, method='softmax', args=None, adaptive_logit=False, threshold=-0.25):
    entropy_recorder = []
    entropy_individual_recorder = []
    plt.figure()
    for (data, target) in dataset_loader:
        test_feature = model(data.to(args.device), method=args.distance)
        test_feature_logit, sigma_ = model.classify(test_feature, sampling=True,
                                            n_samples=args.sampling_centroids, method=method, n_ensemble=args.n_ensemble)
        test_feature_logit = test_feature_logit.clone().reshape(test_feature_logit.shape[0],
                                                                test_feature_logit.shape[1], -1)
        if adaptive_logit:
            far_index = torch.where(test_feature_logit.detach().clone().max(dim=1)[0] < threshold)
            test_feature_logit[far_index[0], :, far_index[1]] = test_feature_logit.detach().clone().max(dim=1)[0].exp()[far_index[0], far_index[1]][:,None]\
                                                                * test_feature_logit.clone()[far_index[0], :, far_index[1]]
            #test_feature_logit = test_feature_logit.detach().clone().max(dim=1)[0].exp()[:,None,:]*test_feature_logit.clone()#

        pred_ensemble = F.softmax(test_feature_logit, dim=1).reshape([args.batch_size, args.num_class, -1])
        pred = pred_ensemble.mean(-1)

        entropy_recorder.append( (-pred*( torch.log(pred+1e-20) )).sum(-1, keepdim=True).cpu().float().detach().numpy() )
        entropy_individual_recorder.append( (-pred_ensemble*( torch.log(pred_ensemble+1e-20) )).sum(1, keepdim=True).mean(-1).cpu().float().detach().numpy() )
    entropy_recorder = np.array(entropy_recorder).reshape([-1])
    entropy_individual_recorder = np.array(entropy_individual_recorder).reshape([-1])
    mutual_information = entropy_recorder - entropy_individual_recorder
    plt.hist(mutual_information, align='mid', edgecolor='None', color='orange', linewidth=5,
             label=dataset + ' MI', bins=100)

    entropy_recorder = []
    entropy_individual_recorder = []

    for (data, target) in OOD_dataset_loader:
        test_feature = model(data.to(args.device), method=args.distance)
        test_feature_logit, sigma_ = model.classify(test_feature, sampling=True,
                                            n_samples=args.sampling_centroids, method=method, n_ensemble=args.n_ensemble)
        test_feature_logit = test_feature_logit.clone().reshape(test_feature_logit.shape[0],
                                                                test_feature_logit.shape[1], -1)
        if adaptive_logit:
            far_index = torch.where(test_feature_logit.detach().clone().max(dim=1)[0] < threshold)
            test_feature_logit[far_index[0], :, far_index[1]] = test_feature_logit.detach().clone().max(dim=1)[0].exp()[far_index[0], far_index[1]][:,None]\
                                                                * test_feature_logit.clone()[far_index[0], :, far_index[1]]
            #test_feature_logit = test_feature_logit.detach().clone().max(dim=1)[0].exp()[:,None,:]*test_feature_logit.clone()#

        pred_ensemble = F.softmax(test_feature_logit, dim=1).reshape([args.batch_size, args.num_class, -1])
        pred = pred_ensemble.mean(-1)

        entropy_recorder.append( (-pred*( torch.log(pred+1e-20) )).sum(-1, keepdim=True).cpu().float().detach().numpy() )
        entropy_individual_recorder.append( (-pred_ensemble*( torch.log(pred_ensemble+1e-20) )).sum(1, keepdim=True).mean(-1).cpu().float().detach().numpy() )

    entropy_recorder = np.array(entropy_recorder).reshape([-1])
    entropy_individual_recorder = np.array(entropy_individual_recorder).reshape([-1])
    mutual_information = entropy_recorder - entropy_individual_recorder

    plt.hist(mutual_information, align='mid', edgecolor='None', color='royalblue', linewidth=5,
             label='OOD_' + OOD_dataset + ' MI', bins=100)
    plt.legend(fontsize=20)
    fig = plt.gcf()
    plt.show()
    if save:
        fig.savefig('{}_{}_{}_vs_{}_MI_tempering_{:.1f}.jpg'.format(args.centroid_reg, method, dataset, OOD_dataset, 1.0),
                    bbox_inches='tight')

def loss_proto(model, prototype):

    if model.centroid_reg == 'variance':
        var_proto = ((prototype - prototype.mean(dim=1, keepdim=True)) ** 2).mean(dim=(1, 2))
        reg_term_proto = torch.relu(model.var_threshold - var_proto).mean()

    elif model.centroid_reg == 'entropy':
        entropy_proto = torch.cdist(prototype, prototype).sort(dim=-1)[0][:, :, 1:model.k + 1].mean(-1)
        # reg_term_proto = torch.relu(self.entropy_threshold - entropy_proto).mean()
        reg_term_proto = torch.relu(torch.exp(model.class_threshold) - entropy_proto.mean(-1)).mean()
    else:
        raise ValueError('Please select centroid regularizer from {variance, entropy}')

    return reg_term_proto

def loss_proto_be(model, prototype):

    if model.centroid_reg == 'variance':
        var_proto = ((prototype - prototype.mean(dim=1, keepdim=True)) ** 2).mean(dim=(1, 2))
        reg_term_proto = torch.relu(model.var_threshold - var_proto).mean()

    elif model.centroid_reg == 'entropy':
        entropy_proto = torch.cdist(prototype, prototype).sort(dim=-1)[0][:, :, :, 1:model.k + 1].mean([-1,-2])
        # reg_term_proto = torch.relu(self.entropy_threshold - entropy_proto).mean()
        reg_term_proto = torch.relu(model.class_threshold - entropy_proto).mean()
    else:
        raise ValueError('Please select centroid regularizer from {variance, entropy}')

    return reg_term_proto

def likelihood_loss(model, prototype, features, true_class, distance='kde', device='cuda', separate=False):
    # prototype = model.get_prototype(num_centroids=model.num_centroids)
    if separate:
        # prototype = prototype.clone().mean(1).transpose(0, 1)
        lki_target = likelihood_loss_separate(model, prototype, features, true_class, distance='kde', device='cuda')
        return lki_target.mean()
    else:
        if distance != 'softmax':
            d = features[:, None, None, :] - prototype[None, :, :, :]

            d = d ** 2
            d = d.sum(dim=-1)
            # kernel = -d
            # lki = torch.exp(-d)
            lki = -d
            index = torch.cat([torch.arange(0, features.shape[0])[..., None].to(device), true_class[..., None]], dim=-1)
            lki_target = lki[index[:, 0], index[:, 1]]
        else:
            lki_target = torch.ones([10]).to(device)
        return lki_target.mean()

def likelihood_loss_separate(model, prototype, features, true_class, distance='kde', device='cuda'):
    # prototype = model.get_prototype(num_centroids=model.num_centroids)

    if distance != 'softmax':
        d = features.transpose(1, 2)[:, :, None, :] - prototype[None, ...]
        d = d.clone()[torch.arange(0, d.shape[0]),true_class]
        d = d ** 2
        d = d.sum(dim=-1)
        # kernel = -d
        # lki = torch.exp(-d)
        lki_target = -d
    else:
        lki_target = torch.ones([10]).to(device)
    return lki_target.mean()

def likelihood_loss_be(model, prototype, features, true_class, distance='kde', device='cuda', separate=False):
    # prototype = model.get_prototype(num_centroids=model.num_centroids)
    if separate:
        # prototype = prototype.clone().mean(1).transpose(0, 1)
        lki_target = likelihood_loss_separate_be(model, prototype, features, true_class, distance='kde', device='cuda')
        return lki_target.mean()
    else:
        if distance != 'softmax':
            d = features[:, None, None, :] - prototype[None, :, :, :]

            d = d ** 2
            d = d.sum(dim=-1)
            # kernel = -d
            # lki = torch.exp(-d)
            lki = -d
            index = torch.cat([torch.arange(0, features.shape[0])[..., None].to(device), true_class[..., None]], dim=-1)
            lki_target = lki[index[:, 0], index[:, 1]]
        else:
            lki_target = torch.ones([10]).to(device)
        return lki_target.mean()

def likelihood_loss_separate_be(model, prototype, features, true_class, distance='kde', device='cuda'):
    # prototype = model.get_prototype(num_centroids=model.num_centroids)
    batch_size = features.shape[0]//model.num_models
    if distance != 'softmax':
        prototype_repeat = prototype[:, None, ...].repeat(1, batch_size, 1, 1, 1)
        d = features.transpose(1, 2).reshape([model.num_models,batch_size,model.num_classes,1,model.feature_dimension]) - prototype_repeat
        d = d.clone()[:, torch.arange(0, batch_size),true_class[:batch_size]]
        d = d ** 2
        d = d.sum(dim=-1)
        # kernel = -d
        # lki = torch.exp(-d)
        lki_target = -d
    else:
        lki_target = torch.ones([10]).to(device)
    return lki_target.mean()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

