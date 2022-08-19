import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import kde, hausdorff_distance, dataset_loader, plot_kernel_distance, plot_ece, plot_bar, \
    plot_entropy, accuracy_confidence_data, likelihood_loss, plot_distance, plot_mutual_information, plot_exp_distance, loss_proto, loss_proto_be, likelihood_loss_be
from networks.batchensemble_layers import *

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 128, 3)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(2 * 2 * 128, 256)

    def compute_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.flatten(1)

        x = F.relu(self.fc1(x))

        return x


class CNN_DUQ(Model):
    def __init__(
        self,
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, 256), 0.05)
        )

        self.register_buffer("N", torch.ones(num_classes) * 12)
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)

        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale

    def update_embeddings(self, x, y):
        z = self.last_layer(self.compute_features(x))

        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum

    def last_layer(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)
        return z

    def output_layer(self, z):
        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        distances = (-(diff**2)).mean(1).div(2 * self.sigma**2).exp()

        return distances

    def forward(self, x):
        z = self.last_layer(self.compute_features(x))
        y_pred = self.output_layer(z)
        self.logit = (y_pred+1e-20).log()
        return y_pred

class Gaussian_Classifier(Model):
    def __init__(
        self,
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
        device='cuda'
    ):
        super().__init__()

        self.feature_dimension = embedding_size
        self.device = device
        self.num_classes = num_classes

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, 256), 0.05)
        )

        self.register_buffer("N", torch.ones(num_classes) * 12)
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)

        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.feature_dimension))
        self.log_covs = nn.Parameter(torch.zeros(self.num_classes, self.feature_dimension))

        self.fc_last_layer = nn.Linear(256, 256)

        self.alpha = 0.3

    def update_embeddings(self, x, y):
        z = self.last_layer(self.compute_features(x))

        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * features_sum

    def last_layer_single(self, z):
        z = self.fc_last_layer(z)
        return z

    def last_layer(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)
        return z

    def output_layer(self, z, index = None):
        covs = torch.exp(self.log_covs)
        tcovs = covs.transpose(1,0).repeat(z.shape[0], 1, 1)

        embeddings = self.centers.transpose(1,0)

        diff = z - embeddings.unsqueeze(0)
        wdiff = torch.div(diff, tcovs)
        wdiff_square = torch.mul(diff, wdiff)
        distances = torch.mean(wdiff_square, dim=1)
        slog_covs = torch.mean(self.log_covs, dim=1)
        tslog_covs = slog_covs.repeat(z.shape[0], 1)
        if index is not None:
            distances[index[:, 0], index[:, 1]] = distances.clone()[index[:, 0], index[:, 1]] * (1 + self.alpha)

            wdiff_true_class = wdiff.clone()[index[:, 0], :, index[:, 1]]
            diff_true_class = diff.clone()[index[:, 0], :, index[:, 1]]
            wdiff_true_class_square = torch.mul(diff_true_class, wdiff_true_class).mean(-1).sum(0) / 2.0
            reg = 0.5 * torch.sum(tslog_covs[index[:, 0], index[:, 1]])
            self.lki_loss = 0.1 / z.shape[0] * (wdiff_true_class_square + reg)
        # distances=distances.exp()

        margin_logits = -0.5 * (tslog_covs + distances)
        self.logit = margin_logits

        return F.softmax(margin_logits, -1) #margin_logits.exp()#

    def output_layer_single(self, z, index = None):
        covs = torch.exp(self.log_covs)
        tcovs = covs.transpose(1,0).repeat(z.shape[0], 1, 1)

        embeddings = self.centers.transpose(1,0)

        diff = z - embeddings.repeat([z.shape[0],1,1])

        wdiff = torch.div(diff, tcovs)
        wdiff_power = torch.mul(diff, wdiff)
        distances = torch.mean(wdiff_power, dim=1)#.div(2 * self.sigma**2)
        slog_covs = torch.sum(self.log_covs, dim=1)
        tslog_covs = slog_covs.repeat(z.shape[0], 1)
        if index is not None:
            distances[index[:, 0], index[:, 1]] = distances.clone()[index[:, 0], index[:, 1]] * (1 + self.alpha)

            diff_true_class = diff.clone()[index[:, 0], :, index[:, 1]]
            diff_true_class = (diff_true_class ** 2).mean(-1).sum(0) / 2.0
            reg = 0.5 * torch.sum(tslog_covs[index[:, 0], index[:, 1]])
            self.lki_loss = 0.1 / z.shape[0] * (diff_true_class + reg)
        # distances=distances.exp()

        margin_logits = -0.5 * (tslog_covs + distances)
        self.logit = margin_logits

        return F.softmax(margin_logits, -1) #margin_logits.exp()#

    def forward(self, x, y=None):
        z = self.last_layer(self.compute_features(x))
        index=None
        if y is not None:
            target = torch.where(y != 0)[1]
            index = torch.cat([torch.arange(0, z.shape[0])[..., None].to(self.device), target[..., None]], dim=-1)
        y_pred = self.output_layer(z, index=index)
        return y_pred

class SNN(Model):
    def __init__(
        self,
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
        noise_dimension=512,
        hidden_dimension=512,
        feature_dimension=256,
        num_centroids = 32,
        k=10,
        centroid_reg = 'entropy',
        var_threshold = 0.1,
        reg_term_weight = 1.0,
        distance='kde',
        device = 'cuda'

    ):
        super().__init__()

        self.noise_dimension = noise_dimension
        self.hidden_dimension = hidden_dimension
        self.feature_dimension = feature_dimension
        self.num_centroids = num_centroids
        self.device = device
        self.num_classes = num_classes
        self.W = nn.Parameter(
            torch.normal(torch.zeros(embedding_size, num_classes, 256), 0.05)
        )
        self.layer_W = nn.Linear(self.feature_dimension, self.feature_dimension*self.num_classes, bias=False)
        nn.init.normal_(self.layer_W.weight, mean=0.0, std=0.05)
        # nn.init.constant_(self.layer_W.bias, 0.0)
        self.register_buffer("N", torch.ones(num_classes) * 12)
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)

        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale

        self.fc1_p = nn.Linear(self.num_classes + self.noise_dimension, self.hidden_dimension)
        self.fc2_p = nn.Linear(self.hidden_dimension, self.feature_dimension)

        self.class_embeddings = torch.eye(self.num_classes).to(self.device)
        self.k = k
        self.centroid_reg = centroid_reg
        self.class_threshold = torch.nn.Parameter(torch.randn((10)))
        self.var_threshold = var_threshold
        self.reg_term_weight = reg_term_weight
        self.distance = distance
        self.alpha = 0.3

    def update_embeddings(self, x, y):
        # z = self.last_layer(self.compute_features(x))
        #
        # # normalizing value per class, assumes y is one_hot encoded
        # self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)
        #
        # # compute sum of embeddings on class by class basis
        # features_sum = torch.einsum("ijk,ik->jk", z, y)
        #
        # self.m = self.gamma * self.m + (1 - self.gamma) * features_sum
        pass

    def last_layer(self, z):
        # z = torch.einsum("ij,mnj->imn", z, self.W)
        z = self.layer_W(z)
        z = z.reshape(z.shape[0], self.num_classes, self.feature_dimension).transpose(1,2)#z.reshape(z.shape[0], self.feature_dimension, self.num_classes)#
        return z

    def output_layer(self, z, prototype, index = None):
        diff = z.transpose(1, 2)[:, :, None, :] - prototype[None, ...]
        distances = (-(diff**2)).mean([-1,-2]).div(2 * self.sigma**2)
        if index is not None:
            distances[index[:, 0], index[:, 1]] = distances.clone()[index[:, 0], index[:, 1]] * (1 + self.alpha)
        return distances

    def sampling_prototype(self, prototype, n_samples=16):
        n_class, n_prototypes = prototype.shape[0], prototype.shape[1]
        indices = torch.randint(n_prototypes, (n_class, n_samples)).to(self.device)
        indices = torch.stack(
            [torch.arange(n_class).reshape(n_class, 1).repeat([1, n_samples]).to(self.device), indices], dim=-1)
        prototype = prototype[indices[:, :, 0], indices[:, :, 1]]

        return prototype

    def classify(self, x_input, prototype=None, sampling=True, n_samples=32, method='kde', sigma=1.0, n_ensemble=1):

        if prototype is None:
            prototype = self.get_prototype(num_centroids=self.num_centroids)
        if sampling:
            prototype = self.sampling_prototype(prototype, n_ensemble * n_samples)
        if method == 'kde':
            x = x_input.clone()
            sigma = torch.ones_like(x).detach().clone()
            logit = kde(x, prototype, sigma=sigma, n_ensemble=n_ensemble)
        elif method == 'hausdorff':
            x = x_input.clone()
            sigma = torch.ones_like(x).detach().clone()
            logit = hausdorff_distance(x, prototype, sigma=sigma, n_ensemble=n_ensemble)
        elif method == 'softmax':
            x = x_input.clone()[:, :self.num_class]
            x_ = x[:,:,None].repeat([1,1,n_ensemble])
            sigma = torch.abs(x_input.clone()[:, self.num_class:])
            sigma_ = (sigma[:, :, None] * torch.randn([self.batch_size, self.num_class, n_ensemble]).to('cuda'))
            logit = x_ + sigma_
        elif method =='vanilla_softmax':
            x = x_input.clone()[:, :-1]
            logit = x
        else:
            raise ValueError('Please specify a method from {kde, hausdorff, softmax}.')
        return logit

    def get_prototype(self, num_centroids):

        prototype = torch.cat([torch.randn(self.num_classes * num_centroids, self.noise_dimension).to(self.device),
                               self.class_embeddings.repeat(1, num_centroids).view(-1, self.num_classes)], dim=1)
        prototype = F.relu(self.fc1_p(prototype))
        prototype = self.fc2_p(prototype)

        return prototype.view(self.num_classes, num_centroids, self.feature_dimension)

    def forward(self, x, y = None, separate=True):
        prototype = self.get_prototype(num_centroids=self.num_centroids)
        feature = self.compute_features(x)
        index = None
        if y is not None:
            target = torch.where(y != 0)[1]
            index = torch.cat([torch.arange(0, feature.shape[0])[..., None].to(self.device), target[..., None]], dim=-1)
        feature = self.last_layer(feature)
        logit = self.output_layer(feature, prototype, index)
        if y is not None:
            self.loss_proto = self.reg_term_weight * loss_proto(self, prototype)
            self.lki_loss = -0.1 * likelihood_loss(self, prototype, feature, target, distance=self.distance,
                                                    device=self.device, separate=separate)
        self.logit = logit
        y_pred = logit.exp()

        return y_pred

class SNN_BE(nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
        noise_dimension=512,
        hidden_dimension=512,
        feature_dimension=256,
        num_centroids = 32,
        k=10,
        centroid_reg = 'entropy',
        var_threshold = 0.1,
        reg_term_weight = 1.0,
        num_models=4,
        distance='kde',
        device = 'cuda',
        gp = 0.1
    ):
        super().__init__()

        self.noise_dimension = noise_dimension
        self.hidden_dimension = hidden_dimension
        self.feature_dimension = feature_dimension
        self.num_centroids = num_centroids
        self.device = device
        self.num_classes = num_classes
        self.num_models=num_models
        # self.W = nn.Parameter(
        #     torch.normal(torch.zeros(embedding_size, num_classes, 256), 0.05)
        # )
        self.layer_W = Ensemble_orderFC(self.feature_dimension, self.feature_dimension*self.num_classes,
                                        self.num_models, first_layer=False, bias=False, layer_W=True)
        self.register_buffer("N", torch.ones(num_classes) * 12)
        self.register_buffer(
            "m", torch.normal(torch.zeros(embedding_size, num_classes), 1)
        )

        self.m = self.m * self.N.unsqueeze(0)

        if learnable_length_scale:
            self.sigma = nn.Parameter(torch.zeros(num_classes) + length_scale)
        else:
            self.sigma = length_scale
        self.gp = gp

        self.fc1_p = Ensemble_orderFC(self.num_classes + self.noise_dimension, self.hidden_dimension, num_models = num_models)
        self.fc2_p = Ensemble_orderFC(self.hidden_dimension, self.feature_dimension, num_models = num_models)

        self.class_embeddings = torch.eye(self.num_classes).to(self.device)
        self.k = k
        self.centroid_reg = centroid_reg
        self.class_threshold = torch.randn((self.num_classes),requires_grad=False).to(device)
        self.var_threshold = var_threshold
        self.reg_term_weight = reg_term_weight
        self.distance = distance

        self.conv1 = Ensemble_Conv2d(1, 64, 3, padding=1, num_models=self.num_models)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = Ensemble_Conv2d(64, 128, 3, padding=1, num_models=self.num_models)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = Ensemble_Conv2d(128, 128, 3, num_models=self.num_models)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = Ensemble_orderFC(2 * 2 * 128, 256, num_models=self.num_models)

        self.alpha = 0.3

    def compute_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.flatten(1)

        x = F.relu(self.fc1(x))

        return x
    def update_embeddings(self, x, y):
        # z = self.last_layer(self.compute_features(x))
        #
        # # normalizing value per class, assumes y is one_hot encoded
        # self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)
        #
        # # compute sum of embeddings on class by class basis
        # features_sum = torch.einsum("ijk,ik->jk", z, y)
        #
        # self.m = self.gamma * self.m + (1 - self.gamma) * features_sum
        pass

    def last_layer(self, z):
        # z = torch.einsum("ij,mnj->imn", z, self.W)
        z = self.layer_W(z)
        z = z.reshape(z.shape[0], self.feature_dimension, self.num_classes)
        return z

    def output_layer(self, z, prototype, index = None):
        batch_size = z.shape[0] // self.num_models
        prototype_repeat = prototype[:,None, ...].repeat(1, batch_size, 1, 1, 1)
        diff = z.transpose(1, 2).reshape([self.num_models,batch_size,self.num_classes,1,self.feature_dimension])- prototype_repeat
        distances = (-(diff**2)).mean([-1,-2]).div(2 * self.sigma**2)
        if index is not None:
            distances[:, index[:batch_size, 0], index[:batch_size, 1]] = distances.clone()[:, index[:batch_size, 0], index[:batch_size, 1]] * (1 + self.alpha)
        return distances

    def sampling_prototype(self, prototype, n_samples=16):
        n_class, n_prototypes = prototype.shape[0], prototype.shape[1]
        indices = torch.randint(n_prototypes, (n_class, n_samples)).to(self.device)
        indices = torch.stack(
            [torch.arange(n_class).reshape(n_class, 1).repeat([1, n_samples]).to(self.device), indices], dim=-1)
        prototype = prototype[indices[:, :, 0], indices[:, :, 1]]

        return prototype

    def classify(self, x_input, prototype=None, sampling=False, n_samples=32, method='kde', sigma=1.0, n_ensemble=1):

        if prototype is None:
            prototype = self.get_prototype(num_centroids=self.num_centroids)
        if sampling:
            prototype = self.sampling_prototype(prototype, n_ensemble * n_samples)
        if method == 'kde':
            x = x_input.clone()
            sigma = torch.ones_like(x).detach().clone()
            logit = kde(x, prototype, sigma=sigma, n_ensemble=n_ensemble)
        elif method == 'hausdorff':
            x = x_input.clone()
            sigma = torch.ones_like(x).detach().clone()
            logit = hausdorff_distance(x, prototype, sigma=sigma, n_ensemble=n_ensemble)
        elif method == 'softmax':
            x = x_input.clone()[:, :self.num_class]
            x_ = x[:,:,None].repeat([1,1,n_ensemble])
            sigma = torch.abs(x_input.clone()[:, self.num_class:])
            sigma_ = (sigma[:, :, None] * torch.randn([self.batch_size, self.num_class, n_ensemble]).to('cuda'))
            logit = x_ + sigma_
        elif method =='vanilla_softmax':
            x = x_input.clone()[:, :-1]
            logit = x
        else:
            raise ValueError('Please specify a method from {kde, hausdorff, softmax}.')
        return logit

    def get_prototype(self, num_centroids):

        prototype = torch.cat([torch.randn(self.num_classes * num_centroids, self.noise_dimension).to(self.device),
                               self.class_embeddings.repeat(1, num_centroids).view(-1, self.num_classes)], dim=1)
        prototype = prototype.repeat(self.num_models, 1)
        prototype = F.relu(self.fc1_p(prototype))
        prototype = self.fc2_p(prototype)

        return prototype.view(self.num_models, self.num_classes, num_centroids, self.feature_dimension)

    def forward(self, x, y = None, separate=True):
        prototype = self.get_prototype(num_centroids=self.num_centroids)
        feature = self.compute_features(x)
        index = None
        if y is not None:
            target = torch.where(y != 0)[1]
            index = torch.cat([torch.arange(0, feature.shape[0])[..., None].to(self.device), target[..., None]], dim=-1)
        feature = self.last_layer(feature)
        logit = self.output_layer(feature, prototype, index)
        if y is not None:
            for m in range(self.num_models):
                for c in range(self.num_classes):
                    feature_class = feature[torch.where(target == c)[0]]
                    if feature_class.shape[0] > 1:
                        self.class_threshold[c] = \
                        torch.cdist(feature[torch.where(target == c)[0]][:, :, c].contiguous(),
                                    feature[torch.where(target == c)[0]][:, :,
                                    c].contiguous()).sort(
                            dim=-1)[0][:, 1:self.k + 1].mean().detach()
                    else:
                        self.class_threshold[c]=1.
                self.loss_proto = self.reg_term_weight * loss_proto_be(self, prototype[m:m+1])

        self.logit = logit.reshape([-1, self.num_classes])
        y_pred = self.logit.exp()

        return y_pred

class Model_softmax(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 256, 3)
        self.bn3 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(2 * 2 * 256, 256)

    def compute_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2, 2)

        x = x.flatten(1)

        x = F.relu(self.fc1(x))

        return x

class SoftmaxModel(Model_softmax):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.last_layer = nn.Linear(256, num_classes)
        self.output_layer = nn.LogSoftmax(dim=1)

    def forward(self, x):
        z = self.last_layer(self.compute_features(x))
        y_pred = F.softmax(z, dim=1)

        return y_pred
