import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.batchensemble_layers import *
from utils.utils import kde, hausdorff_distance, dataset_loader, plot_kernel_distance, plot_ece, plot_bar, \
    plot_entropy, accuracy_confidence_data, likelihood_loss, plot_distance, plot_mutual_information, plot_exp_distance, loss_proto, loss_proto_be, likelihood_loss_be

class ResNet_DUQ(nn.Module):
    def __init__(
        self,
        feature_extractor,
        num_classes,
        centroid_size,
        model_output_size,
        length_scale,
        gamma,
    ):
        super().__init__()

        self.gamma = gamma

        self.W = nn.Parameter(
            torch.zeros(centroid_size, num_classes, model_output_size)
        )
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

        self.feature_extractor = feature_extractor

        self.register_buffer("N", torch.zeros(num_classes) + 13)
        self.register_buffer(
            "m", torch.normal(torch.zeros(centroid_size, num_classes), 0.05)
        )
        self.m = self.m * self.N

        self.sigma = length_scale

    def rbf(self, z):
        z = torch.einsum("ij,mnj->imn", z, self.W)

        embeddings = self.m / self.N.unsqueeze(0)

        diff = z - embeddings.unsqueeze(0)
        diff = (diff ** 2).mean(1).div(2 * self.sigma ** 2).mul(-1).exp()

        self.logit = diff.log()+1e-20
        return diff

    def update_embeddings(self, x, y):
        self.N = self.gamma * self.N + (1 - self.gamma) * y.sum(0)

        z = self.feature_extractor(x)

        z = torch.einsum("ij,mnj->imn", z, self.W)
        embedding_sum = torch.einsum("ijk,ik->jk", z, y)

        self.m = self.gamma * self.m + (1 - self.gamma) * embedding_sum

    def forward(self, x):
        z = self.feature_extractor(x)
        y_pred = self.rbf(z)

        return y_pred


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Ensemble_Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Ensemble_Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Ensemble_Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Ensemble_Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Ensemble_Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Ensemble_Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                Ensemble_Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_BE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_BE, self).__init__()
        self.in_planes = 64

        self.conv1 = Ensemble_Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = Ensemble_orderFC(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class SNN_BE_ResNet(nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_size,
        learnable_length_scale,
        length_scale,
        gamma,
        feature_extractor,
        noise_dimension=512,
        hidden_dimension=512,
        feature_dimension=512,
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
        self.feature_extractor = feature_extractor
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

        self.alpha = 0.3

    def compute_features(self, x):
        x = self.feature_extractor(x)

        return x

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