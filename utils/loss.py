import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layer import *
from functools import reduce
import numpy as np

def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))

def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.
    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl

class IDMRFLoss(nn.Module):
    def __init__(self, featlayer=VGG19FeatLayer, device=0, shallow_feats=False):
        super(IDMRFLoss, self).__init__()
        self.featlayer = featlayer(device=device)
        if shallow_feats:
            self.feat_style_layers = {'relu2_2': 1.0, 'relu3_2': 1.0}
            self.feat_content_layers = {'relu3_2': 1.0}
        else:
            self.feat_style_layers = {'relu3_2': 1.0, 'relu4_2': 1.0}
            self.feat_content_layers = {'relu4_2': 1.0}
        self.bias = 1.0
        self.nn_stretch_sigma = 0.5
        self.lambda_style = 1.0
        self.lambda_content = 1.0

    def sum_normalize(self, featmaps):
        epsilon = 1e-6
        reduce_sum = torch.sum(featmaps, dim=1, keepdim=True)
        temp = featmaps / (reduce_sum + epsilon)
        return temp

    def patch_extraction(self, featmaps):
        patch_size = 1
        patch_stride = 1
        patches_as_depth_vectors = featmaps.unfold(2, patch_size, patch_stride).unfold(3, patch_size, patch_stride)
        self.patches_OIHW = patches_as_depth_vectors.permute(0, 2, 3, 1, 4, 5)
        dims = self.patches_OIHW.size()
        self.patches_OIHW = self.patches_OIHW.view(-1, dims[3], dims[4], dims[5])
        return self.patches_OIHW

    def compute_relative_distances(self, cdist):
        epsilon = 1e-3
        div = torch.min(cdist, dim=1, keepdim=True)[0]
        div[div<epsilon] = div[div<epsilon] + epsilon
        relative_dist = cdist / div
        return relative_dist

    def exp_norm_relative_dist(self, relative_dist):
        epsilon = 1e-5
        scaled_dist = relative_dist
        dist_before_norm = torch.exp((self.bias - scaled_dist)/(self.nn_stretch_sigma+ epsilon))
        self.cs_NCHW = self.sum_normalize(dist_before_norm)
        return self.cs_NCHW

    def mrf_loss(self, gen, tar):
        epsilon = 1e-5
        meanT = torch.mean(tar, 1, keepdim=True)
        gen_feats, tar_feats = gen - meanT, tar - meanT

        gen_feats_norm = torch.norm(gen_feats, p=2, dim=1, keepdim=True)
        tar_feats_norm = torch.norm(tar_feats, p=2, dim=1, keepdim=True)

        gen_normalized = gen_feats / (gen_feats_norm + epsilon)
        tar_normalized = tar_feats / (tar_feats_norm + epsilon)

        cosine_dist_l = []
        BatchSize = tar.size(0)

        for i in range(BatchSize):
            tar_feat_i = tar_normalized[i:i+1, :, :, :]
            gen_feat_i = gen_normalized[i:i+1, :, :, :]
            patches_OIHW = self.patch_extraction(tar_feat_i)

            cosine_dist_i = F.conv2d(gen_feat_i, patches_OIHW)
            cosine_dist_l.append(cosine_dist_i)
        cosine_dist = torch.cat(cosine_dist_l, dim=0)
        cosine_dist_zero_2_one = - (cosine_dist - 1) / 2
        relative_dist = self.compute_relative_distances(cosine_dist_zero_2_one)
        rela_dist = self.exp_norm_relative_dist(relative_dist)
        dims_div_mrf = rela_dist.size()
        k_max_nc = torch.max(rela_dist.view(dims_div_mrf[0], dims_div_mrf[1], -1), dim=2)[0]
        div_mrf = torch.mean(k_max_nc, dim=1)
        div_mrf_sum = -torch.log(div_mrf)
        div_mrf_sum = torch.sum(div_mrf_sum)
        return div_mrf_sum

    def forward(self, gen, tar):
        gen_vgg_feats = self.featlayer(gen)
        tar_vgg_feats = self.featlayer(tar)

        style_loss_list = [self.feat_style_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_style_layers]
        self.style_loss = reduce(lambda x, y: x+y, style_loss_list) * self.lambda_style

        content_loss_list = [self.feat_content_layers[layer] * self.mrf_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.feat_content_layers]
        self.content_loss = reduce(lambda x, y: x+y, content_loss_list) * self.lambda_content

        return self.style_loss + self.content_loss

class ContentLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0], content_total_weight=0.1):
        super(ContentLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.weights = weights
        self.content_total_weight = content_total_weight

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss * self.content_total_weight

class StyleLoss(nn.Module):
    def __init__(self, style_total_weight=100.0):
        super(StyleLoss, self).__init__()
        self.add_module('vgg', VGG19())
        self.criterion = torch.nn.L1Loss()
        self.style_total_weight = style_total_weight

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * ch)
        return G

    def __call__(self, x, y):
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss * self.style_total_weight

# # 24.10.25 Feature Similarity Loss (MSOI loss)
# class FS_Loss(nn.Module):
#     def __init__(self, featlayer=VGG19FeatLayer, device=0):
#         super(FS_Loss, self).__init__()
#         self.featlayer = featlayer(device=device)
#         self.L2_loss = nn.MSELoss()
#
#         self.multi_scale_layers = {'relu2_2': 1.0, 'relu3_2': 1.0, 'relu4_2': 1.0, 'relu5_2': 1.0}  # 후반 layer 일수록 고차원적이고 구조적인 정보 제공한다는 것 인지
#         self.FS_loss_weight = 1.0
#
#     def forward(self, gen, tar):
#         gen_vgg_feats = self.featlayer(gen)
#         tar_vgg_feats = self.featlayer(tar)
#
#         FS_loss_list = [self.multi_scale_layers[layer] * self.L2_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) for layer in self.multi_scale_layers]
#         # self.FS_loss = (reduce(lambda x, y: x+y, FS_loss_list) / len(self.multi_scale_layers)) * self.FS_loss_weight
#         self.FS_loss = reduce(lambda x, y: x + y, FS_loss_list) * self.FS_loss_weight
#
#         return self.FS_loss

# class Perceptual_loss(nn.Module):
#     def __init__(self, featlayer=VGG16FeatLayer, device=0, style_weight=1.0, content_weight=1.0):
#         super(Perceptual_loss, self).__init__()
#         self.featlayer = featlayer(device=device)
#         # self.L2_loss = nn.MSELoss()
#         self.L1_loss = nn.L1Loss()
#         self.gram_matrix = GramMatrix()
#
#         self.style_loss_layers = {'relu1_2': 1.0, 'relu2_2': 1.0, 'relu3_3': 1.0, 'relu4_3': 1.0}  # 후반 layer 일수록 고차원적이고 구조적인 정보 제공한다는 것 인지
#         self.content_loss_layers = {'relu3_3': 1.0}
#
#         self.style_loss_weight = style_weight
#         self.content_loss_weight = content_weight
#
#     def forward(self, gen, tar):
#         gen_vgg_feats = self.featlayer(gen)
#         tar_vgg_feats = self.featlayer(tar)
#
#         style_loss_list = [
#             self.L1_loss(self.gram_matrix(gen_vgg_feats[layer]), self.gram_matrix(tar_vgg_feats[layer])) * self.style_loss_layers[layer] for layer in self.style_loss_layers
#         ]
#
#         self.style_loss = sum(style_loss_list) * self.style_loss_weight
#
#         content_loss_list = [
#             self.L1_loss(gen_vgg_feats[layer], tar_vgg_feats[layer]) * self.content_loss_layers[layer] for layer in self.content_loss_layers
#         ]
#
#         self.content_loss = sum(content_loss_list) * self.content_loss_weight
#
#         self.Perceptual_loss = self.style_loss + self.content_loss
#
#         return self.Perceptual_loss
#
# class GramMatrix(nn.Module):
#     def forward(self, input):
#         b, c, h, w = input.size()
#         F = input.view(b, c, h * w)
#         G = torch.bmm(F, F.transpose(1, 2))
#         G.div_(h * w * c)
#         return G