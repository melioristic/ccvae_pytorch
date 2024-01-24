import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchvision.utils import make_grid, save_image
import torch.distributions as dist
import os
from .networks import (Classifier, CondPrior,
                       MLPDecoder, MLPEncoder)


def compute_kl(locs_q, scale_q, locs_p=None, scale_p=None):
    """
    Computes the KL(q||p)
    """
    if locs_p is None:
        locs_p = torch.zeros_like(locs_q)
    if scale_p is None:
        scale_p = torch.ones_like(scale_q)

    dist_q = dist.Normal(locs_q, scale_q)
    dist_p = dist.Normal(locs_p, scale_p)
    return dist.kl.kl_divergence(dist_q, dist_p).sum(dim=-1)

def img_log_likelihood(recon, xs):

    return dist.Laplace(recon, torch.ones_like(recon)).log_prob(xs).sum(dim=(1,2))

class CCVAE(nn.Module):
    """
    CCVAE
    """
    def __init__(self, z_dim, num_classes,
                 in_shape, prior_fn):
        super(CCVAE, self).__init__()
        self.z_dim = z_dim
        self.z_classify = num_classes
        self.z_style = z_dim - num_classes
        self.im_shape = in_shape
        self.num_classes = num_classes
        self.ones = torch.ones(1, self.z_style)
        self.zeros = torch.zeros(1, self.z_style)
        self.y_prior_params = prior_fn()

        self.encoder = MLPEncoder(in_shape[1], self.z_dim)
        self.decoder = MLPDecoder(in_shape[1], self.z_dim)
        self.classifier = Classifier(self.num_classes)
        self.cond_prior = CondPrior(self.num_classes)

    def sup(self, x, y):
        bs = x.shape[0]
        #inference
        post_params = self.encoder(x)
        z = dist.Normal(*post_params).rsample()

        zc, zs = z.split([self.z_classify, self.z_style], 1)
        qyzc = dist.Bernoulli(logits=self.classifier(zc))
        log_qyzc = qyzc.log_prob(y).sum(dim=-1)

        # compute kl
        locs_p_zc, scales_p_zc = self.cond_prior(y)
        prior_params = (torch.cat([locs_p_zc, self.zeros.expand(bs, -1)], dim=1), 
                        torch.cat([scales_p_zc, self.ones.expand(bs, -1)], dim=1))
        #prior_params = (self.zeros.expand(bs, -1), self.ones.expand(bs, -1))
        kl = compute_kl(*post_params, *prior_params)

        #compute log probs for x and y
        recon = self.decoder(z)
        log_py = dist.Bernoulli(self.y_prior_params.expand(bs, -1)).log_prob(y).sum(dim=-1)
        log_qyx = self.classifier_loss(x, y)
        log_pxz = img_log_likelihood(recon, x)

        # we only want gradients wrt to params of qyz, so stop them propogating to qzx
        log_qyzc_ = dist.Bernoulli(logits=self.classifier(zc.detach())).log_prob(y).sum(dim=-1)
        w = torch.exp(log_qyzc_ - log_qyx)
        elbo = (w * (log_pxz - kl - log_qyzc) + log_py + log_qyx).mean()
        return -elbo, log_qyzc.mean(), log_qyzc_.mean(), log_qyx.mean(), log_pxz.mean(), kl.mean(), log_py.mean()

    def classifier_loss(self, x, y, k=100):
        """
        Computes the classifier loss.
        """
        zc, _ = dist.Normal(*self.encoder(x)).rsample(torch.tensor([k])).split([self.z_classify, self.z_style], -1)
        logits = self.classifier(zc.view(-1, self.z_classify))
        d = dist.Bernoulli(logits=logits)
        y = y.expand(k, -1, -1).contiguous().view(-1, self.num_classes)
        lqy_z = d.log_prob(y).view(k, x.shape[0], self.num_classes).sum(dim=-1)
        lqy_x = torch.logsumexp(lqy_z, dim=0) - np.log(k)
        return lqy_x

    def reconstruct_img(self, x):
        return self.decoder(dist.Normal(*self.encoder(x)).rsample())

    def classifier_acc(self, x, y=None, k=1):
        preds = self.classifier_pred(x, y=None, k=1)
        y = y.expand(k, -1, -1).contiguous().view(-1, self.num_classes)
        acc = (preds.eq(y)).float().mean()
        return acc
    
    def classifier_pred(self, x, k=1, prob=False):
        zc, _ = dist.Normal(*self.encoder(x)).rsample(torch.tensor([k])).split([self.z_classify, self.z_style], -1)
        logits = self.classifier(zc.view(-1, self.z_classify)).view(-1, self.num_classes)
        
        preds = torch.sigmoid(logits) if prob else torch.round(torch.sigmoid(logits))

        return preds

    def accuracy(self, data_loader, *args, **kwargs):
        acc = 0.0
        for (x, y) in data_loader:
            batch_acc = self.classifier_acc(x, y)
            acc += batch_acc
        return acc / len(data_loader)

    def get_latent(self, x):
        loc, scale = self.encoder(x)    
        return loc, scale
    
    def latent_walk(self, x):
        """
        Does latent walk between all possible classes
        """
        mult = 5
        num_imgs = 7
        z_ = dist.Normal(*self.encoder(x.unsqueeze(0))).sample()
        imgs_1_list_1 = []
        for i in range(self.num_classes): # 4
            y_1 = torch.zeros(1, self.num_classes) # y_1 [0, 0, 0, 0]
            locs_false, scales_false = self.cond_prior(y_1)
            y_1[:, i].fill_(1.0)
            locs_true, scales_true = self.cond_prior(y_1)
            sign = torch.sign(locs_true[:, i] - locs_false[:, i])
            # y axis
            z_1_false_lim = (locs_false[:, i] - mult * sign * scales_false[:, i]).item()    
            z_1_true_lim = (locs_true[:, i] + mult * sign * scales_true[:, i]).item()   
            
            imgs_1_list_2 = []
            for j in range(self.num_classes):
                z = z_.clone()
                z = z.expand(num_imgs**2, -1).contiguous()
                if i == j:
                    continue
                y_2 = torch.zeros(1, self.num_classes)

                locs_false, scales_false = self.cond_prior(y_2)
                y_2[:, i].fill_(1.0)
                locs_true, scales_true = self.cond_prior(y_2)
                sign = torch.sign(locs_true[:, i] - locs_false[:, i])
                # x axis
                z_2_false_lim = (locs_false[:, i] - mult * sign * scales_false[:, i]).item()    
                z_2_true_lim = (locs_true[:, i] + mult * sign * scales_true[:, i]).item()

                # construct grid
                range_1 = torch.linspace(z_1_false_lim, z_1_true_lim, num_imgs)
                range_2 = torch.linspace(z_2_false_lim, z_2_true_lim, num_imgs)
                grid_1, grid_2 = torch.meshgrid(range_1, range_2)
                z[:, i] = grid_1.reshape(-1)
                z[:, j] = grid_2.reshape(-1)

                imgs_1_list_2.append(self.decoder(z).view(-1, *self.im_shape))  

            imgs_1_list_1.append(torch.stack(imgs_1_list_2))    

            # mult = 8
            # for j in range(self.num_classes):
            #     z = z_.clone()
            #     z = z.expand(10, -1).contiguous()
            #     y = torch.zeros(1, self.num_classes)
            #     locs_false, scales_false = self.cond_prior(y)
            #     y[:, i].fill_(1.0)
            #     locs_true, scales_true = self.cond_prior(y)
            #     sign = torch.sign(locs_true[:, i] - locs_false[:, i])
            #     z_false_lim = (locs_false[:, i] - mult * sign * scales_false[:, i]).item()    
            #     z_true_lim = (locs_true[:, i] + mult * sign * scales_true[:, i]).item()
            #     range_ = torch.linspace(z_false_lim, z_true_lim, 10)
            #     z[:, j] = range_

            #     imgs_2 = self.decoder(z).view(-1, *self.im_shape)
                

        return torch.stack(imgs_1_list_1)