import torch
import torch.nn as nn
import random
import math
#from torchvision.models import resnet50
from resnet import resnet50,resnet18
#from resnet_kl import resnet50
import clip
from sklearn.decomposition import PCA
import numpy as np


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


class MoCo_nnclr(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_nnclr, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, num_mixup):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            # nnclr 我的

            similarity_matrix = torch.einsum("nd,dm->nm", k, self.queue)
            index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
            # k_neighbours = torch.index_select(
            #        self.queue, dim=1, index=index_nearest_neighbours
            #    ).contiguous().t().detach()

            k_neighbours = torch.tensor([]).cuda()
            for i in range(1, num_mixup+1):
                _, index_nearest_neighbours = torch.topk(
                    similarity_matrix, k=i, dim=1)
                index_nearest_neighbours = index_nearest_neighbours[:, i-1]
                k_num_neighbours = torch.index_select(
                    self.queue, dim=1, index=index_nearest_neighbours).contiguous().t().detach().unsqueeze(1)

                k_neighbours = torch.cat(
                    (k_neighbours, k_num_neighbours), dim=1)
            k_neighbours = torch.mean(k_neighbours, dim=1).detach()
            # k_neighbours,_ = torch.max(k_neighbours,dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k_neighbours]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels



class MoCo_PCA(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_PCA, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(num_classes=dim)
        self.encoder_k = resnet50(num_classes=dim)
        
        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        #q = self.encoder_q.fc(q)
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            #k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        #PCA
        X_centered = self.queue.T - torch.mean(self.queue.T, axis=0)
        cov_matrix = torch.matmul(X_centered.T, X_centered) / (self.K - 1)
        # 3. 对协方差矩阵进行特征值分解
        # torch.linalg.eigh 用于对对称矩阵（如协方差矩阵）进行分解
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        # 特征值从小到大排序，需要反转顺序
        eigenvalues = eigenvalues.flip(0)  # 翻转特征值
        eigenvectors = eigenvectors.flip(1)  # 翻转特征向量
        # 最大的主成分对应于最大的特征值和对应的特征向量
        largest_eigenvalue = eigenvalues[0]  # 最大特征值
        largest_eigenvector = eigenvectors[:, 0]  # 最大特征值对应的特征向量
        positive_contributions = torch.where(largest_eigenvector > 0, largest_eigenvector, torch.zeros_like(largest_eigenvector))  # 正相关部分
        feature_weights = positive_contributions * largest_eigenvalue
        feature_importance_normalized = (feature_weights - torch.min(feature_weights)) / (
            torch.max(feature_weights) - torch.min(feature_weights)
        )

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q * feature_importance_normalized, k * feature_importance_normalized]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q * feature_importance_normalized, (self.queue.clone().detach().T * feature_importance_normalized).T])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


class MoCo_nnclr_signle(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_nnclr_signle, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        idx_shuffle = torch.randperm(x.shape[0]).cuda()    #将0~n-1（包括0和n-1）随机打乱后获得的数字序列

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)         #原始数字在打乱后的位置

        return x[idx_shuffle], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        return x[idx_unshuffle]

    def forward(self, im_q, im_k, num_mixup):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            # nnclr 我的

            similarity_matrix = torch.einsum("nd,dm->nm", k, self.queue)
            index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
            # k_neighbours = torch.index_select(
            #        self.queue, dim=1, index=index_nearest_neighbours
            #    ).contiguous().t().detach()

            k_neighbours = torch.tensor([]).cuda()
            for i in range(1, num_mixup+1):
                _, index_nearest_neighbours = torch.topk(
                    similarity_matrix, k=i, dim=1)
                index_nearest_neighbours = index_nearest_neighbours[:, i-1]
                k_num_neighbours = torch.index_select(
                    self.queue, dim=1, index=index_nearest_neighbours).contiguous().t().detach().unsqueeze(1)

                k_neighbours = torch.cat(
                    (k_neighbours, k_num_neighbours), dim=1)
            k_neighbours = torch.mean(k_neighbours, dim=1).detach()
            # k_neighbours,_ = torch.max(k_neighbours,dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k_neighbours]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels


class MoCo_three(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_three, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr2[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, is_first=True,is_second = True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second:
            self._dequeue_and_enqueue2(k)

        return logits, labels

class MoCo_three_onlyOneQuene(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_three_onlyOneQuene, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k,is_first = True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue(k)

        return logits, labels


class MoCo_three_nnclr(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_three_nnclr, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr2[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, is_first=True,is_second = True,is_nnclr = False,num_mixup = 5):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            if is_nnclr:
                similarity_matrix = torch.einsum("nd,dm->nm", k, self.queue2)
                index_nearest_neighbours = torch.argmax(similarity_matrix, dim=1)
                # k_neighbours = torch.index_select(
                #        self.queue, dim=1, index=index_nearest_neighbours
                #    ).contiguous().t().detach()

                k_neighbours = torch.tensor([]).cuda()
                for i in range(1, num_mixup+1):
                    _, index_nearest_neighbours = torch.topk(
                        similarity_matrix, k=i, dim=1)
                    index_nearest_neighbours = index_nearest_neighbours[:, i-1]
                    k_num_neighbours = torch.index_select(
                        self.queue2, dim=1, index=index_nearest_neighbours).contiguous().t().detach().unsqueeze(1)

                    k_neighbours = torch.cat(
                        (k_neighbours, k_num_neighbours), dim=1)
                k_neighbours = torch.mean(k_neighbours, dim=1).detach()


        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        if is_nnclr:
            l_pos = torch.einsum("nc,nc->n", [q, k_neighbours]).unsqueeze(-1)
        else:
            l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)


        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second and is_nnclr == False:
            self._dequeue_and_enqueue2(k)

        return logits, labels


#在对比学习的投影头后，计算学生网络特征与CLIP文本之间的相似度，使该相似度尽可能接近教师模型，利用交叉熵或者KL散度
class MoCo_my_KL(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, teacher_dim = 1024,mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_my_KL, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.encoder_k = resnet50(num_classes=dim,teacher_dim = teacher_dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_cnn = self.encoder_q(im_q)  # queries: NxC
        #q_feature_like_teacher = self.encoder_q.distill(q_cnn)
        #q_feature_like_teacher = nn.functional.normalize(q_feature_like_teacher, dim = 1)   #, p=2.0
        q = self.encoder_q.fc(q_cnn)
        q = nn.functional.normalize(q, dim=1, p=2.0)
        #q_cnn = nn.functional.normalize(q_cnn, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1, p=2.0)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels,q#, q_feature_like_teacher


class MoCo_gradcam_KL(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, teacher_dim = 1024,mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_gradcam_KL, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.encoder_k = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.use_ddp = torch.distributed.is_initialized()

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.bilinear = 32
        self.conv16 = nn.Conv2d(2048, self.bilinear, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(7, stride=1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr2[0] = ptr

    def max_mask(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)  # 改为非就地操作

        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        return att_max

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, is_first=True,is_second = True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_cnn,featmap = self.encoder_q(im_q)  # queries: NxC
        #q_feature_like_teacher = self.encoder_q.distill(q_cnn)
        #q_feature_like_teacher = nn.functional.normalize(q_feature_like_teacher, dim = 1)   #, p=2.0
        q = self.encoder_q.fc(q_cnn)
        q = nn.functional.normalize(q, dim=1, p=2.0)
        #q_cnn = nn.functional.normalize(q_cnn, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k,_ = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1, p=2.0)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second:
            self._dequeue_and_enqueue2(k)

        return logits, labels, q, featmap
    
class MoCo_gradcam_KL_rs18(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, teacher_dim = 1024,mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_gradcam_KL_rs18, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet18(num_classes=dim,teacher_dim = teacher_dim)
        self.encoder_k = resnet18(num_classes=dim,teacher_dim = teacher_dim)
        self.use_ddp = torch.distributed.is_initialized()
        self.backbone_out_channels = self._get_backbone_out_channels(self.encoder_q)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.bilinear = 32
        self.conv16 = nn.Conv2d(self.backbone_out_channels, self.bilinear, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _get_backbone_out_channels(self, encoder):
        """
        从 encoder.layer4 的最后一个 block 推断输出通道数。
        兼容 ResNet-18/34 的 BasicBlock（bn2）和 ResNet-50/101 的 Bottleneck（bn3）。
        """
        last_block = encoder.layer4[-1]
        if hasattr(last_block, "bn3"):   # Bottleneck
            return last_block.bn3.num_features
        if hasattr(last_block, "bn2"):   # BasicBlock
            return last_block.bn2.num_features
        # 兜底：跑一次前向推断
        with torch.no_grad():
            x = torch.zeros(1, 3, 224, 224)
            x = encoder.relu(encoder.bn1(encoder.conv1(x)))
            x = encoder.maxpool(x)
            x = encoder.layer1(x); x = encoder.layer2(x)
            x = encoder.layer3(x); x = encoder.layer4(x)
            return x.shape[1]

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr2[0] = ptr

    def max_mask(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)  # 改为非就地操作

        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        return att_max

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, is_first=True,is_second = True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_cnn,featmap = self.encoder_q(im_q)  # queries: NxC
        #q_feature_like_teacher = self.encoder_q.distill(q_cnn)
        #q_feature_like_teacher = nn.functional.normalize(q_feature_like_teacher, dim = 1)   #, p=2.0
        q = self.encoder_q.fc(q_cnn)
        q = nn.functional.normalize(q, dim=1, p=2.0)
        #q_cnn = nn.functional.normalize(q_cnn, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k,_ = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1, p=2.0)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second:
            self._dequeue_and_enqueue2(k)

        return logits, labels, q, featmap


class MoCo_three_gradcam(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_three_gradcam, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(num_classes=dim)
        self.encoder_k = resnet50(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.bilinear = 32
        self.conv16 = nn.Conv2d(2048, self.bilinear, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr2[0] = ptr

    def max_mask(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)

        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        return att_max

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, is_first=True,is_second = True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q,featmap = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        #att_max = self.max_mask(featmap)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k,_ = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second:
            self._dequeue_and_enqueue2(k)

        return logits, labels, featmap


    def inference(self, img):
        projfeat, featmap = self.encoder_q(img)

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)


        img = featcov16.cpu().detach().numpy()
        img = np.max(img, axis=1)
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        img = torch.from_numpy(img)



        img = img[:, None, :, :]
        img = img.repeat(1, 2048, 1, 1)
        PFM = featmap.cuda() * img.cuda()
        aa = self.avgpool(PFM)
        bp_out_feat = aa.view(aa.size(0), -1)
        bp_out_feat = nn.functional.normalize(bp_out_feat, dim=1)

        return bp_out_feat
    
    def inference_cam(self, img):
        projfeat, featmap = self.encoder_q(img)

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)


        img = featcov16.cpu().detach().numpy()
        img = np.max(img, axis=1)
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
  
        return img

import torch
import torch.nn as nn
import torch.nn.functional as F

class MoCo_three_gradcam_rs18(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_three_gradcam_rs18, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet18(num_classes=dim)
        self.encoder_k = resnet18(num_classes=dim)

        # === 获取主干输出通道 feat_dim（先尝试从 fc 推断，失败则跑一次前向取 featmap 通道） ===
        self.feat_dim = None
        try:
            if isinstance(self.encoder_q.fc, nn.Linear):
                self.feat_dim = int(self.encoder_q.fc.in_features)
            elif isinstance(self.encoder_q.fc, nn.Sequential):
                lin = None
                for m in self.encoder_q.fc.modules():
                    if isinstance(m, nn.Linear):
                        lin = m
                        break
                if lin is not None:
                    self.feat_dim = int(lin.in_features)
        except Exception:
            pass
        if self.feat_dim is None:
            with torch.no_grad():
                _x = torch.zeros(1, 3, 224, 224)
                _, _featmap = self.encoder_q(_x)
                self.feat_dim = int(_featmap.shape[1])

        # （可选）MLP：用 feat_dim 作为输入，不再访问 self.encoder_q.fc.weight.shape
        if mlp:
            in_dim = self.feat_dim
            proj_head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, dim),
            )
            self.encoder_q.fc = proj_head
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim, dim),
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue2
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.bilinear = 32
        # 2048 -> self.feat_dim（自适配 rs18=512/rs50=2048）
        self.conv16 = nn.Conv2d(self.feat_dim, self.bilinear, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU(inplace=True)
        # 固定 7x7 -> 自适应 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr2[0] = ptr

    def max_mask(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)
        img = torch.amax(featcov16, dim=1)  # [N,H,W]
        img = img - img.amin(dim=(1, 2), keepdim=True)
        att_max = img / (1e-7 + img.amax(dim=(1, 2), keepdim=True))
        return att_max

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        idx_shuffle = torch.randperm(batch_size_all).cuda()
        torch.distributed.broadcast(idx_shuffle, src=0)
        idx_unshuffle = torch.argsort(idx_shuffle)
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]
        num_gpus = batch_size_all // batch_size_this
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]
        return x_gather[idx_this]

    def forward(self, im_q, im_k, is_first=True, is_second=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q, featmap = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k, _ = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second:
            self._dequeue_and_enqueue2(k)

        return logits, labels, featmap

    def inference(self, img):
        projfeat, featmap = self.encoder_q(img)

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)

        # torch-only CAM mask (per-sample normalization)
        att = torch.amax(featcov16, dim=1)                                # [N,H,W]
        att = att - att.amin(dim=(1, 2), keepdim=True)
        att = att / (1e-7 + att.amax(dim=(1, 2), keepdim=True))

        att = att.unsqueeze(1).expand(-1, self.feat_dim, -1, -1)          # [N,C,H,W]
        PFM = featmap * att
        aa = self.avgpool(PFM)                                            # [N,C,1,1]
        bp_out_feat = aa.view(aa.size(0), -1)                             # [N,C]
        bp_out_feat = nn.functional.normalize(bp_out_feat, dim=1)

        return bp_out_feat

    def inference_cam(self, img):
        projfeat, featmap = self.encoder_q(img)

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)

        # keep numpy output but compute in torch with per-sample normalization
        att = torch.amax(featcov16, dim=1)                                # [N,H,W]
        att = att - att.amin(dim=(1, 2), keepdim=True)
        att = att / (1e-7 + att.amax(dim=(1, 2), keepdim=True))
        img = att.detach().cpu().numpy()

        return img


class MoCo_three_gradcam_different(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_three_gradcam_different, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(num_classes=dim)
        self.encoder_k = resnet50(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.bilinear = 32
        self.conv16 = nn.Conv2d(2048, self.bilinear, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.conv16_zhengti = nn.Conv2d(2048, self.bilinear, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn16_zhengti = nn.BatchNorm2d(self.bilinear)
        self.relu_zhengti = nn.ReLU(inplace=True)
        self.avgpool_zhengti = nn.AvgPool2d(7, stride=1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr2[0] = ptr

    def max_mask(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)

        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        return att_max
    
    def max_mask_zhengti(self, featmap):
        featcov16 = self.conv16_zhengti(featmap)
        featcov16 = self.bn16_zhengti(featcov16)
        featcov16 = self.relu_zhengti(featcov16)

        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        return att_max

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, is_first=True,is_second = True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q,featmap = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        #att_max = self.max_mask(featmap)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k,_ = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second:
            self._dequeue_and_enqueue2(k)

        return logits, labels, featmap


    def inference(self, img):
        projfeat, featmap = self.encoder_q(img)

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)


        img = featcov16.cpu().detach().numpy()
        img = np.max(img, axis=1)
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        img = torch.from_numpy(img)



        img = img[:, None, :, :]
        img = img.repeat(1, 2048, 1, 1)
        PFM = featmap.cuda() * img.cuda()
        aa = self.avgpool(PFM)
        bp_out_feat = aa.view(aa.size(0), -1)
        bp_out_feat = nn.functional.normalize(bp_out_feat, dim=1)

        return bp_out_feat

    def inference1(self, img):
        projfeat, featmap = self.encoder_q(img)

        featcov16 = self.conv16_zhengti(featmap)
        featcov16 = self.bn16_zhengti(featcov16)
        featcov16 = self.relu_zhengti(featcov16)


        img = featcov16.cpu().detach().numpy()
        img = np.max(img, axis=1)
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        img = torch.from_numpy(img)



        img = img[:, None, :, :]
        img = img.repeat(1, 2048, 1, 1)
        PFM = featmap.cuda() * img.cuda()
        aa = self.avgpool_zhengti(PFM)
        bp_out_feat = aa.view(aa.size(0), -1)
        bp_out_feat = nn.functional.normalize(bp_out_feat, dim=1)

        return bp_out_feat
    
    def inference2(self, img):
        projfeat, featmap = self.encoder_q(img)  

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)


        img = featcov16.cpu().detach().numpy()
        img = np.max(img, axis=1)
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        img = torch.from_numpy(img)



        img = img[:, None, :, :]
        img = img.repeat(1, 2048, 1, 1)
        PFM = featmap.cuda() * img.cuda()
        aa = self.avgpool(PFM)
        bp_out_feat = aa.view(aa.size(0), -1)
        bp_out_feat = nn.functional.normalize(bp_out_feat, dim=1)

        return projfeat,bp_out_feat

class Classifier(nn.Module):
    def __init__(self, inputs, class_num):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(inputs, class_num)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier_layer(x)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def average_loss_across_gpus(loss):
    # 确保损失是一个张量
    loss_tensor = torch.tensor(loss, device='cuda')
    
    # 将所有 GPU 上的损失相加
    torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
    
    # 计算平均损失
    loss_tensor /= torch.distributed.get_world_size()
    
    return loss_tensor.item()



class MoCo_three_gradcam_onegpu(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_three_gradcam_onegpu, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(num_classes=dim)
        self.encoder_k = resnet50(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.bilinear = 32
        self.conv16 = nn.Conv2d(2048, self.bilinear, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr2[0] = ptr

    def max_mask(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)

        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        return att_max

    @torch.no_grad()
    def _batch_shuffle_single_gpu(self, x):
        """
        Batch shuffle for single GPU.
        Returns:
            shuffled_x: shuffled batch
            idx_unshuffle: indices to undo the shuffle
        """
        batch_size = x.shape[0]
        idx_shuffle = torch.randperm(batch_size).to(x.device)
        x_shuffled = x[idx_shuffle]
        idx_unshuffle = torch.argsort(idx_shuffle)
        return x_shuffled, idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
        """
        Undo batch shuffle for single GPU.
        """
        return x[idx_unshuffle]


    def forward(self, im_q, im_k, is_first=True,is_second = True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q,featmap = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        #att_max = self.max_mask(featmap)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)

            k,_ = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second:
            self._dequeue_and_enqueue2(k)

        return logits, labels, featmap


    def inference(self, img):
        projfeat, featmap = self.encoder_q(img)

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)


        img = featcov16.cpu().detach().numpy()
        img = np.max(img, axis=1)
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        img = torch.from_numpy(img)



        img = img[:, None, :, :]
        img = img.repeat(1, 2048, 1, 1)
        PFM = featmap.cuda() * img.cuda()
        aa = self.avgpool(PFM)
        bp_out_feat = aa.view(aa.size(0), -1)
        bp_out_feat = nn.functional.normalize(bp_out_feat, dim=1)

        return bp_out_feat
    
    def inference_cam(self, img):
        projfeat, featmap = self.encoder_q(img)

        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)


        img = featcov16.cpu().detach().numpy()
        img = np.max(img, axis=1)
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
  
        return img
    

class MoCo_my_KL(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, teacher_dim = 1024,mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_my_KL, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.encoder_k = resnet50(num_classes=dim,teacher_dim = teacher_dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_cnn = self.encoder_q(im_q)  # queries: NxC
        #q_feature_like_teacher = self.encoder_q.distill(q_cnn)
        #q_feature_like_teacher = nn.functional.normalize(q_feature_like_teacher, dim = 1)   #, p=2.0
        q = self.encoder_q.fc(q_cnn)
        q = nn.functional.normalize(q, dim=1, p=2.0)
        #q_cnn = nn.functional.normalize(q_cnn, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1, p=2.0)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels,q#, q_feature_like_teacher


class MoCo_gradcam_KL_2(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, teacher_dim = 1024,mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_gradcam_KL_2, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.encoder_k = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.use_ddp = torch.distributed.is_initialized()

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        if isinstance(self.encoder_q.fc, nn.Sequential):
            feat_dim = self.encoder_q.fc[0].in_features
        else:
            feat_dim = self.encoder_q.fc.in_features
        self.kd_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, teacher_dim),
        )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.bilinear = 32
        self.conv16 = nn.Conv2d(2048, self.bilinear, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(7, stride=1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr2[0] = ptr

    def max_mask(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)  # 改为非就地操作

        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        return att_max

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, is_first=True,is_second = True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_cnn,featmap = self.encoder_q(im_q)  # queries: NxC
        #q_feature_like_teacher = self.encoder_q.distill(q_cnn)
        #q_feature_like_teacher = nn.functional.normalize(q_feature_like_teacher, dim = 1)   #, p=2.0
        q = self.encoder_q.fc(q_cnn)
        q = nn.functional.normalize(q, dim=1, p=2.0)
        #q_cnn = nn.functional.normalize(q_cnn, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k,_ = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1, p=2.0)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second:
            self._dequeue_and_enqueue2(k)

        return logits, labels, q, featmap
    
    def forward_single_kd(self, x):
        """
        学生 KD 单图特征：encoder_q backbone -> kd_head（teacher_dim=512）
        """
        q_cnn, _ = self.encoder_q(x)     # q_cnn 是 GAP 后的向量（比如 2048）
        kd_feat = self.kd_head(q_cnn)    # [N, 512]
        return kd_feat
    

class MoCo_gradcam_KL_temp(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, T_cross=0.15, teacher_dim = 1024,mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_gradcam_KL_temp, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.T_cross = T_cross

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.encoder_k = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.use_ddp = torch.distributed.is_initialized()

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.bilinear = 32
        self.conv16 = nn.Conv2d(2048, self.bilinear, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(7, stride=1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr2[0] = ptr

    def max_mask(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)  # 改为非就地操作

        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        return att_max

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, is_first=True,is_second = True, is_cross_view=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_cnn,featmap = self.encoder_q(im_q)  # queries: NxC
        q = self.encoder_q.fc(q_cnn)
        q = nn.functional.normalize(q, dim=1, p=2.0)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k,_ = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1, p=2.0)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        T = self.T_cross if is_cross_view else self.T
        logits /= T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second:
            self._dequeue_and_enqueue2(k)

        return logits, labels, q, featmap

class MoCo_gradcam_KL_3q(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, T_cross=0.15, teacher_dim = 1024,mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_gradcam_KL_3q, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.T_cross = T_cross

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.encoder_k = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.use_ddp = torch.distributed.is_initialized()

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue_gl", torch.randn(dim, K))
        self.queue_gl = nn.functional.normalize(self.queue_gl, dim=0)
        self.register_buffer("queue_ptr_gl", torch.zeros(1, dtype=torch.long))

        self.bilinear = 32
        self.conv16 = nn.Conv2d(2048, self.bilinear, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(7, stride=1)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr2[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_gl(self, keys):  # <-- 新增
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr_gl)
        assert self.K % batch_size == 0
        self.queue_gl[:, ptr: ptr + batch_size] = keys.T
        self.queue_ptr_gl[0] = (ptr + batch_size) % self.K


    def max_mask(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)  # 改为非就地操作

        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        return att_max

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q, im_k, is_first=True,is_second = True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_cnn,featmap = self.encoder_q(im_q)  # queries: NxC
        #q_feature_like_teacher = self.encoder_q.distill(q_cnn)
        #q_feature_like_teacher = nn.functional.normalize(q_feature_like_teacher, dim = 1)   #, p=2.0
        q = self.encoder_q.fc(q_cnn)
        q = nn.functional.normalize(q, dim=1, p=2.0)
        #q_cnn = nn.functional.normalize(q_cnn, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k,_ = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1, p=2.0)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        elif is_second:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue2.clone().detach()])
        else:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue_gl.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        T = self.T_cross if (is_first == False and is_second == False) else self.T
        logits /= T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second:
            self._dequeue_and_enqueue2(k)
        else:
            self._dequeue_and_enqueue_gl(k)

        return logits, labels, q, featmap


class MoCo_gradcam_KL_cos(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, teacher_dim = 1024,mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo_gradcam_KL_cos, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.encoder_k = resnet50(num_classes=dim,teacher_dim = teacher_dim)
        self.use_ddp = torch.distributed.is_initialized()

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc
            )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # create the queue
        self.register_buffer("queue2", torch.randn(dim, K))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)

        self.register_buffer("queue_ptr2", torch.zeros(1, dtype=torch.long))

        self.bilinear = 32
        self.conv16 = nn.Conv2d(2048, self.bilinear, kernel_size=1, stride=1, padding=0,
                                bias=False)
        self.bn16 = nn.BatchNorm2d(self.bilinear)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(7, stride=1)

        self.lg_proj = nn.Sequential(
            nn.Linear(2048, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        self.lg_use_distill = False   # 先关；有 teacher 再开
        self.lg_distill_type = 'cos'  # 'cos' 或 'kl'
        self.lg_distill_temp = 0.07

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + \
                param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue1(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr2)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr: ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr2[0] = ptr

    def max_mask(self, featmap):
        featcov16 = self.conv16(featmap)
        featcov16 = self.bn16(featcov16)
        featcov16 = self.relu(featcov16)  # 改为非就地操作

        img, _ = torch.max(featcov16, axis=1)
        img = img - torch.min(img)
        att_max = img / (1e-7 + torch.max(img))

        return att_max

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def local_global_align(self, im_g, im_locals, lambda_align=1.0, lambda_distill=0.0, teacher=None):
        """
        只用来替换原来的 cross 对比损失（全局↔局部）。
        不影响你原有的 全局↔全局、局部↔局部对比，也不动原来的蒸馏。
        """
        import torch
        import torch.nn.functional as F

        if not isinstance(im_locals, (list, tuple)):
            im_locals = [im_locals]

        # 共享 encoder_q。你的 encoder_q(x) 本来就返回 GAP 后的向量（如 [B,2048]）
        qg_cnn, _ = self.encoder_q(im_g)              # 和你 forward 里的一样取法
        z_g = F.normalize(self.lg_proj(qg_cnn), dim=-1)  # [B,256]，proj 来自“改动1”的 self.lg_proj

        loss_aligns = []
        for im_l in im_locals:
            ql_cnn, _ = self.encoder_q(im_l)
            z_l = F.normalize(self.lg_proj(ql_cnn), dim=-1)
            cos_sim = (z_g * z_l).sum(-1)             # 余弦相似度
            loss_aligns.append(1.0 - cos_sim.mean())

        loss_align = torch.stack(loss_aligns).mean() if loss_aligns else torch.tensor(0., device=z_g.device)

        # 这里不做任何蒸馏（你要求保留原蒸馏逻辑不变，且 cross 只换成对齐）
        # 所以固定返回 distill = 0
        loss_distill = torch.tensor(0., device=z_g.device)

        return lambda_align * loss_align, loss_distill

    def forward(self, im_q, im_k, is_first=True,is_second = True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q_cnn,featmap = self.encoder_q(im_q)  # queries: NxC
        #q_feature_like_teacher = self.encoder_q.distill(q_cnn)
        #q_feature_like_teacher = nn.functional.normalize(q_feature_like_teacher, dim = 1)   #, p=2.0
        q = self.encoder_q.fc(q_cnn)
        q = nn.functional.normalize(q, dim=1, p=2.0)
        #q_cnn = nn.functional.normalize(q_cnn, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k,_ = self.encoder_k(im_k)  # keys: NxC
            k = self.encoder_k.fc(k)
            k = nn.functional.normalize(k, dim=1, p=2.0)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        if is_first:
            l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        else:
            l_neg = torch.einsum(
                "nc,ck->nk", [q, self.queue2.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        if is_first:
            self._dequeue_and_enqueue1(k)
        elif is_second:
            self._dequeue_and_enqueue2(k)

        return logits, labels, q, featmap