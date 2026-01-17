import torch
import torch.nn as nn
import random
import math

from resnet import resnet50,resnet18

import clip
from sklearn.decomposition import PCA
import numpy as np

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