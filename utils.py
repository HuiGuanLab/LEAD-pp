import torch
import numpy as np
import random
import os
from PIL import ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F


class Similarity(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, g_s, g_t):

        return self.similarity_loss(g_s, g_t)

    def similarity_loss(self, f_s, f_t):
        # print(f_s.shape)
        bsz = f_s.shape[0]
        # print(bsz)
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        # print("===============")
        return loss

class Similarity_entropy(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity_entropy, self).__init__()

    def forward(self, g_s, g_t,weight):

        return self.similarity_loss(g_s, g_t,weight)

    def similarity_loss(self, f_s, f_t,entropy):
        # print(f_s.shape)
        bsz = f_s.shape[0]
        # print(bsz)
        f_s = f_s.view(bsz, -1)
        f_t = f_t.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        # G_s = G_s / G_s.norm(2)
        G_s = torch.nn.functional.normalize(G_s)
        G_t = torch.mm(f_t, torch.t(f_t))
        # G_t = G_t / G_t.norm(2)
        G_t = torch.nn.functional.normalize(G_t)

        G_diff = G_t - G_s
        #G_diff_weighted = G_diff * entropy.view(bsz, 1)  # 广播权重到每一行

        loss = ((G_diff * G_diff) * entropy.view(bsz, 1)).view(-1, 1).sum(0) / (bsz * bsz)
        # print("===============")
        return loss


class Similarity_mixup(nn.Module):
    """Similarity-Preserving Knowledge Distillation, ICCV2019, verified by original author"""
    def __init__(self):
        super(Similarity_mixup, self).__init__()
        self.criterion_KLD = nn.KLDivLoss(reduction='batchmean')


    def forward(self, g_s, g_t1,g_t2, temperature):

        return self.similarity_loss(g_s, g_t1, g_t2, temperature)

    def similarity_loss(self, f_s, f_t_vit, f_t_clip, temperature):
        # print(f_s.shape)
        bsz = f_s.shape[0]
        # print(bsz)
        f_s = f_s.view(bsz, -1)
        f_t_vit = f_t_vit.view(bsz, -1)
        f_t_clip = f_t_clip.view(bsz, -1)

        G_s = torch.mm(f_s, torch.t(f_s))
        #G_s = torch.nn.functional.normalize(G_s)
        G_t_vit = torch.mm(f_t_vit, torch.t(f_t_vit))
       # G_t_vit = torch.nn.functional.normalize(G_t_vit)
        G_t_clip = torch.mm(f_t_clip, torch.t(f_t_clip))
       # G_t_clip = torch.nn.functional.normalize(G_t_clip)
        
        # 创建一个新的张量用于存储融合结果
        fused_tensor = torch.empty(bsz, bsz).cuda()

        # 处理主对角线上的值
        diagonal_indices = torch.arange(bsz).cuda()
        fused_tensor[diagonal_indices, diagonal_indices] = torch.max(G_t_vit[diagonal_indices, diagonal_indices], G_t_clip[diagonal_indices, diagonal_indices])

        # 处理非主对角线上的值
        for i in range(bsz):
            for j in range(bsz):
                if i != j:
                    fused_tensor[i, j] = torch.min(G_t_vit[i, j], G_t_clip[i, j])

        #G_t = F.softmax(fused_tensor.div(temperature), dim=-1)
        #G_s = F.log_softmax(G_s.div(temperature), dim=-1)
        G_t = stable_softmax(fused_tensor, temperature)
        G_s = stable_log_softmax(G_s, temperature)

        loss = self.criterion_KLD(G_s,G_t)
        
        return loss
    
def stable_softmax(logits, temperature=1.0, eps=1e-8):
    logits = logits / temperature
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    exp_logits = torch.exp(logits - max_logits)
    softmax_output = exp_logits / (torch.sum(exp_logits, dim=-1, keepdim=True) + eps)
    return softmax_output

def stable_log_softmax(logits, temperature=1.0, eps=1e-8):
    logits = logits / temperature
    max_logits, _ = torch.max(logits, dim=-1, keepdim=True)
    log_sum_exp = torch.log(torch.sum(torch.exp(logits - max_logits), dim=-1, keepdim=True) + eps)
    log_softmax_output = logits - max_logits - log_sum_exp
    return log_softmax_output


class teacher_mixup(nn.Module):
    #t1:vit
    #t2:clip
    def __init__(self,t1,t2):
        super(teacher_mixup, self).__init__()
        self.t1 = t1
        self.t2 = t2

    #可以增加temperature
    def forward(self, im_1,im_2,temperature_t):    
        #特征提取
        feature_t1_q = self.t1(im_1)
        feature_t1_k = self.t1(im_2)
        feature_t2_q = self.t2.module.encode_image(im_1)
        feature_t2_k = self.t2.module.encode_image(im_2)
        feature_size1 = feature_t1_q.shape[-1]
        feature_size2 = feature_t2_q.shape[-1]

        #标准化
        feature_t1_q = nn.functional.normalize(feature_t1_q,dim = 1)    #, p=2.0
        feature_t1_k = nn.functional.normalize(feature_t1_k,dim = 1)
        feature_t2_q = nn.functional.normalize(feature_t2_q,dim = 1)
        feature_t2_k = nn.functional.normalize(feature_t2_k,dim = 1)
        
        #计算相似度
        cosine_similarity_t1 = F.cosine_similarity(feature_t1_q, feature_t1_k, dim=1)
        cosine_similarity_t2 = F.cosine_similarity(feature_t2_q, feature_t2_k, dim=1)
        cosine_similarity_t1 = cosine_similarity_t1.unsqueeze(1)
        cosine_similarity_t2 = cosine_similarity_t2.unsqueeze(1)
        cosine_similarity = torch.concat([cosine_similarity_t1,cosine_similarity_t2],dim=1)
        cosine_similarity_softmax = F.softmax(cosine_similarity.div(temperature_t), dim=1)
        #print(cosine_similarity_softmax)

        #权重与原始特征相乘
        weight1 = cosine_similarity_softmax[:,0].unsqueeze(1)
        weight2 = cosine_similarity_softmax[:,1].unsqueeze(1)
        weight1_expand = weight1.expand(-1, feature_size1)
        weight2_expand = weight2.expand(-1, feature_size2)

        output_feature1 = feature_t1_q * weight1_expand
        output_feature2 = feature_t2_q * weight2_expand

        output_feature = torch.concat([output_feature1,output_feature2], dim = -1)
        output_feature = nn.functional.normalize(output_feature, p=2.0, dim = -1)



        return output_feature


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.set_grad_enabled(False)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        if param_group["lr"] is not None:
            return param_group["lr"]

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
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



def gradient_normalizers(grads, losses, normalization_type = 'loss+'):
    gn = {}
    if normalization_type == 'l2':
        for t in grads:
            gn[t] = np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'loss':
        for t in grads:
            gn[t] = losses[t]
    elif normalization_type == 'loss+':
        for t in grads:
            gn[t] = losses[t] * np.sqrt(np.sum([gr.pow(2).sum().data.cpu() for gr in grads[t]]))
    elif normalization_type == 'none':
        for t in grads:
            gn[t] = 1.0
    else:
        print('ERROR: Invalid Normalization Type')
    return gn