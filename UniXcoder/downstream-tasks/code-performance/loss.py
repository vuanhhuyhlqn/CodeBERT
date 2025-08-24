import torch
import torch.nn as nn
import torch.nn.functional as F

class PerformanceMetricLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, perfs):
        # print("embeddings.requires_grad =", embeddings.requires_grad)
        # print("perfs.requires_grad =", perfs.requires_grad)

        mean = perfs.mean()
        std = perfs.std()
        perfs_norm = (perfs - mean) / (std + 1e-8)

        target = torch.abs(perfs_norm[:, None] - perfs_norm[None, :])
        
        embeddings = embeddings.float()

        cos_sim = F.cosine_similarity(embeddings[:, None, :], embeddings[None, :, :], dim=-1)
        dist_mat = 1 - cos_sim

        loss = (dist_mat - target).pow(2).mean()
        return loss