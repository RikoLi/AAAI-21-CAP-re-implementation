# Memory bank implementation
# Reference: https://github.com/yxgeee/SpCL/blob/master/spcl/models/hm.py

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import nn, autograd

# 重构
class UpdateFunction(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, indices, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, indices)
        return inputs.mm(ctx.features.t()) # inner-product similarity

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, indices = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features) # BP
        
        # update memory bank on CPU
        for x, y in zip(inputs, indices):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1 - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm() # L2-normalization

        return grad_inputs, None, None, None

def _update_memory(inputs, indices, features, momentum):
    return UpdateFunction.apply(inputs, indices, features, torch.Tensor([momentum]).to(inputs.device))

class MemoryBank(nn.Module):
    '''
    Memory bank of feature centroids.

    Args:
        num_feature_dims: int, number of feature dimensions.
        num_samples: int, number of centroid features stored in the memory bank.
        temp: float, temperature factor of the contrastive loss.
        momentum: float, momentum factor in updating of the memory bank.
    
    Returns:
        A memory bank instance.
    '''
    def __init__(self, num_feature_dims, num_samples, temp=0.05, momentum=0.2):
        super(MemoryBank, self).__init__()
        self.num_features = num_feature_dims
        self.num_samples = num_samples
        self.momentum = momentum
        self.temp = temp

        # memory bank storage structure
        self.register_buffer('features', torch.zeros((self.num_samples, self.num_features)))
        self.register_buffer('labels', torch.zeros(num_samples).long())
    
    def forward(self, inputs, indices):
        # update the memory bank
        inputs = _update_memory(inputs, indices, self.features, self.momentum) # -> shape: (num_samples, num_features)
        inputs /= self.temp
        
        # compute contrastive loss
        targets = self.labels[indices.long()].clone()
        logits = F.log_softmax(inputs, dim=1)
        return F.nll_loss(logits, targets)

