from copy import deepcopy
import torch
import numpy as np
import math

def get_parameter(model, name):
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)

def func_linear(original_s, delta_s):
    max_ori=original_s[0]
    for i in range(len(delta_s)):
        if delta_s[i]>=max_ori:
            delta_s[i]=1/2*delta_s[i]+1/2*max_ori
    return delta_s
    

def func_log(original_s, delta_s):
    max_ori=original_s[0]
    for i in range(len(delta_s)):
        if delta_s[i]>=max_ori:
            delta_s[i]=np.log(delta_s[i])+max_ori-np.log(max_ori)
    return delta_s

def func_logn(original_s, delta_s, n):
    max_ori=original_s[0]
    for i in range(len(delta_s)):
        if delta_s[i]>=max_ori:
            delta_s[i]=math.log(delta_s[i],n)+max_ori-math.log(max_ori,n)
    return delta_s

class PRUNE:
    def __init__(self, reduce_name="log1_2"):
        self.reduce_name = reduce_name

    def initializtion(self, init_weights, device):
        self.init_weights_copy = {n: p.clone().cpu() for n, p in init_weights.items()}
        self.device = f"cuda:{device}"

    def reduce_weights(self, model):
        with torch.no_grad():
            for name, par_origin in self.init_weights_copy.items():
                delta = get_parameter(model, name).clone().detach().cpu() - par_origin
                u,s,v=np.linalg.svd(delta.numpy(),full_matrices=1,compute_uv=1)
                u0,s0,v0=np.linalg.svd(par_origin.numpy(),full_matrices=1,compute_uv=1)

                rank=np.linalg.matrix_rank(delta.numpy())
                if self.reduce_name=="linear":
                    s2=func_linear(s0, s)
                elif self.reduce_name=="log":
                    s2=func_log(s0, s)
                elif self.reduce_name=="log2":
                    s2=func_logn(s0, s, 2)
                elif self.reduce_name=="log1_5":
                    s2=func_logn(s0, s, 1.5)
                elif self.reduce_name=="log1_2":
                    s2=func_logn(s0, s, 1.2)
                
                u2=u[:,:rank]
                s2=np.diag(s2[:rank])
                v2=v[:rank]
        
                delta1=np.dot(np.dot(u2,s2),v2)

                get_parameter(model, name)[...] = (par_origin+delta1).to(dtype=par_origin.dtype).to(self.device)
        return model