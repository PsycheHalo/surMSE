import torch
import math


def surMSE(input,target,dim=None,scale=False,eps=1e-8,meanOut=True,haveBatch=True):
    if dim is None and haveBatch:
        dim=tuple(range(1,len(torch.broadcast_shapes(input.shape,target.shape))))
    
    inputNormSquare=torch.linalg.norm(input,ord=2,dim=dim,keepdim=False).square()
    targetNormSquare=torch.linalg.norm(target,ord=2,dim=dim,keepdim=False).square()
    dot=(input*target).sum(dim=dim,keepdim=False)
    
    loss=(inputNormSquare*targetNormSquare-dot.square())+(targetNormSquare-dot).clamp(min=0)
    if scale:
        loss=loss/(targetNormSquare+eps)
    
    if meanOut:
        return loss.mean()
    else:
        return loss
       
