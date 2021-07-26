import torch
import models
import os.path as osp


def save_check(folder, epoch, model, optimizer, loss, model_str=None, optimizer_str=None):
    torch.save({
        'epoch': epoch,
        'model_str': model_str,
        'model_state_dict': model.state_dict(),
        'optimizer_str': optimizer_str,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        # 'edge_index': edge_index,
    }, osp.join(folder, "checkpoint%d.chk" %(epoch)) )


def load_check(f, model=None, optimizer=None):
    checkpoint = torch.load(f)
    epoch = checkpoint['epoch']
    # model
    if model is None and 'model_str' in checkpoint and checkpoint['model_str'] is not None:
        model_str = checkpoint['model_str']
        model = eval('models.'+model_str)      
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer
    # optimizer_str = checkpoint['optimizer_str']
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    elif optimizer is None and 'optimizer_str' in checkpoint and checkpoint['optimizer_str'] is not None:
        # optimizer = checkpoint['optimizer']
        optimizer_str = checkpoint['optimizer_str']
        optimizer = eval('torch.optim.'+optimizer_str)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss = checkpoint['loss']
    # if 'edge_index' in checkpoint:
    #     edge_index = checkpoint['edge_index']
    # else:
    #     edge_index = None
    return epoch, model, optimizer, loss



