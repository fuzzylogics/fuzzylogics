import os
import yaml
import numpy as np
import models
import torch
import torch.nn

def read_options_yaml(fn):
    with open(os.path.join(os.getcwd() + '/' + fn), 'r') as f:
        opts = yaml.full_load(f)
        return opts

def print_options_yaml(opts):
    print(yaml.dump(opts))



def define_train(train_opts, is_ddp=False, device_or_rank=0):
    # model
    model = define_model(train_opts, device_or_rank, is_ddp)
    # optimizer
    optimizer = define_optimizer(train_opts, model)   
    #loss
    loss_func_dict = {
        'mse': torch.nn.MSELoss
    }
    loss_func = loss_func_dict[train_opts['loss_function'].lower()]()    
    return model, optimizer, loss_func
def define_model(train_opts, device_or_rank, is_ddp):
    model_name = train_opts['model']['name']
    if 'arguments' in train_opts['model']:
        model_arguments = train_opts['model']['arguments']
    else: model_arguments = {}
    if model_arguments==None or model_arguments==[None] or model_arguments=='' or model_arguments==[]:
        model_arguments = {}
    model = eval('models.'+model_name)(**model_arguments)
    model.to(device_or_rank)
    if is_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        print('converting to DDP model ...')
        model = DDP(model, device_ids=[device_or_rank])
    return model
def define_optimizer(train_opts, model):
    optimizer_name = train_opts['optimizer']['name']
    if 'arguments' in train_opts['optimizer']:
        optimizer_arguments = train_opts['optimizer']['arguments']
    else: optimizer_arguments = {}
    if optimizer_arguments==None or optimizer_arguments==[None] or optimizer_arguments=='' or optimizer_arguments==[]:
        optimizer_arguments = {}
    optimizer = eval('torch.optim.'+optimizer_name)(model.parameters(), **optimizer_arguments)   
    return optimizer




def define_test(test_opts, is_ddp=False, device=0):
    model = define_model(test_opts, device, is_ddp)
    return model


def gen_test_opts(opts):
    import copy
    test_opts = copy.deepcopy(opts['test'])
    test_opts['molecules'] = combine_opts(test_opts['molecules'], opts['molecules'])
    test_opts['sampling'] = combine_opts(test_opts['sampling'], opts['sampling'])
    return test_opts
def gen_val_opts(opts):
    import copy
    val_opts = copy.deepcopy(opts['validate'])
    val_opts['molecules'] = combine_opts(val_opts['molecules'], opts['molecules'])
    val_opts['sampling'] = combine_opts(val_opts['sampling'], opts['sampling'])
    # print('val______________', val_opts)
    return val_opts

class Gen_mol_kwargs():
    def __init__(self, mol_opts):
        self.i = 0
        self.struc_from = mol_opts['struc_from']
        if mol_opts['struc_from'] == 'dimer_range':
            self.atoms = [mol_opts['dimer_range']['atoms']]
            if 'dists' in mol_opts['dimer_range'] and mol_opts['dimer_range']['dists'] is not None:
                self.dimer_dists = mol_opts['dimer_range']['dists']
            elif all(x in mol_opts['dimer_range'] for x in ['n_struc','bg_dist','ed_dist']):
                self.dimer_dists = np.linspace(mol_opts['dimer_range']["bg_dist"], mol_opts['dimer_range']["ed_dist"], mol_opts['dimer_range']["n_struc"])
            else: raise NotImplementedError
            self.basis_names = [mol_opts['dimer_range']['basis']]
            self.n_mols = len(self.dimer_dists)
        elif mol_opts['struc_from'] == 'function':
            import gen_molecules
            gen_mols_kwargs_func = eval('gen_molecules.' + mol_opts['struc_func']['name'])
            self.all_kwargs = gen_mols_kwargs_func(**mol_opts['struc_func']['arguments'])
            self.n_mols = len(self.all_kwargs)

    def __next__(self):
        if self.i < self.n_mols:
            if self.struc_from == 'dimer_range':
                # only one set of atoms (self.atoms[0]) and basis for a range of dimers
                kwargs = {'linear_dists':[self.dimer_dists[self.i]], 'linear_atoms':self.atoms[0], 'basis_name':self.basis_names[0]}
                self.i += 1
                return kwargs
            if self.struc_from == 'function':
                self.i += 1
                return self.all_kwargs[self.i-1]

        else: raise StopIteration

    def __len__(self):
        return self.n_atoms

    def __iter__(self):
        return self


def compare_opts(opts1, opts2):
    if opts1 == opts2:
        return True
    elif type(opts1) is dict and type(opts2) is dict:
        return len(opts1)==len(opts2) and all(k in opts2 and compare_opts(opts1[k], opts2[k]) for k in opts1)
    elif type(opts1) is list and type(opts2) is list:
        opts1, opts2 = tuple(sorted(opts1)), tuple(sorted(opts2))
        return len(opts1)==len(opts2) and compare_opts(opts1, opts2)
    else:
        return False

def combine_opts(opts1, opts2):
    import copy
    opts = copy.deepcopy(opts1)
    if type(opts) is not dict:
        return opts
    else:
        for key in opts2:
            if key not in opts:
                opts[key] = opts2[key]
            else:
                opts[key] = combine_opts(opts[key], opts2[key])
    return opts
    




def test():
    opts = read_options_yaml('options.yaml')
    print_options_yaml(opts)
    # for key, value in opts.items():
    #     print (key + " : " + str(value))
    mol_kwargs = Gen_mol_kwargs(opts['molecules'])

    for s in mol_kwargs:
        print(s)

if __name__ == '__main__':
    test()