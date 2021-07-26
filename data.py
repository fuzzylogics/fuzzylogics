from re import S
from torch.utils.data import Dataset, DataLoader
from options import Gen_mol_kwargs
from grids import SamplingCoords
import os.path as osp
import numpy as np
import torch
from molecule import Molecule
from simple_wy import oep_search
import pickle
import h5py
import time

class MolOnGrid(Dataset):
    """dataset of molecules and its density on grid"""

    def __init__(self, 
                 mol_opts, sampling_opts, 
                 path, 
                 input_shape='vanilla', 
                 device='cpu', gen_rho_device = 'cuda',
                 save_mol_in_disk=True, save_mol_in_mem=True,
                 x_is_sparse=False, x_to_zero_cutoff=1.0e-8,
                 save_x_in_disk=True,
                 save_x_in_mem=True, save_x_in_device=False):
        
        # self.opts = opts
        self.mol_opts = mol_opts
        self.sampling_opts = sampling_opts
        self.device = device
        self.gen_rho_device = gen_rho_device
        self.grid_level = mol_opts['grids']['grid_level']
        self.grid_sym_list = mol_opts['grids']['sym']
        self.mol_kwargs_list = [kwargs for kwargs in Gen_mol_kwargs(self.mol_opts)]
        n = len(self.mol_kwargs_list)
        self.n = n
        # mols
        self.save_mol_in_disk = save_mol_in_disk
        self.save_mol_in_mem = save_mol_in_mem
        self.mols_in_disk = [None]*n
        self.mols_in_mem = [None]*n
        # edges in the grid
        self.gen_edge = bool('edges' in mol_opts['grids'])
        # self.nnn = 10 if not self.gen_edge else mol_opts['grids']['edges']['nnn']
        if self.gen_edge:
            if 'nnn' in mol_opts['grids']['edges']:
                self.nnn = mol_opts['grids']['edges']['nnn']
            else: self.nnn = 10
        else: self.nnn = None

        # rho data
        self.input_shape = input_shape
        self.save_x_in_disk = save_x_in_disk
        self.save_x_in_disk_with = 'pickle'
        self.save_x_in_mem = save_x_in_mem
        self.save_x_in_device = save_x_in_device
        self.xs_in_disk = [None]*n
        self.xs_in_mem = [None]*n
        self.xs_in_device = [None]*n
        # zeros
        self.x_is_sparse = x_is_sparse
        self.x_to_zero_cutoff = x_to_zero_cutoff
        # paths
        self.root = path
        self.mols_folder = osp.join(path, 'molecules')
        self.xs_folder = osp.join(path, 'data_rho')


    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # molecule
        if self.mols_in_mem[i] is not None:
            mol = self.mols_in_mem[i]
        elif self.mols_in_disk[i] is not None:
            mol = self.load_mol(self.mols_in_disk[i])
        else:
            mol = self.gen_mol(i)
        # input x
        if self.xs_in_device[i] is not None:
            x = self.xs_in_device[i]
        elif self.xs_in_mem[i] is not None:
            x = self.xs_in_mem[i].to(self.device)
        elif self.xs_in_disk[i] is not None:
            x = self.load_x(self.xs_in_disk[i])
            x = x.to(self.device)
        else:
            x = self.gen_x(mol, i)
        return {'i':i, 'mol':mol, 'x':x}

    def gen_all(self):
        print("Preparing all structures and data ...")
        for i in range(self.n):
            mol = self.gen_mol(i)
            x = self.gen_x(mol, i)
    
    def gen_mol(self, i):
        mol = Molecule(**self.mol_kwargs_list[i])
        mol.build_grid(grid_level=self.grid_level, sym_list=self.grid_sym_list, 
                       gen_edge=self.gen_edge, nnn=self.nnn)
        mol_fn = "mol_%s_grid_%s.pkl"%(mol.id, mol._grid.id)
        if osp.isfile(osp.join(self.mols_folder, mol_fn)):
            mol = self.load_mol(mol_fn)
        else:
            mol.generate_all(grid_level=self.grid_level, grid_sym_list=self.grid_sym_list,
                             gen_edge=self.gen_edge, nnn=self.nnn)
            if self.save_mol_in_disk:
                mol.save(folder_path=self.mols_folder, fn=mol_fn)
                mol.write_structure(osp.join(self.mols_folder, "mol_%s.str"%(mol.id)))
            if hasattr(mol, '_pyscf_mol'): del mol._pyscf_mol # DDP seems to need to pickle
        if self.save_mol_in_mem: self.mols_in_mem[i] = mol
        if self.save_mol_in_disk: self.mols_in_disk[i] = mol_fn
        return mol

    def gen_x(self, mol, i):
        # density on grid
        sampling_coords = SamplingCoords(mol.grid, sampling_opts=self.sampling_opts)
        sampling_coords.mesh_out()
        if self.save_x_in_disk_with == 'pickle': ext = 'pkl'
        elif self.save_x_in_disk_with == 'h5py': ext = 'hdf5'
        else: raise NotImplementedError
        x_fn = "mol_%s_grid_%s_dtype_%d%s.%s" \
                %(mol.id, sampling_coords.id, 32, 'sp' if self.x_is_sparse else 'ds', ext)  # float32
        if osp.isfile(osp.join(self.xs_folder, x_fn)):
            x = self.load_x(x_fn)
            self.xs_in_disk[i] = x_fn
        else:
            x = mol.rho('ccsd', grid=sampling_coords, deriv=(0,1), 
                        package='torch', device=self.gen_rho_device)
            x = x.transpose(1,0)  # dim0=4=1+3deriv, dim1=grid_size
            # remove data augmentation dims (rotations and mirrors)
            if self.input_shape == 'vanilla': x = x.squeeze(2).squeeze(2)            
            else: raise NotImplementedError
            # remove small values in x
            if self.x_to_zero_cutoff is not None:
                x[x<self.x_to_zero_cutoff] = 0.0
            # make x sparse
            if self.x_is_sparse:
                x_n0_idx = torch.nonzero(x).t()
                x_n0_v = x[tuple(x_n0_idx[i] for i in range(len(x_n0_idx)))]
                x = torch.sparse.FloatTensor(x_n0_idx, x_n0_v, x.shape)
            else: x = x.contiguous()
            # save x
            if self.save_x_in_disk:
                if self.x_is_sparse:
                    x_save = {'indices':x_n0_idx, 'values':x_n0_v, 'size':x.shape}
                else: x_save = x.cpu().numpy()
                if self.save_x_in_disk_with == 'pickle':
                    with open(osp.join(self.xs_folder, x_fn), 'wb') as f:
                        pickle.dump(x_save, f, protocol=pickle.HIGHEST_PROTOCOL)
                elif self.save_x_in_disk_with == 'h5py':
                    with h5py.File(osp.join(self.xs_folder, x_fn), 'w') as f:
                        h5data_w = f.create_dataset('x', x_save.shape, dtype='f4', data=x_save, 
                                                    compression="gzip", compression_opts=9)
                else: raise NotImplementedError
                self.xs_in_disk[i] = x_fn 
        if self.save_x_in_mem: self.xs_in_mem[i] = x.to('cpu')
        x = x.to(self.device)
        if self.save_x_in_device: self.xs_in_device[i] = x        
        return x

    def load_mol(self, mol_fn):
        print('loading molecule: %s'%(mol_fn))
        with open(osp.join(self.mols_folder, mol_fn), 'rb') as f:
            mol = pickle.load(f)
        return mol

    def load_x(self, x_fn):
        print('loading input: %s ... '%(x_fn))
        t0 = time.time()
        if self.save_x_in_disk_with == 'pickle':
            with open(osp.join(self.xs_folder, x_fn), 'rb') as f:
                x = pickle.load(f)
        elif self.save_x_in_disk_with == 'h5py':
            with h5py.File(osp.join(self.xs_folder, x_fn), 'r') as f:
                x = f['x'][:]
        else: raise NotImplementedError
        t1 = time.time()
        if self.x_is_sparse:
            x = torch.sparse.FloatTensor(*(x.values()))
        else: x = torch.from_numpy(x)
        t2 = time.time()
        print(' Done. Loading time: %f, Converting time: %f'%(t1-t0, t2-t1))
        return x


class MolOnGrid_graph(MolOnGrid):
    def __init__(self, nnn,
                 mol_opts,
                 path,
                 device='cpu', gen_rho_device = 'cuda',
                 x_is_sparse=False, x_to_zero_cutoff=1.0e-8,
                 save_mol_in_disk=True, save_mol_in_mem=True, 
                 save_x_in_disk=True, save_x_in_mem=True, save_x_in_device=False):
        super().__init__(mol_opts=mol_opts, sampling_opts={}, 
                         path=path, 
                         input_shape='vanilla', 
                         device='cpu', gen_rho_device = 'cuda',
                         save_mol_in_disk=save_mol_in_disk, save_mol_in_mem=save_mol_in_mem,
                         x_is_sparse=x_is_sparse, x_to_zero_cutoff=x_to_zero_cutoff,
                         save_x_in_disk=save_x_in_disk, save_x_in_mem=save_x_in_mem, save_x_in_device=save_x_in_device)
        # edges are always needed for 
        self.gen_edge = True
        if nnn is not None:
            self.nnn = nnn
        elif 'nnn' in mol_opts['grids']['edges']:
            self.nnn = mol_opts['grids']['edges']['nnn']
        else: self.nnn = 10
        self.input_shape = 'vanilla'
        


def collate_fn_MolOnGrid(data):
    idx = [d['i'] for d in data]
    mol_batch = [d['mol'] for d in data]
    x_batch = torch.cat([d['x'] for d in data], 0)
    grid_sizes = [d['x'].shape[0] for d in data]
    return {'i': idx, 'mol': mol_batch, 'x': x_batch, 'grid_size':grid_sizes}

def collate_fn_MolOnGrid_w_grid_edges(data):
    # idx
    idx = [d['i'] for d in data]
    # mol
    mol_batch = [d['mol'] for d in data]
    # x
    x_batch = torch.cat([d['x'] for d in data], 0)
    grid_sizes = [d['x'].shape[0] for d in data]
    # edge
    ei_batch = torch.cat([torch.from_numpy(d['mol'].grid.edge_idx).long() for d in data], 1)
    ev_batch = torch.cat([torch.from_numpy(d['mol'].grid.edge_vec).float() for d in data], 0)
    edge_sizes = [d['mol'].grid.edge_vec.shape[0] for d in data]
    return {'i': idx, 'mol': mol_batch, 
            'x': x_batch, 'grid_size':grid_sizes,
            'ei': ei_batch, 'ev': ev_batch, 'edge_size':edge_sizes}

def reg_1deriv(y, grid, device):
    assert tuple(grid.coords.shape[:-1]) == tuple(y.shape)
    k = grid.nnn
    edge_vec = torch.from_numpy(grid.edge_vec).float().to(device, non_blocking=True)
    # edge_idx = torch.from_numpy(grid.edge_idx).long().to(device)
    # print('y,', y[grid.edge_idx[1]].shape)
    # print('d,', edge_vec.shape)
    # print('dmin,', (edge_vec**2).sum(-1))
    # print('edge_index,', grid.edge_idx[0,0:10], grid.edge_idx[1,0:10])
    # print('coords,', grid.coords[grid.edge_idx[0,0:10]], grid.coords[grid.edge_idx[1,0:10]])
    # print('loop,', [i for i in range(grid.edge_idx.shape[1]) if grid.edge_idx[0][i]==grid.edge_idx[1][i]])
    # print('loop', grid.coords[132])
    deriv2 = (y[grid.edge_idx[1]] - y[grid.edge_idx[0]])**2 / (edge_vec**2).sum(-1)

    # print('reg_____________')
    # print(edge_vec.sum(-1))
    # print(deriv2)
    # print(deriv2.sum())

    # deriv2 = (y[edge_idx[1]] - y[edge_idx[0]])**2 / (edge_vec**2).sum(-1)
    # print(sum(deriv2))
    return deriv2.mean()

def reg_1deriv_pow(y, grid, device, pow):
    assert tuple(grid.coords.shape[:-1]) == tuple(y.shape)
    k = grid.nnn
    edge_vec = torch.from_numpy(grid.edge_vec).float().to(device, non_blocking=True)
    if pow == 0:
        deriv2 = (y[grid.edge_idx[1]] - y[grid.edge_idx[0]])**2
    elif pow == 2:
        deriv2 = (y[grid.edge_idx[1]] - y[grid.edge_idx[0]])**2 / ((edge_vec**2).sum(-1))
    else:
        deriv2 = (y[grid.edge_idx[1]] - y[grid.edge_idx[0]])**2 / ((edge_vec**2).sum(-1).sqrt()**pow)
    return deriv2.mean()

def reg_2deriv(y, grid, device):
    assert tuple(grid.coords.shape[:-1]) == tuple(y.shape)
    k = grid.nnn
    edge_vec = torch.from_numpy(grid.edge_vec).float().to(device, non_blocking=True)
    # print('edge_vec', edge_vec.shape)
    source_y = y[grid.edge_idx[0]].reshape(-1, k)
    drain_y = y[grid.edge_idx[1]].reshape(-1, k)
    assert all(drain_y[:,0] == y)
    edge_len = (edge_vec**2).sum(-1).sqrt().reshape(-1, k)
    # print('edge_len', edge_len.shape)
    w = 1.0 / edge_len
    y_hat = (source_y * w).sum(-1) / w.sum(-1)
    err2 = (y - y_hat)**2
    return err2.mean()

def reg_0deriv(y):
    return (y**2).sqrt().mean()

# def eval_I(rho_diff, rho_t, weights):
#     assert rho_diff.shape == rho_t.shape
#     assert rho_diff.device == rho_t.device
#     weights = torch.from_numpy(weights).float().to(rho_diff.device, non_blocking=True)
#     print(weights.shape, rho_t.shape)
#     assert weights.shape == rho_t.shape
#     err_I = (rho_diff**2 * weights).sum() / (rho_t**2 * weights).sum()
#     return err_I
def eval_I(rho_i, rho_t, weights):
    assert rho_i.shape == rho_t.shape
    assert rho_i.device == rho_t.device
    weights = torch.from_numpy(weights).float().to(rho_i.device, non_blocking=True)
    # print(weights.shape, rho_t.shape)
    assert weights.shape == rho_t.shape
    err_I = ((rho_i-rho_t)**2 * weights).sum() / \
            ( (rho_i**2 * weights).sum() + (rho_t**2 * weights).sum() )
    return err_I


def aprox_Hartree(rho, grid, device):
    assert tuple(grid.coords.shape[:-1]) == tuple(rho.shape)
    k = grid.nnn
    edge_vec = torch.from_numpy(grid.edge_vec).float().to(device, non_blocking=True)
    weights = torch.from_numpy(grid.weights).float().to(device, non_blocking=True)
    Htr = rho[grid.edge_idx[0]] * weights[grid.edge_idx[0]] / (edge_vec**2).sum(-1).sqrt()
    Htr = Htr.view((-1,k)+tuple(Htr.shape[1:])).sum(1)
    assert tuple(Htr.shape) == tuple(rho.shape)
    return Htr

def loss_w_grid_weights(y, t, w, device):
    if type(w) == np.ndarray:
        w = torch.from_numpy(w).float().to(device, non_blocking=True)
    return ((y - t)**2 * w).sum() / w.sum()
def mse(y, t=None):
    if t is None: return (y**2).mean()
    else: return ((y-t)**2).mean()

def loss_rho_diff_Hartree(rho_diff, itg_2e):
    from opt_einsum import contract
    return contract('ijkl, ji, lk->', itg_2e, rho_diff, rho_diff)

# class MolOnGrid(Dataset):
#     """dataset of molecules and its density on grid"""

#     def __init__(self, opts, path, device='cpu'):
#         print("Preparing all structures and data ...")
#         self.path = path
#         # self.opts = opts
#         self.mol_opts = opts['molecules']
#         self.sampling_opts = opts['sampling']
#         self.device = device
#         self.grid_level = opts['molecules']['grids']['grid_level']
#         self.mol_kwargs_list = [mol for mol in Gen_mol_kwargs(self.mol_opts)]
#         self.n_mol = len(self.mol_kwargs_list)
#         self.saved_mols = [None]*self.n_mol
#         self.saved_x = [None]*self.n_mol

#     def __len__(self):
#         return self.n_mol

#     def __getitem__(self, idx):

#         if self.saved_x[idx] is None:
#             # molecule
#             if self.saved_mols[idx] is None:
#                 mol = self.gen_mol(idx)
#                 self.save_mol(mol, idx)
#             else: mol = load_mol(idx)
#             # density on grid
#             sampling_coords = SamplingCoords(mol.grid, sampling_opts=self.sampling_opts)
#             sampling_coords.mesh_out()
#             folder_path = osp.join(self.path, 'data_rho')
#             rho, rho_fn = mol.rho_n_save('ccsd', grid=sampling_coords, deriv=(0,1), folder_path=folder_path, package='torch', device=self.device)
#             self.saved_x[idx] = rho_fn
#         else: rho = self.load_rho(idx)
#         return {'x':rho, 'mol':mol}

#     def gen_mol(self, idx):
#         print("m",self.mol_kwargs_list[idx])
#         mol = Molecule(**self.mol_kwargs_list[idx])
#         mol.generate_all(grid_level=self.grid_level)
#         return mol

#     def save_mol(self, mol, idx):
#         mol_fn = mol.save(folder_path='')
#         self.saved_mols[idx] = mol_fn

#     def load_mol(self, idx):
#         mol_fn = self.saved_mols[idx]
#         with open(osp.join(self.path, 'molecules', mol_fn), 'rb') as f:
#             mol = pickle.load(f)
#         return mol

#     def load_rho(self, idx):
#         rho_fn = self.saved_mols[idx]
#         with open(osp.join(self.path, 'data_rho', rho_fn), 'rb') as f:
#             rho = pickle.load(f)
#         return rho





class MolOnBasis4Graph_wMol(Dataset):
    """dataset from molecules' DMs"""
    def __init__(self, opts, path, device='cuda',
                 n_nearest_neighbor=None, build_graph_by=None,
                 save_mol_in_disk=True, save_mol_in_mem=True, 
                 save_graph_in_disk=True, save_graph_in_mem=True):
        
        # self.opts = opts
        self.mol_opts = opts['molecules']
        self.data_opts = opts['data']
        self.device = device
        self.mol_kwargs_list = [kwargs for kwargs in Gen_mol_kwargs(self.mol_opts)]
        n = len(self.mol_kwargs_list)
        self.n = n
        self.k = n_nearest_neighbor
        # mols
        self.save_mol_in_disk = save_mol_in_disk
        self.save_mol_in_mem = save_mol_in_mem
        self.mols_in_disk = [None]*n
        self.mols_in_mem = [None]*n
        self.all_n_orbs = [None]*n
        # rho data
        self.input_shape = input_shape
        self.save_graph_in_disk = save_graph_in_disk
        self.save_graph_in_mem = save_graph_in_mem
        self.graphs_in_disk = [None]*n
        self.graphs_in_mem = [None]*n
        # paths
        self.root = path
        self.mols_folder = osp.join(path, 'molecules')
        self.graphs_folder = osp.join(path, 'data_graphs')


    def __len__(self):
        return self.n

    def get_mol(self, i):
        if self.mols_in_mem[i] is not None:
            mol = self.mols_in_mem[i]
        elif self.mols_in_disk[i] is not None:
            mol = self.load_mol(self.mols_in_disk[i])
        else:
            mol = self.gen_mol(i)
        return mol
    def get_graph(self, i):
        if self.graphs_in_mem[i] is not None:
            graph = self.graphs_in_mem[i]
        elif self.graphs_in_disk[i] is not None:
            graph = self.load_graph(self.graphs_in_disk[i])
        else:
            graph = self.gen_graph(self.get_mol(i), i)
        return graph
    def __getitem__(self, i):
        mol = self.get_mol(i) # molecule 
        graph = self.get_graph(i) # input graphs
        # node_feats, edge_idces, edge_feats = graph['node_feats'], graph['edge_idces'], graph['edge_feats']
        return {'mol':mol, 'graph':graph}

    def gen_all_mols(self):
        for i in range(self.n):
            self.get_mol(i)
    def gen_all_graphs(self):
        for i in range(self.n):
            self.get_graph(i)
    def gen_all(self):
        print("Preparing all molecular structures and data ...")
        for i, data in enumerate(self):
            print("#: %d, from Molecule: %s"%(i, data['mol'].custom_description))
    
    def check_nnn(self):
        self.gen_all_mols()
        min_n_orbs = min(self.all_n_orbs)
        if self.k is None:
            print("setting to min n_orbs%d."%min_n_orbs)
            self.k = min_n_orbs
        elif self.k > min_n_orbs:
            print("N nearest neighbor too large, resetting to min n_orbs%d."%min_n_orbs)
            self.k = min_n_orbs
        return self.k


    def gen_mol(self, i):
        mol = Molecule(**self.mol_kwargs_list[i])
        mol_fn = "mol_%s.pkl"%(mol.id)       
        if osp.isfile(osp.join(self.mols_folder, mol_fn)):
            mol = self.load_mol(mol_fn)
        else:
            mol.generate_all_mats()
            if self.save_mol_in_disk: mol.save(folder_path=self.mols_folder, fn=mol_fn)
            if hasattr(self, '_pyscf_mol'): del self._pyscf_mol # DDP seems to need to pickle
        self.all_n_orbs[i] = mol.basis.n_orbs
        if self.save_mol_in_mem: self.mols_in_mem[i] = mol
        if self.save_mol_in_disk: self.mols_in_disk[i] = mol_fn
        return mol

    def gen_graph(self, mol, i):
        graph_fn = "mol_%s_nnn_%d.pkl"%(mol.id, self.k)
        if osp.isfile(osp.join(self.graphs_folder, graph_fn)):
            graph = self.load_graph(graph_fn)
        else:
            graph = build_graph(mol)
        if self.save_graph_in_mem: self.graphs_in_mem[i] = graph
        return graph
    def build_graph(self, mol):
        ...

    def load_mol(self, mol_fn):
        print('loading molecule: %s'%(mol_fn))
        with open(osp.join(self.root, 'molecules', mol_fn), 'rb') as f:
            mol = pickle.load(f)
        return mol

    def load_graph(self, graph_fn):
        print('loading input: %s'%(graph_fn))
        with open(osp.join(self.root, 'data_rho', graph_fn), 'rb') as f:
            graph = pickle.load(f)
        return graph

class MolGraph():
    """ graph for a molecule """
    def __init__(self, mol, graph_opts, device):
        self.node_feats = self.gen_node_feats(mol, node_feats_from)
        self.edge_idces = self.gen_edge_idces(mol, edge_idces_from)
        self.edge_feats = self.gen_edge_feats(mol, edge_feats_from)

    def gen_node_feats(mol, node_feats_from):
        node_feats = []
        if 'centers' in node_feats_from:
            centers = [ g['center']
                        for g in mol.basis.gaussian_dict
                        for _ in range(mol.basis.ang_n_orbs(g['ang'])) ]
            node_feats += torch.tensor(centers, dtype=float).T
        if 'gauss_exponents' in node_feats_from or 'gaussian_exponents' in node_feats_from:
            gauss_exponents = [ sum(g['exponents'] * g['front_factors']) / sum(g['front_factors'])
                                for g in mol.basis.gaussian_dict
                                for _ in range(mol.basis.ang_n_orbs(g['ang'])) ]
            node_feats += torch.tensor(gauss_exponents, dtype=float).unsqueeze(0)
        if 'onehot_sh' in node_feats_from:
            onehot_sh = ...
        return note_feats


class MolOnBasis4Graph_InMem(Dataset):
    """dataset from molecules' DMs"""

    def __init__(self, opts, device='cpu', path=None):
        print("Preparing all structures and data ...")
        self.device = device
        # self.path = path
        # self.opts = opts
        self.mol_opts = opts['molecules']
        self.data_opts = opts['data']
        self.oep_opts = opts['oep']
        self.mol_kwargs_list = [mol for mol in Gen_mol_kwargs(self.mol_opts)]
        self.n_mol = len(self.mol_kwargs_list)
        self.k = opts['data']['n_nearest_neighbor']
        # in mem
        self.mols = [None]*self.n_mol
        self.xs = [None]*self.n_mol
        self.edge_indices = [None]*self.n_mol
        self.es = [None]*self.n_mol
        self.bs = [None]*self.n_mol
        self.normalized = False
        self.norm_params = {}

    def __len__(self):
        return self.n_mol

    def __getitem__(self, i):
        # inputs
        if self.xs[i] is None or self.edge_indices[i] is None or self.es[i] is None:
            mol = self.get_mol(i)
            x, edge_index, e = self.gen_inputs(mol)
            x = torch.from_numpy(x).float().to(self.device)
            edge_index = torch.from_numpy(edge_index).long().to(self.device)
            e = torch.from_numpy(e).float().to(self.device)
            self.xs[i], self.edge_indices[i], self.es[i] = x, edge_index, e
        else: x, edge_index, e = self.xs[i], self.edge_indices[i], self.es[i]
        # targets
        if self.bs[i] is None:
            mol = self.get_mol(i)
            b = self.gen_target(mol)
            b = torch.from_numpy(b).float().to(self.device)
            self.bs[i] = b
        else: b = self.bs[i]
        return {'x':x, 'edge_index':edge_index, 'e':e, 'target':b}

    def get_mol(self, i):
        if self.mols[i] is None:
            mol = self.gen_mol(i)
            self.mols[i] = mol
        else: mol = self.mols[i]
        return mol
    def gen_mol(self, i):
        print("Generating Molecule with string:\n",self.mol_kwargs_list[i])
        mol = Molecule(**self.mol_kwargs_list[i])
        mol.generate_all_mats()
        return mol

    def gen_inputs(self, mol):
        node_feats_str = self.data_opts['node_features']
        node_feats_mats = [getattr(mol, M) for M in node_feats_str]
        x = np.stack([M.diagonal() for M in node_feats_mats], 1)
        sort_by_mat = np.abs(eval('mol.'+self.data_opts['sort_by_mat']))
        source = np.concatenate([np.flip(sort_by_mat[i].argsort())[:self.k] for i in range(len(sort_by_mat))])
        drain = np.concatenate([np.repeat(i, self.k) for i in range(len(sort_by_mat))])
        edge_index = np.concatenate((np.expand_dims(source, 0), np.expand_dims(drain, 0)), 0)        
        edge_feats_str = self.data_opts['edge_features']
        edge_feats_mats = [getattr(mol, M) for M in edge_feats_str]
        e = np.array([[M[pair[0],pair[1]] for M in edge_feats_mats] for pair in edge_index.T])
        return x, edge_index, e

    def gen_target(self, mol):
        return oep_search(mol.dm_ccsd, mol.H, mol.S, mol.n_elec, mol.integrals_3c1e, self.oep_opts['max_itr'], self.oep_opts['conv_crit'])
    
    def normalize_together(self):       
        for itm in ['xs', 'es', 'bs']:
            n_eles = sum(getattr(self, itm)[i].shape[0] for i in range(len(self)))
            tot_sum = torch.cat([getattr(self, itm)[i].sum(0, keepdim=True) for i in range(len(self))], 0).sum(0, keepdim=True)
            tot_sum2 = torch.cat([(getattr(self, itm)[i]**2).sum(0, keepdim=True) for i in range(len(self))], 0).sum(0, keepdim=True)
            mu, sigma = tot_sum/n_eles, tot_sum2/n_eles - tot_sum**2/n_eles**2
            self.norm_params[itm] = (mu, sigma)
            for i in range(len(self)):
                getattr(self, itm)[i] = (getattr(self, itm)[i] - mu) / torch.sqrt(sigma)
                # getattr(self, itm)[i] = getattr(self, itm)[i] / torch.sqrt(sigma)
        self.normalized = True
        
    def save_data(self, folder_path, fn=None):
        # if folder_path is None: folder_path = self.path
        self.to_device('cpu')
        if fn is None: fn = "dataset_MolOnBasis4Graph_InMem_%dmols.pkl"%(self.n_mol)
        fp = osp.join(folder_path, fn)
        for mol in self.mols:
            if hasattr(mol, '_pyscf_mol'): del mol._pyscf_mol
        with open(fp, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return fn
    def del_mols_n_save_data(self, folder_path, fn):
        if None in self.xs: raise Warning('not all xs calculated')
        if None in self.edge_indices: raise Warning('not all edge indices calculated') 
        if None in self.es: raise Warning('not all edge features calculated') 
        if None in self.bs: raise Warning('not all target calculated')
        if hasattr(self, 'mols'): del self.mols      
        self.save_data(folder_path, fn)

    def to_device(self, device):
        for x in self.xs: x = x.to(device)
        for edge_index in self.edge_indices: edge_index = x.to(device)
        for e in self.es: e = e.to(device)
        for b in self.bs: b = b.to(device)


def collate_fn_graph(data):
    x_batch = torch.cat([d['x'] for d in data], 0)    
    edge_index_batch = torch.cat([d['edge_index'] for d in data], 1)
    bg_node, bg_edge = 0, 0
    for i, d in enumerate(data):
        n_nodes = d['x'].shape[0]
        n_edges = d['edge_index'].shape[0]  
        edge_index_batch[bg_edge : bg_edge+n_edges] += bg_node
        bg_node += n_nodes
        bg_edge += n_edges
    e_batch = torch.cat([d['e'] for d in data], 0)
    b_batch = torch.cat([d['target'] for d in data], 0)
    # x_batch = torch.from_numpy(x_batch)
    # edge_index_batch = torch.from_numpy(edge_index_batch)
    # e_batch = torch.from_numpy(e_batch)
    # b_batch = torch.from_numpy(b_batch)
    return {'x': x_batch, 'edge_index': edge_index_batch, 'e': e_batch, 'target': b_batch}
   

# class MolOnBasis4Graph(Dataset):
#     """dataset of molecules DMs"""

#     def __init__(self, opts, path, device='cpu'):
#         print("Preparing all structures and data ...")
#         self.path = path
#         # self.opts = opts
#         self.mol_opts = opts['molecules']
#         self.device = device
#         self.mol_kwargs_list = [mol for mol in Gen_mol_kwargs(self.mol_opts)]
#         self.n_mol = len(self.mol_kwargs_list)
#         self.k = opts['data']['n_nearest_neighbor']
#         self.saved_mols = [None]*self.n_mol
#         self.saved_edge_indices = [None]*self.n_mol

#     def __len__(self):
#         return self.n_mol

#     def __getitem__(self, i):
#         # molecule
#         if self.saved_mols[i] is None:
#             mol = self.gen_mol(i)
#             self.save_mol(mol, i)
#         else: mol = load_mol(i)
#         # edge_indices
#         if self.saved_edge_indices[i] is None:
#             edge_index = gen_edge_index(mol)
#             self.saved_edge_indices[i] = ei_fn
#         else: edge_index = self.load_edge_index(i)
#         return {}


#     def gen_mol(self, i):
#         print("m",self.mol_kwargs_list[i])
#         mol = Molecule(**self.mol_kwargs_list[i])
#         mol.generate_all_mats()
#         return mol

#     def save_mol(self, mol, i):
#         mol_fn = mol.save(folder_path='')
#         self.saved_mols[i] = mol_fn

#     def load_mol(self, i):
#         mol_fn = self.saved_mols[i]
#         with open(osp.join(self.path, 'molecules', mol_fn), 'rb') as f:
#             mol = pickle.load(f)
#         return mol

#     def gen_edge_index(self, mol):
#         ...

#     def save_edge_index(self, edge_index, i):
#         ei_fn = mol.save(folder_path='')
#         self.saved_mols[i] = mol_fn

#     def load_edge_index(self, i):
#         mol_fn = self.saved_mols[i]
#         with open(osp.join(self.path, 'molecules', mol_fn), 'rb') as f:
#             mol = pickle.load(f)
#         return mol