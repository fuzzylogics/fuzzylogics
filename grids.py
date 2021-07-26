#%%
import numpy as np

class Grid():
    def __init__(self, level, coords, weights, sym_list=[]):
        self.level = level
        self.coords = coords
        self.weights = weights
        self.sym_list = sym_list
        self.sym_applied = []
        self.pruned = False
        
    @property
    def level(self):
        return self._level
    @level.setter
    def level(self, level):
        if level not in range(1,10):
            raise Exception("Invalid grid level ", level)
        self._level = level

    @property
    def coords(self):
        return self._coords
    @coords.setter
    def coords(self, coords):
        self._coords = coords

    @property
    def weights(self):
        return self._weights
    @weights.setter
    def weights(self, weights):
        self._weights = weights


    def gen_edge(self, k):
        assert self.sym_applied == [], "edges should be calculated before pruning"
        assert len(self.coords.shape) == 2 and self.coords.shape[1] == 3
        assert k <= self.coords.shape[0]
        self.nnn = k
        import torch
        from torch_geometric.nn import knn_graph
        self.edge_idx = knn_graph(torch.from_numpy(self.coords), k, loop=False).numpy()
        self.edge_vec = self.coords[self.edge_idx[1]] - self.coords[self.edge_idx[0]]
        return self.edge_idx, self.edge_vec

        
    def fold_towards(self, side='x+'):
        coords = self.coords
        if side is None or side=='':
            return coords
        elif side == 'x+':
            coords[...,0] = np.abs(coords[...,0])
        elif side == 'x-':
            coords[...,0] = -np.abs(coords[...,0])
        elif side == 'y+':
            coords[...,1] = np.abs(coords[...,1])
        elif side == 'y-':
            coords[...,1] = -np.abs(coords[...,1])
        elif side == 'z+':
            coords[...,2] = np.abs(coords[...,2])
        elif side == 'z-':
            coords[...,2] = -np.abs(coords[...,2])
        else:
            raise NotImplementedError
        self.coords = coords
        return coords

    def fan_along(self, axis=None):
        coords = self.coords
        # print(coords)
        if axis is None or axis=='':
            print('no fanning, grid.coords unchanged.')
            return coords
        rot_ax = {'x':0, 'y':1, 'z':2}[axis]
        all_ax = [0,1,2]
        ox = all_ax[rot_ax+1:]+all_ax[:rot_ax]
        # print(ox)
        coords[...,ox[0]] = np.sqrt(np.sum(coords[...,ox]**2, -1))
        coords[...,ox[1]] = 0.0
        # print(coords)
        self.coords = coords
        return coords

    def apply_symmetry(self, sym_list=None):
        if sym_list == None:
            sym_list = self.sym_list
        for sym in sym_list:
            plane_dict = {'yz':'x+', 'zy':'x-', 'zx':'y+', 'xz':'y-', 'xy':'z+', 'yx':'z-'}
            if sym in ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']:
                self.fold_towards(sym)
                self.sym_applied += [sym]
            elif sym in plane_dict:
                self.fold_towards(plane_dict[sym])
                self.sym_applied += [sym]
            elif sym in ['x', 'y', 'z']:
                self.fan_along(sym)
                self.sym_applied += [sym]
            else:
                raise NotImplementedError('symmetry only impemented for xyz axes and planes')

    def prune(self, crit=1.0e-10):
        assert self.sym_list == self.sym_applied
        coords = self.coords
        weights = self.weights
        assert coords.shape[-1] == 3
        coords = np.reshape(coords, (-1, coords.shape[-1]))
        all_prune_mapping_idx = np.array(range(len(coords)))
        for sort_by in [0]:
            n = len(coords)
            mapping_idx = np.array([*range(n)])         
            sort_priority = [*range(sort_by,3)]+[*range(0,sort_by)]
            sort_priority.reverse()  # lexsort uses last column as the first priority
            si = np.lexsort(coords[:,sort_priority].T)
            i = 0
            while i < n:
                j = i + 1
                while j < n and (np.abs(coords[si[j]] - coords[si[i]]) < crit).all():
                    j += 1
                weights[si[i]] += np.sum(weights[si[i+1:j]])
                weights[si[i+1:j]] = 0.0
                mapping_idx[si[i+1:j]] = mapping_idx[si[i]]
                i = j
            fill = np.array(weights, dtype=bool)
            coords = coords[fill]
            weights = weights[fill]
            pruned_range = np.zeros(n, dtype=int)
            pruned_range[fill] = np.array(range(len(coords)))
            mapping_idx = pruned_range[mapping_idx]
            all_prune_mapping_idx = mapping_idx[all_prune_mapping_idx]

        self.coords = coords
        self.weights = weights
        self.pruned = True
        self.prune_mapping_idx = all_prune_mapping_idx
        if getattr(self, 'edge_idx', None) is not None and getattr(self, 'edge_vec', None) is not None:
            source = self.edge_idx[0].reshape(-1, self.nnn)
            vec = self.edge_vec.reshape(-1, self.nnn, 3)
            _, inverse_mapping = np.unique(all_prune_mapping_idx, return_index=True)
            # print('s1',source)
            source = source[inverse_mapping]
            # print('s2',source)
            vec = vec[inverse_mapping]
            source = all_prune_mapping_idx[source].reshape(-1)
            # print(all_prune_mapping_idx)
            # print(inverse_mapping)
            # print('s3',source)
            vec = vec.reshape(-1, 3)
            drain = np.repeat(np.arange(len(coords)), self.nnn)
            self.edge_idx = np.stack((source, drain), 0)
            self.edge_vec = vec
        return coords, weights

    def mirror_n_duplicate(self, axis):
        ax = {'x':0,'yz':0,'zy':0, 'y':1,'zx':1,'xz':1, 'z':2,'xy':2,'yx':2}[axis]
        import copy
        mirror_coords = copy.deepcopy(self.coords)
        new_weights = copy.deepcopy(self.weights)
        mirror_coords[...,ax] = -mirror_coords[...,ax]
        new_weights = new_weights / 2.0
        self.coords = np.concatenate((self.coords, mirror_coords), 0)
        self.weights = np.concatenate((new_weights, new_weights), 0)
        return self.coords

    @property
    def id(self):
        # assert self.pruned == True
        if getattr(self, '_id', None) is None:
            digits = 10
            import hashlib
            nnn_str = str(self.nnn) if getattr(self, 'edge_idx', None) is not None else ''
            b = bytes(str(self.coords)+str(self.weights)+nnn_str, 'utf-8')
            encoded = hashlib.sha1(b)
            self._id = encoded.hexdigest()[:digits]
        return self._id

    def plot_coords(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        x, y, z = [tmp.squeeze(-1) for tmp in np.split(np.reshape(self.coords, (-1, 3)), 3, axis=-1)]
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter3D(x, y, z)

#%%

class SamplingCoords():
    
    def __init__(self, target, sampling_opts=None):
        from molecule import Molecule
        # print(sampling_opts)
        if isinstance(target, Grid):
            self.coords = target.coords
        elif isinstance(target, dict):
            self.coords = target['coords']
        elif isinstance(target, np.ndarray):
            self.coords = target
        elif isinstance(target, Molecule):
            self.coords = target.grid.coords
            self.discr = {'grid level': target.grid.level,
                          'molecule': str(target)}
        else: raise Exception("Invalid target for SamplingCoords")
        self.sampling_opts = sampling_opts
        # self.mesh_out()


    @property
    def meshed(self):
        if getattr(self, 'ori_coords', None) is None:
            return False
        else: True



    def gen_neighborhood_mesh(self, nb_opts):
        print("generating neighborhood mesh ...", end='')

        mesh = np.array([0])
        if 'shape' in nb_opts:
            if nb_opts['shape'] == 'cube':
                n = nb_opts['n_points']
                if 'cube_length' in nb_opts:
                    l = nb_opts['cube_length']
                elif 'step_length' in nb_opts:
                    l = nb_opts['step_length'] * (n - 1)   
                mesh = SamplingCoords.cube(l, n)
            
            elif nb_opts['shape'] == 'ball':
                ...

        if 'transform' in nb_opts: trans_opts = nb_opts['transform']
        else: trans_opts = {'apply':[]}

        # flipping (mirroring)
        if 'mirror' in trans_opts['apply'] and 'mirror' in trans_opts:
            mirror_opts = trans_opts['mirror']
            if 'axes' in mirror_opts and mirror_opts['axes']!='all':
                nr = len(mirror_opts['axes'])
                self.mirror_vecs = []
                for s in mirror_opts['axes']:
                    m_x, m_y, m_z = SamplingCoords.count_xyz(s)
                    self.mirror_vecs += np.ndarray(m_x, m_y, m_z)
            elif mirror_opts['axes'] == 'all':
                nr = 2**3 - 1
                all_vecs = np.array(np.meshgrid([0,1],[0,1],[0,1], indexing='ij'))
                all_vecs = all_vecs.transpose(1,2,3,0).reshape(-1,3)[1:]
                self.mirror_vecs = [v.squeeze() for v in np.split(all_vecs, nr, 0)]
            elif 'random_repeat' in mirror_opts:
                nr = min(mirror_opts['random_repeat'], 2**3 - 1)
                all_vecs = np.array(np.meshgrid([0,1],[0,1],[0,1], indexing='ij'))
                all_vecs = all_vecs.transpose(1,2,3,0).reshape(-1,3)[np.random.permutation(8)[:nr]]
                self.mirror_vecs = [v.squeeze() for v in np.split(all_vecs, nr, 0)]
            else: raise NotImplementedError
            mirrored_mesh = np.empty((nr,) + mesh.shape)    
            for i, v in enumerate(self.mirror_vecs):
                tmp_mesh = SamplingCoords.mirror(mesh, v)
                mirrored_mesh[i] = tmp_mesh
            if 'keep_original' in mirror_opts and mirror_opts['keep_original']:
                mesh = np.concatenate((np.expand_dims(mesh, 0), mirrored_mesh), 0)
            else: mesh = mirrored_mesh
        else:
            mesh = np.expand_dims(mesh, 0)

        # rotations   
        if 'rotation' in trans_opts['apply'] and 'rotation' in trans_opts:
            rot_opts = trans_opts['rotation']                           
            if 'vectors' in rot_opts:
                nr = len(rot_opts['vectors'])
                self.rot_vecs = [np.ndarray(v) for v in rot_opts['vectors']]
            elif 'random_repeat' in rot_opts:
                nr = rot_opts['random_repeat']                 
                rot_vecs = np.random.randn(nr, 3)
                rot_vecs = rot_vecs / np.sqrt(np.sum(rot_vecs**2, axis=1, keepdims=True))
                rot_vecs = 2.0 * np.pi * np.random.rand(nr, 1) * rot_vecs
                self.rot_vecs = [v.squeeze() for v in np.split(rot_vecs, nr, 0)]
            elif 'random_nAxis_nRepeat' in rot_opts:
                na, nr = rot_opts['random_nAxis_nRepeat']                 
                rot_vecs = np.random.randn(nr, na, 3)
                rot_vecs = rot_vecs / np.sqrt(np.sum(rot_vecs**2, axis=-1, keepdims=True))
                rot_vecs = 2.0 * np.pi * np.random.rand(nr, 1, 1) * rot_vecs
                self.rot_vecs = [v.squeeze() for v in np.split(rot_vecs, nr, 0)]
            else: raise NotImplementedError
            rotated_mesh = np.empty((nr,) + mesh.shape)
            for i, v in enumerate(self.rot_vecs):
                tmp_mesh = SamplingCoords.rotate(mesh, v)
                rotated_mesh[i] = tmp_mesh
            if 'keep_original' in rot_opts and rot_opts['keep_original']:
                mesh = np.concatenate((np.expand_dims(mesh, 0), rotated_mesh), 0)
            else: mesh = rotated_mesh
        else:
            mesh = np.expand_dims(mesh, 0) 
        
        self.mesh = mesh
        print('Done.')
        return mesh



    @staticmethod
    def cube(l, n):
        x = np.linspace(-l/2, l/2, n)
        y = np.linspace(-l/2, l/2, n)
        z = np.linspace(-l/2, l/2, n)
        x, y, z = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack((x, y, z), -1)
        # return coords.reshape(n*n*n, 3)
        return coords
    
    def rotate_coords(self, rot_vec):
        self.coords = SamplingCoords.rotate(self.coords, rot_vec)
    @staticmethod
    def rotate(coords, rot_vec):
        from scipy.spatial.transform import Rotation
        if rot_vec.shape == (3,) or rot_vec.shape == (1, 3) or rot_vec.shape == (3, 1):
            rot = Rotation.from_rotvec(rot_vec).as_matrix()
        elif rot_vec.shape[1] == 3 and len(rot_vec.shape) == 2:
            rot = np.eye(3)
            for v in rot_vec:
                rot = np.matmul(Rotation.from_rotvec(v).as_matrix(), rot)
        else: raise NotImplementedError
        return np.matmul(rot, np.expand_dims(coords, -1)).squeeze(-1)

    def mirror_coords(self, axis):
        self.coords = SamplingCoords.mirror(self.coords, axis)
    @staticmethod
    def mirror(coords, axis):
        new_coords = np.empty_like(coords)
        new_coords[:] = coords
        if isinstance(axis, str) and axis == 'x':
            new_coords[..., 0] = - coords[..., 0]
        elif isinstance(axis, str) and axis == 'y':
            new_coords[..., 1] = - coords[..., 1]
        elif isinstance(axis, str) and axis == 'z':
            new_coords[..., 2] = - coords[..., 2]
        elif isinstance(axis, np.ndarray) or isinstance(axis, tuple) or isinstance(axis, list):
            for i, flip in enumerate(axis):
                new_coords[..., i] = - (flip * 2 - 1) * coords[..., i]
        return new_coords


    def __len__(self):
        l = 1
        for n in self.coords.shape:
            l *= n
        return l

    def mesh_out(self):
        self.ori_coords = self.coords
        if getattr(self, 'coords', None) is not None:
            coords = self.coords
        elif getattr(self, 'ori_coords', None) is not None:
            coords = self.ori_coords
        else: raise Exception("mesh on what coords?")
        if getattr(self, 'mesh', None) is not None:
            mesh = self.mesh
        elif getattr(self, 'sampling_opts', None) is not None:
            if 'nb_opts' not in self.sampling_opts:
                self.sampling_opts['nb_opts'] = {}
            nb_opts = self.sampling_opts['nb_opts']
            mesh = self.gen_neighborhood_mesh(nb_opts)
        else: raise Exception("how to sample?")
        self.coords = SamplingCoords.outer(coords, mesh)
        return self.coords

    @staticmethod
    def outer(arr1, arr2):
        l1, l2 = len(arr1.shape), len(arr2.shape)
        arr1 = np.expand_dims(arr1, [i for i in range(l1-1, l1+l2-2)])
        arr2 = np.expand_dims(arr2, [i for i in range(l1-1)])
        return arr1 + arr2
       
    @property
    def id(self):
        if getattr(self, '_id', None) is None:
            digits = 10
            import hashlib
            b = bytes(str(self.ori_coords)+str(self.mesh), 'utf-8')
            encoded = hashlib.sha1(b)
            self._id = encoded.hexdigest()[:digits]
        return self._id


    def plot_coords(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        x, y, z = [tmp.squeeze(-1) for tmp in np.split(np.reshape(self.coords, (-1, 3)), 3, axis=-1)]
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter3D(x, y, z)


# %%

if __name__ == '__main__':
    from molecule import Molecule
#     mol1 = Molecule(linear_dists=[0.7], linear_atoms=['H','H'], charge=0, spin=0, basis_name='aug-cc-pvqz')
#     print(str(mol1))
#     mol1.grid = 1
#     print(mol1._grid.coords.shape)
    
#     from options import read_options_yaml
#     opts = read_options_yaml('options.yaml')

#     c = SamplingCoords(np.array([[1,2,3], [0,0,0]]), opts['sampling'])
#     c.plot_coords()

#     c.mesh_out()
#     c.plot_coords()


    # import gen_molecules
    # eql_hch = 116.133 / 180 * np.pi
    # eql_ch = 1.111
    # eql_co = 1.205
    # mol_ch2o = Molecule(struc_dict = gen_molecules.gen_sym_formaldehyde_struc_dict(eql_hch, eql_ch, eql_co), 
    #                     basis_name='aug-cc-pvdz')
    # mol_ch2o.build_grid(grid_level=3, sym_list=[])
    # # print(sum([mol_ch2o._grid.coords[i,0]<1.0e-8 and mol_ch2o._grid.coords[i,0]>-1.0e-8 for i in range(len(mol_ch2o._grid.coords))]) )
    # mol_ch2o._grid.plot_coords()
    # dft_rho = mol_ch2o.rho('dft')
    # hf_rho = mol_ch2o.rho('hf')
    # oep_rho = mol_ch2o.rho('oep')
    # ccsd_rho = mol_ch2o.rho('ccsd')
    # oep_ccsd_diff = oep_rho - ccsd_rho
    # print(oep_ccsd_diff.shape)
    # hf_ccsd_diff = hf_rho - ccsd_rho
    # dft_ccsd_diff = dft_rho - ccsd_rho
    # print('max_err_oep:', np.abs(oep_ccsd_diff).max())
    # print('max_err_hf:', np.abs(hf_ccsd_diff).max())
    # print('max_err_dft:', np.abs(dft_ccsd_diff).max())
    # print('max_hf_dft:', np.abs(dft_rho-hf_rho).max())

    grid = Grid(1, np.array([[0,0,0], [-1,0,0], [1,0,0], [0,0,1]],dtype=float), np.array([1.0,1.0,1.0,1.0]), sym_list=['x+'])
    # grid.gen_edge_idx(2)
    grid.gen_edge(2)
    print(grid.coords)
    print(grid.edge_idx)
    print(grid.edge_vec)
    grid.apply_symmetry()
    grid.prune()
    print(grid.coords)
    print(grid.edge_idx)
    print(grid.edge_vec)

#%%
    # from find_coords import find_idx
    
    # import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # # ix, x = find_idx(mol_ch2o.grid_coords,0)
    # # axes[0].plot(x, oep_ccsd_diff[0][ix])
    # # axes[0].plot(x, hf_ccsd_diff[0][ix])
    # # axes[0].plot(x, dft_ccsd_diff[0][ix])

    # iy, y = find_idx(mol_ch2o._grid.coords, 1)
    # ax.plot(y, oep_ccsd_diff[0][iy])
    # ax.plot(y, hf_ccsd_diff[0][iy])
    # ax.plot(y, dft_ccsd_diff[0][iy])
    # plt.xlim([-4, 2])
    # ax.legend(['oep', 'hf', 'b3lyp'])
    # fig.savefig('ch2o_oep_hf_b3lyp.png', dpi=300)

# %%
