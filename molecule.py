import os.path as osp
import re
import pyscf
import pyscf.cc
import scipy
import numpy as np
import torch
import h5py
import time
from opt_einsum import contract
from grids import Grid, SamplingCoords
import pickle
import os.path as osp

import ks

A2Bohr = 1.8897261246258
CPU_MEM = 4.0e10
GPU_MEM = 2.0e10


class Molecule():
    def __init__(self, struc_dict=None, struc_path=None, 
                 linear_dists=None, linear_atoms=None, 
                 charge=0, spin=0, basis_name='aug-cc-pvdz',
                 custom_description=''):
        if struc_path is not None:
            self.read_structure(struc_path)
            self.struc_path = struc_path
        elif struc_dict is not None:
            self.read_structure_dict(struc_dict)
        elif linear_dists is not None and linear_atoms is not None:
            self.generate_linear_molecule(linear_dists, linear_atoms, charge, spin)
        # self.basis_name = re.sub('[_.-]', '', basis_name).lower()
        self.basis_name = basis_name
        self.basis = self.generate_basis_functions(self.basis_name)
        # self.generate_fixed_matrices()

        self.custom_description = custom_description

    @property
    def struc_path(self):
        return self._struc_path
    @struc_path.setter
    def struc_path(self, struc_path):
        self._struc_path = struc_path

    

    def generate_linear_molecule(self, linear_dists, linear_atoms, charge, spin):
        n = len(linear_atoms)
        assert len(linear_dists) == n-1
        self.n_atom = n
        self._atoms_list = linear_atoms
        self._atoms_coords = []
        self._atoms_str = ""
        z = -sum(linear_dists)/2
        for i in range(n):
            self._atoms_coords.append(tuple([0.0, 0.0, A2Bohr*float(z)]))
            self._atoms_str += "%s %16.8e %16.8e %16.8e; " % (linear_atoms[i], 0.0, 0.0, float(z))
            if i < n-1: z += linear_dists[i]
        self.charge = charge
        self.spin = spin

    # IOs _______________________________________________________________
    def read_structure(self, fn):
        """
        str file uses a format similar to .xyz file.

        Line 1      number of atoms
        Line 2      atom    x   y   z   [AA]
        Line 3      atom    x   y   z   [AA]
        ...                             
        Line N      atom    x   y   z   [AA]
        Line N+1    charge  spin    

        The `spin' follows the definition in PySCF, 
        equalling to N(alpha)-N(beta)
        The last line is OPTIONAL.
        If not provided, the default value is 0, 0
        """
        atoms_list = []
        atoms_coords = []
        atoms_str = ""
        with open(fn, 'r') as fp:
            n = int(fp.readline())  # reading first line
            for i in range(n):  # reading each atom
                ss = fp.readline().split()
                atoms_list.append(ss[0])
                atoms_coords.append(tuple([A2Bohr*float(ss[1]), A2Bohr*float(ss[2]), A2Bohr*float(ss[3])]))
                atoms_str += "%s %16.8e %16.8e %16.8e; " % (
                    ss[0], float(ss[1]), float(ss[2]), float(ss[3]))
            ss = fp.readline().split()  # reading last line
            if len(ss) > 0:
                charge, spin = int(ss[0]), int(ss[1])
            else:
                charge, spin = 0, 0
        self.n_atom = n
        self._atoms_list = atoms_list
        self._atoms_coords = atoms_coords
        self._atoms_str = atoms_str
        self.charge = charge
        self.spin = spin


    def read_structure_dict(self, struc_dict):       
        if 'atoms' in struc_dict and struc_dict['atoms'] is not None:   
            atoms_list = []
            atoms_coords = []
            atoms_str =""
            for i, atom in enumerate(struc_dict['atoms']):
                if type(atom) is str: ss = atom.split()
                elif type(atom) is list: ss = atom
                atoms_list.append(ss[0])
                atoms_coords.append(tuple([A2Bohr*float(ss[1]), A2Bohr*float(ss[2]), A2Bohr*float(ss[3])]))
                atoms_str += "%s %16.8e %16.8e %16.8e; " % (
                    ss[0], float(ss[1]), float(ss[2]), float(ss[3]))
            self.n_atoms = len(struc_dict['atoms'])
            self._atoms_list = atoms_list
            self._atoms_coords = atoms_coords
            self._atoms_str = atoms_str
        else: raise NotImplementedError()
        self.charge = struc_dict['charge']
        self.spin = struc_dict['spin']
        

    def write_structure(self, fn):
        with open(fn, 'w') as fp:
            if getattr(self, '_atoms_string', None) is not None:
                n = len(self._atoms_str)
                fp.write(str(n)+"\n")   # write n mol
                fp.write(re.sub(';', '\n', self._atoms_str))
            elif getattr(self, '_atoms_list', None) is not None and getattr(self, '_atoms_coords', None) is not None:
                n = len(self._atoms_list)
                fp.write(str(n)+"\n")   # write n mol
                for i in range(n):
                    fp.write("%s %16.8e %16.8e %16.8e\n" % (
                        self._atoms_list[i], 
                        self._atoms_coords[i][0]/A2Bohr, 
                        self._atoms_coords[i][1]/A2Bohr, 
                        self._atoms_coords[i][2]/A2Bohr))
            else:
                raise Exception("NO valid structure to write.")
            fp.write("%d\t%d" % (self.charge, self.spin))
        self.struc_path = fn
    #IOs_end___________________________________________________________

    @property
    def atoms_str(self):
        if getattr(self, '_atoms_str', None) is None:
            self._atoms_str = ""
            for i in range(self.n_atom):
                self._atoms_str += "%s %16.8e %16.8e %16.8e; " % (
                    self._atoms_list[i], 
                    self._atoms_coords[i][0]/A2Bohr, 
                    self._atoms_coords[i][1]/A2Bohr, 
                    self._atoms_coords[i][2]/A2Bohr)
        return self._atoms_str

    def calculate_n_elec(self):
        atom_number_dict = {
            'H':1,
            'He':2,
            'C':6,
            'N':7,
            'O':8,
            'F':9,
        }
        n = sum(atom_number_dict[atom] for atom in self._atoms_list)
        n = n + self.charge
        self._n_elec = n
        return n
    @property
    def n_elec(self):
        if getattr(self, '_n_elec', None) is None:
            self.calculate_n_elec()
        return self._n_elec

    @property
    def pyscf_mol(self):
        if getattr(self, '_pyscf_mol', None) is not None:
            return self._pyscf_mol
        else:
            return self.generate_pyscf_molecule()


    def generate_pyscf_molecule(self, save_attr=True):
        pyscf_mol = pyscf.gto.M(
                atom=self._atoms_str, 
                basis=self.basis_name, 
                charge=self.charge, 
                spin=self.spin)    
        n_elec = self.calculate_n_elec()
        n_ao = pyscf_mol.nao_nr()
        self.mo_occ = np.zeros(n_ao)
        self.mo_occ[:n_elec//2] = 2
        self.mo_occ[n_elec//2] = 0 if n_elec%2==0 else 1
        self.n_occ = n_elec//2 if n_elec%2==0 else n_elec//2+1
        if save_attr:
            self._pyscf_mol = pyscf_mol
        return pyscf_mol

    @property
    def grid(self):
        return self._grid
    @grid.setter
    def grid(self, grid_level):
        pyscf_mol = self.pyscf_mol
        dft = pyscf.scf.RKS(pyscf_mol)
        dft.xc = 'b3lypg'
        dft.grids.level = grid_level
        dft.grids.build()
        self._grid = Grid(level=grid_level, coords=dft.grids.coords, weights=dft.grids.weights)
    @property
    def grid_coords(self):
        return self.grid.coords

    def build_grid(self, grid_level, sym_list, gen_edge=True, nnn=10):
        print('nnn___________', nnn)
        pyscf_mol = self.pyscf_mol
        dft = pyscf.scf.RKS(pyscf_mol)
        dft.xc = 'b3lypg'
        dft.grids.level = grid_level
        dft.grids.build()
        self._grid = Grid(level=grid_level, coords=dft.grids.coords, weights=dft.grids.weights, sym_list=sym_list)
        if gen_edge: self.grid.gen_edge(nnn)
        self.grid.apply_symmetry()
        self.grid.prune()

    def generate_all(self, grid_level, grid_sym_list, gen_edge=True, nnn=10):
        self.build_grid(grid_level=grid_level, sym_list=grid_sym_list, gen_edge=gen_edge, nnn=nnn)
        # self.grid = grid_level
        self.generate_all_mats()
        self.phi

    def generate_all_mats(self):
        self.generate_fixed_matrices()
        self.dm_dft
        self.dm_hf
        # self.dm_hartree
        self.dm_ccsd
        self.id
        self.itg_2e


    def save(self, folder_path=None, fn=None):
        t0 = time.time()
        if folder_path is None: folder_path = './molecules'
        if fn is None:
            if getattr(self, '_grid', None) is None: fn = "mol_%s_grid_None.pkl"%(self.id)
            else: fn = "mol_%s_grid_%s.pkl"%(self.id, self._grid.id)
        fp = osp.join(folder_path, fn)
        had_pyscf_mol = hasattr(self, '_pyscf_mol')
        if had_pyscf_mol:
            pyscf_mol = self._pyscf_mol
            del self._pyscf_mol
        with open(fp, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        t1 = time.time()
        # print("Time for pickle saving: ", t1-t0)
        if had_pyscf_mol: self._pyscf_mol = pyscf_mol
        return fn

    def generate_fixed_matrices(self):
        dft_hf = pyscf.scf.RKS(self.pyscf_mol)
        dft_hf.xc = 'hf,'
        dft_hf.kernel()
        dm_dft_hf = dft_hf.make_rdm1()
        self._v_hf_fixed = pyscf.dft.rks.get_veff(dft_hf, self.pyscf_mol, dm_dft_hf)
        self._S = self.pyscf_mol.intor('int1e_ovlp_sph')
        self._T = self.pyscf_mol.intor('cint1e_kin_sph')
        self._vn = self.pyscf_mol.intor('cint1e_nuc_sph')
        self._integrals_3c1e = self.pyscf_mol.intor('int3c1e_sph')
        self._H = self._T + self._vn + self._v_hf_fixed
    @property
    def S(self):
        if getattr(self, '_S', None) is None:
            self.generate_fixed_matrices()
        return self._S
    @property
    def T(self):
        if getattr(self, '_T', None) is None:
            self.generate_fixed_matrices()
        return self._T
    @property
    def vn(self):
        if getattr(self, '_vn', None) is None:
            self.generate_fixed_matrices()
        return self._vn
    @property
    def integrals_3c1e(self):
        if getattr(self, '_integrals_3c1e', None) is None:
            self.generate_fixed_matrices()
        return self._integrals_3c1e
    @property
    def H(self):
        if getattr(self, '_H', None) is None:
            self.generate_fixed_matrices()
        return self._H

    @property
    def itg_2e(self):
        if getattr(self, '_itg_2e', None) is None:
            self._itg_2e = self.pyscf_mol.intor('int2e_sph')
        return self._itg_2e


    @property
    def phi(self):
        if getattr(self, '_phi', None) is None:
            self._phi = self.basis.on_grid_w_numpy(self.grid.coords, deriv=0)
        return self._phi

    def dm_from_Vxc_mat(self, Vxc):
        _, _, dm = Molecule.solve_KS(self.H + Vxc, self.S, self.mo_occ)
        return dm 
    @staticmethod
    def solve_KS(F, S, mo_occ):
        e, C = scipy.linalg.eigh(F, S)
        mocc = C[:, mo_occ > 0]
        dm = np.dot(mocc * mo_occ[mo_occ > 0], mocc.T.conj())
        return e, C, dm

    @staticmethod
    def solve_KS_torch(F, S, mo_occ):
        assert F.device == S.device
        device = F.device
        # F, S, are torch tensors
        occ_cutoff = 1.0e-8
        s_cutoff = 1.0e-8
        s, U = torch.symeig(S, eigenvectors=True)
        s, U = s[s>s_cutoff], U[:,s>s_cutoff]
        X = U * (1.0/(s.sqrt())).unsqueeze(0)
        Fp = contract('ji,jk,kl->il', X, F, X)
        e, Cp = torch.symeig(Fp, eigenvectors=True)
        C = contract('ij,jk->ik', X, Cp)
        if type(mo_occ)==list: mo_occ = torch.tensor(mo_occ).float().to(device)
        elif type(mo_occ)==np.ndarray: mo_occ = torch.from_numpy(mo_occ).float().to(device)
        assert (mo_occ>occ_cutoff).sum().item() <= C.shape[1]
        C_occ = C[:,(mo_occ>occ_cutoff)[:C.shape[1]]]
        occn = mo_occ[(mo_occ>occ_cutoff)].unsqueeze(0)
        dm = contract('ij,kj->ik', C_occ*occn, C_occ)
        return e, C, dm



    
    @property
    def dm_hf(self):
        if getattr(self, '_dm_hf', None) is None:
            hf = pyscf.scf.RHF(self.pyscf_mol)
            hf.kernel()
            self._dm_hf = hf.make_rdm1()
        return self._dm_hf
    @property
    def dm_hartree(self):
        if getattr(self, '_dm_hartree', None) is None:
            self._dm_hartree = self.dm_scf(self.gen_hartree_H)
        return self._dm_hartree
    @property
    def dm_fixed_H(self):
        if getattr(self, '_dm_fixed_H', None) is None:
            _,_,self._dm_fixed_H = self.solve_KS(self.H, self.S, self.mo_occ)
        return self._dm_fixed_H

    def gen_hf_H(self, dm):
        J = contract('ijkl, lk-> ij', self.itg_2e, dm)
        K = contract('ijkl, jk-> il', self.itg_2e, dm)
        H = self.T + self.vn + J - K/2.0
        return H
    def gen_hartree_H(self, dm):
        J = contract('ijkl, lk-> ij', self.itg_2e, dm)
        K = contract('ijkl, jk-> il', self.itg_2e, dm)
        H = self.T + self.vn + J * (self.n_elec - 1) / self.n_elec
        return H
    def dm_scf(self, H_func):
        crit = 1.0e-8
        mixing = 0.2
        _, _, dm_before = self.solve_KS(self.T+self.vn, self.S, self.mo_occ)
        while True:
            # H = self.gen_hf_H(dm_before)
            H = H_func(dm_before)
            _, _, dm_after = self.solve_KS(H, self.S, self.mo_occ)
            err = np.max(np.abs(dm_after - dm_before))
            print(err)
            if err < crit: break
            dm_before = (1.0 - mixing) * dm_after + mixing * dm_before
        return dm_after

    def dm_scf_ml(self, model, sampling_coords, dm_init, device=0, input_shape='vanilla'):
        crit = 1.0e-6
        init_mixing = 0.1
        if dm_init is not None:
            dm_before = dm_init
        else:
            _, _, dm_before = self.solve_KS(self.H, self.S, self.mo_occ)
        while True:
            dm_after = self.dm_ml(model, sampling_coords, dm_before, device, input_shape)
            # dm_err = np.max(np.abs(dm_after - dm_before))
            dm_err = (1.0/np.prod(dm_after.shape)) * np.sqrt(np.sum((dm_after-dm_before)**2))
            if dm_err < crit: break
            # elif dm_err < crit*1.0e+1: mixing = init_mixing + (1.0 - init_mixing)*0.9
            # elif dm_err < crit*1.0e+2: mixing = init_mixing + (1.0 - init_mixing)*0.5
            # elif dm_err < crit*1.0e+3: mixing = init_mixing + (1.0 - init_mixing)*0.5
            # elif dm_err < crit*1.0e+4: mixing = init_mixing + (1.0 - init_mixing)*0.5
            # elif dm_err < crit*1.0e+5: mixing = init_mixing + (1.0 - init_mixing)*0.5
            # elif dm_err < crit*1.0e+6: mixing = init_mixing + (1.0 - init_mixing)*0.5
            elif dm_err < crit*1.0e+1: mixing = init_mixing + (1.0 - init_mixing)*0.9
            elif dm_err < crit*1.0e+3: mixing = init_mixing + (1.0 - init_mixing)*0.5
            else: mixing = init_mixing
            print("dm error =%1.6e,\tnext_mixing = %1.3f"%(dm_err, mixing))
            dm_before = (1.0 - mixing) * dm_after + mixing * dm_before
        return dm_after
    def dm_ml(self, model, sampling_coords, dm, device, input_shape):       
        x = self.density_on_grid(dm, grid_coords=sampling_coords,
                                 deriv=(0, 1), package='torch', device=device)
        x = x.transpose(1,0)  # dim0=4=1+3deriv, dim1=grid_size
        if input_shape == 'vanilla': x = x.squeeze(2).squeeze(2)
        else: raise NotImplementedError
        x = x.contiguous().to(device)
        y = model(x).mean(1)
        vxc = y.detach()
        vxc_mat = self.V_mat(vxc, package='torch', device=device) 
        dm_new = self.dm_from_Vxc_mat(vxc_mat)
        return dm_new




    @property
    def dm_dft(self):
        if getattr(self, '_dm_dft', None) is None:
            dft = pyscf.scf.RKS(self.pyscf_mol)
            dft.xc = 'b3lyp'
            dft.kernel()
            self._dm_dft = dft.make_rdm1()
        return self._dm_dft

    @property
    def dm_ccsd(self):
        if getattr(self, '_dm_ccsd', None) is None:
            hf = pyscf.scf.RHF(self.pyscf_mol)
            hf.kernel()
            if getattr(self, '_dm_hf', None) is None:
                self._dm_hf = hf.make_rdm1()
            C = hf.mo_coeff
            ccsd = pyscf.cc.CCSD(hf)
            ecc, t1, t2 = ccsd.kernel()
            rdm1_hfmo = ccsd.make_rdm1()
            rdm1_ao = contract('pi,ij,qj->pq', C, rdm1_hfmo, C.conj())
            self._dm_ccsd = rdm1_ao
        return self._dm_ccsd
    @property
    def rho_ccsd(self):
        if getattr(self, '_rho_ccsd', None) is None:
            # self._rho_ccsd = self.rho('ccsd', package='torch', device='cuda').cpu().numpy().squeeze(0)
            self._rho_ccsd = self.rho('ccsd', package='numpy', device='cpu').squeeze(0)
        return self._rho_ccsd

    @property
    def oep_b(self):
        if getattr(self, '_oep_b', None) is None:
            from simple_wy import oep_search
            b = oep_search(self.dm_ccsd, self.H, self.S, self.n_elec, self.integrals_3c1e, 1000, 1.0e-10)
            self._oep_b = b
        return self._oep_b
    def oep_on_grid(self):
        return contract('i...,i->...', self.phi, self.oep_b)
    @property
    def Vmat_oep(self):
        if getattr(self, '_Vmat_oep', None) is None:       
            self._Vmat_oep = contract('ijk, k->ij', self.integrals_3c1e, self.oep_b)
        return self._Vmat_oep
    @property
    def dm_oep(self):
        if getattr(self, '_dm_oep', None) is None:
            self._dm_oep = self.dm_from_Vxc_mat(self.Vmat_oep)
        return self._dm_oep
    


    def rho(self, method, grid=None, deriv=0, package='numpy', device='cpu'):
        # if isinstance(grid, Grid):
        #     grid_coords = grid.coords
        # elif isinstance(grid, dict):
        #     grid_coords = grid.coords
        # elif isinstance(grid, SamplingCoords):
        #     grid_coords = grid.coords
        # elif isinstance(grid, np.ndarray) or isinstance(grid, torch.tensor):
        #     grid_coords = grid
        # elif grid is None:
            # grid_coords = self.grid.coords
        if getattr(self, 'basis', None) is None:
            self.generate_basis_functions()

        if method == 'hf' or method == 'rhf': dm = self.dm_hf
        elif method == 'hartree': dm = self.dm_hartree
        elif method == 'hf_fixed' or method == 'fixed_hf': dm = self.dm_from_Vxc_mat(0.0)
        elif method == 'dft' or method == 'b3lyp': dm = self.dm_dft
        elif method == 'ccsd': dm = self.dm_ccsd
        elif method == 'oep': dm = self.dm_oep
        else: raise NotImplementedError       
        # print('method',method, dm)
        # if package == 'numpy': device = 'cpu'
        # if package == 'torch': device = 'cuda'       
        return self.density_on_grid(dm, grid_coords=grid, deriv=deriv, package=package, device=device)


    def rho_n_save(self, method, grid, deriv=(0,1), folder_path=None, fn=None, package='numpy', device='cpu'):
        assert isinstance(grid, SamplingCoords)
        t0 = time.time()
        # print("Calculating density on sampling coordinates ...", end='')
        rho = self.rho(method, grid=grid, deriv=deriv, package=package, device=device)
        t1 = time.time()
        # print("Done.\ttime spent: %4.5f"%(t1-t0))
        
        # print("Saving density on sampling coordinates ...", end='')
        if folder_path is None: folder_path = './data_rho'
        if package == 'torch': rho_save = rho.cpu().numpy()
        if fn is None: fn = "mol_%s_grid_%s_dtype_%d.pkl"%(self.id, grid.id, 8*rho_save.itemsize)
        fp = osp.join(folder_path, fn)
        with open(fp, 'wb') as f:
            pickle.dump(rho_save, f, protocol=pickle.HIGHEST_PROTOCOL)
        t2 = time.time()
        # print("Done.\ttime spent: %4.5f"%(t2-t1))
        return rho, fn
    

    def V_mat(self, v, grid=None, coords=None, weights=None, package='numpy', device='cpu', save_phi_in_mem=False, save_phi_to_disk=False):
        # print("--------------------Integrating V matrix ... ----------------------")
        # if package == 'torch': device = 'cuda'
        if grid is not None:
            if isinstance(grid, Grid): coords, weights = grid.coords, grid.weights
            if isinstance(grid, dict): coords, weights = grid.coords, grid['weights']
        if coords is None and weights is None:
            coords, weights = self.grid.coords, self.grid.weights
        phi_shape = (self.basis.n_orbs,1)+tuple(weights.shape)
        time0 = time.time()
        # memory estimation for batching
        if device == 'cpu': mem, ele_size = CPU_MEM, 8
        else: mem, ele_size = GPU_MEM, 4
        mem_size_phi = ele_size * np.prod(weights.shape) * self.basis.n_orbs  
        n_batch_phi = int(np.ceil(mem_size_phi / (mem/2))) # half mem for phi, half for einsum
        n_batch_einsum = int(np.ceil(mem_size_phi*self.basis.n_orbs/30 / (mem/2)))
        batch_size_phi = int(weights.shape[0] // n_batch_phi)
        batch_size_einsum = int(weights.shape[0] // n_batch_einsum)
        assert min(batch_size_phi, batch_size_einsum) > 0, "grid first dim not enough for batching ..."
        if getattr(self, '_phi', None) is not None:
            phi_in_mem = self._phi
            phi_from_mem = True
        else: phi_from_mem = False
        if save_phi_in_mem:
            assert mem_size_phi < CPU_MEM / 2
            if getattr(self, '_phi', None) is None:
                self._phi = np.empty(phi_shape)
            else: save_phi_in_mem = False
        # read or write binary file
        bit = 32 if package=='torch' else 64
        bfn = "./binaries/phi_deriv0_%dbit_mol_%s.hdf5"%(bit, str(self.id))
        if osp.isfile(bfn) and not phi_from_mem:
            # print("reading hdf5 ... ")
            fr = h5py.File(bfn, 'r')
            reading, writting = True, False
            h5data_r = fr['phi']
        elif save_phi_to_disk and not osp.isfile(bfn):
            # print("writing hdf5 ... ")
            fw = h5py.File(bfn, 'w')
            reading, writting = False, True
            h5_dtype = 'f8' if device=='cpu' else 'f4'
            h5data_w = fw.create_dataset('phi', phi_shape, dtype=h5_dtype)
        else: reading, writting = False, False
        # accumulate V matrix
        V = np.zeros((self.basis.n_orbs, self.basis.n_orbs))
        if package == 'torch': weights = torch.from_numpy(weights).float().to(device)
        for i in range(0, coords.shape[0], batch_size_phi):
            bg, ed = i, min(i+batch_size_phi, coords.shape[0])
            # print('%s phi(r), grid points: %d - %d ...'%('Loading' if reading else 'Getting/Calculating', bg, ed))
            tt0 = time.time()
            if phi_from_mem:
                phi = phi_in_mem[:, :, bg:ed]
                if package == 'torch': phi = torch.from_numpy(phi).float().to(device)
            elif reading:
                phi = h5data_r[:, :, bg:ed]
                if package == 'torch': phi = torch.from_numpy(phi).float().to(device)
            else:
                if package == 'numpy': phi = self.basis.on_grid_w_numpy(coords[bg:ed], deriv=0)
                if package == 'torch': phi = self.basis.on_grid_w_torch(coords[bg:ed], deriv=0, device=device)
            if save_phi_in_mem: 
                if package == 'torch': phi_save = phi.to('cpu').numpy()
                self._phi[:, :, bg:ed] = phi_save
            if writting: h5data_w[:, :, bg:ed] = phi.to('cpu') if package=='torch' else phi
            tt1 = time.time()
            for j in range(i, min(i+batch_size_phi, coords.shape[0]), batch_size_einsum):
                bg, ed = j, min(j+batch_size_einsum, i+batch_size_phi, coords.shape[0])
                # print('Integrating grid points: %d - %d ...'%(bg, ed))
                tmp_mat = contract('i..., j..., ..., ... -> ij', phi[:,0,bg-i:ed-i], phi[:,0,bg-i:ed-i], v[bg-i:ed-i], weights[bg-i:ed-i]) # for dim1=1:4, d(phi) * dm * phi
                if package == 'torch': tmp_mat = tmp_mat.to('cpu').numpy()
                V += tmp_mat
            tt2 = time.time()
            # print("Time for %s phi = "%('LOADING' if reading else 'GETTING/CALULATING'), tt1 - tt0)
            # print("Time for integrating = ", tt2 - tt1)
        if reading: fr.close()
        if writting: fw.close()
        time1 = time.time()
        # print("Time for calculating V Matrix = ", time1 - time0)
        return V



    def generate_basis_functions(self, basis_name, save_attr=True):
        from basis import BasisSet
        basis = BasisSet()
        this_file = osp.abspath(__file__)
        this_dir = osp.dirname(this_file)
        atom_orbital_idx = [0]
        atom_gaussian_idx = [0]
        for i, atom in enumerate(self._atoms_list):
            atom_basis = BasisSet(filepath=osp.join(this_dir,"basis_files", self.basis_name+".1.gbs"),
                                  atom=atom, centers=self._atoms_coords[i])
            basis += atom_basis
            atom_orbital_idx += [atom_orbital_idx[-1] + atom_basis.n_orbs]
            atom_gaussian_idx += [atom_gaussian_idx[-1] + atom_basis.n_g]
        if save_attr: self.basis = basis
        return basis


    def density_on_grid(self, dm, grid_coords=None, deriv=0, package='numpy', device='cpu', save_phi_to_disk=False):
        # print("------------------Calculating denisty on grid ... -----------------")
        if grid_coords is None:
            coords = self.grid.coords
            grid_id = 'level'+str(self.grid.level)
            # save_phi_in_mem = True
            save_phi_in_mem = False
        elif isinstance(grid_coords, SamplingCoords):
            coords = grid_coords.coords
            grid_id = str(grid_coords.id)
            save_phi_in_mem = False
        # elif isinstance(grid_coords, Grid):
        #     coords = grid_coords.coords
        #     # grid_id = 'level'+str(grid_coords.level)
        #     save_phi_to_disk = False
        #     save_phi_in_mem = False
        # elif isinstance(grid_coords, dict):
        #     coords = grid_coords['coords']
        #     # grid_id = 'level'+str(grid_coords['level'])
        #     save_phi_to_disk = False
        #     save_phi_in_mem = False
        # elif isinstance(grid_coords, np.ndarray):
        #     coords = grid_coords
        #     save_phi_in_mem = False
        #     save_phi_to_disk = False
        else: raise Exception("Invalid grid_coords")
        assert self.basis.n_orbs == len(dm)
        if type(deriv)==int: deriv = (deriv,)
        from scipy.special import factorial
        deriv_dim = int(sum(factorial(i+2)/factorial(i)/factorial(2) for i in deriv))
        time0 = time.time()
        if package == 'numpy':
            rho = np.empty((deriv_dim,) + coords.shape[:-1])
            mem_size_phi = 8 * self.basis.n_orbs * int(np.prod(rho.shape))
            mem = CPU_MEM
        elif package == 'torch':
            rho = torch.empty((deriv_dim,) + coords.shape[:-1], dtype=torch.float32, device=device)           
            dm = torch.from_numpy(dm).float().to(device) 
            mem_size_phi = 4 * self.basis.n_orbs * int(torch.prod(torch.tensor(rho.shape)))
            mem = CPU_MEM if device == 'cpu' else GPU_MEM
            # print('mem_size_phi: ', mem_size_phi, 'mem, ', mem)
        phi_shape = (self.basis.n_orbs,)+tuple(rho.shape)
        time1 = time.time()
        # memory estimation for batching
        n_batch_phi = int(np.ceil(mem_size_phi / (mem/2))) # half mem for phi, half for einsum
        n_batch_einsum = int(np.ceil(mem_size_phi*self.basis.n_orbs/30 / (mem/2)))
        batch_size_phi = int(coords.shape[0] // n_batch_phi)
        batch_size_einsum = int(coords.shape[0] // n_batch_einsum)
        assert min(batch_size_phi, batch_size_einsum) > 0, "grid first dim not enough for batching ..."
        if deriv==(0,) and getattr(self, '_phi', None) is not None:
            phi_in_mem = self._phi
            phi_from_mem = True
        elif deriv==(0,1) and getattr(self, '_phi_d01', None) is not None:
            phi_in_mem = self._phi_d01
            phi_from_mem = True
        else: phi_from_mem = False
        if save_phi_in_mem:
            assert mem_size_phi < CPU_MEM / 2
            if deriv==(0,) and getattr(self, '_phi', None) is None:
                self._phi = np.empty(phi_shape)
            elif deriv==(0,1) and getattr(self, '_phi_d01', None) is None:
                self._phi_d01 = np.empty(phi_shape)
            else: save_phi_in_mem = False
        # read or write binary file
        bit = 32 if package=='torch' else 64
        deriv_str = "".join([str(d) for d in deriv])
        bfn = "./binaries/phi_deriv%s_%dbit_mol_%s_grid_%s.hdf5"%(deriv_str, bit, str(self.id), grid_id)
        if osp.isfile(bfn) and not phi_from_mem:
            # print("reading hdf5 ... ")
            fr = h5py.File(bfn, 'r')
            reading, writting = True, False
            h5data_r = fr['phi']
        elif save_phi_to_disk and not osp.isfile(bfn):
            # print("writing hdf5 ... ")
            fw = h5py.File(bfn, 'w')
            reading, writting = False, True
            h5_dtype = 'f4' if package == 'torch' else 'f8'
            h5data_w = fw.create_dataset('phi', phi_shape, dtype=h5_dtype)
        else: reading, writting = False, False
        # fill in rho
        for i in range(0, coords.shape[0], batch_size_phi):
            bg, ed = i, min(i+batch_size_phi, coords.shape[0])
            # print('%s phi(r), grid points: %d - %d ...'%('Loading' if reading else 'Getting/Calculating', bg, ed))
            tt0 = time.time()
            if phi_from_mem:
                phi = phi_in_mem[:, :, bg:ed]
                if package == 'torch': phi = torch.from_numpy(phi).float().to(device)
            elif reading:
                phi = h5data_r[:, :, bg:ed]
                if package == 'torch': phi = torch.from_numpy(phi).float().to(device)
            else:
                if package == 'numpy': phi = self.basis.on_grid_w_numpy(coords[bg:ed], deriv=deriv)
                if package == 'torch': phi = self.basis.on_grid_w_torch(coords[bg:ed], deriv=deriv, device=device)
                # print("phi shape: ", phi.shape)
            if save_phi_in_mem: 
                if package == 'torch': phi = phi.to('cpu').numpy()
                if deriv==(0,): self._phi[:, :, bg:ed] = phi
                elif deriv==(0,1): self._phi_d01[:, :, bg:ed] = phi
            if writting: h5data_w[:, :, bg:ed] = phi.to('cpu') if package=='torch' else phi
            tt1 = time.time()
            for j in range(i, min(i+batch_size_phi, coords.shape[0]), batch_size_einsum):
                bg, ed = j, min(j+batch_size_einsum, i+batch_size_phi, coords.shape[0])
                # print('Calculating phi(r) * DM * phi(r), grid points: %d - %d ...'%(bg, ed))
                rho[:, bg:ed] = contract('i..., j..., ij->...', phi[:,:,bg-i:ed-i], phi[:,0,bg-i:ed-i], dm) # for dim1=1:4, d(phi) * dm * phi
            tt2 = time.time()
            # print("Time for %s phi = "%('LOADING' if reading else 'GETTING/CALULATING'), tt1 - tt0)
            # print("Time for calculating einsum = ", tt2 - tt1)
        if deriv == (0,1): rho[1:4] *= 2.0 # d(phi * dm * phi) = 2*d(phi) * dm * phi  
        if reading: fr.close()
        if writting: fw.close()
        time2 = time.time()
        # print("Time for calculating rho = ", time2 - time1)
        return rho
    



    def __str__(self):
        discription = "Molecule\n"
        def sorted_atoms_str(self):
            idx = sorted(range(len(self._atoms_list)),key=self._atoms_list.__getitem__)
            atoms_str = ""
            for i in idx:
                atoms_str += "%s %16.4e %16.4e %16.4e;\n" % (
                    self._atoms_list[i], 
                    self._atoms_coords[i][0]/A2Bohr, 
                    self._atoms_coords[i][1]/A2Bohr, 
                    self._atoms_coords[i][2]/A2Bohr)
            return atoms_str
        atoms_str = sorted_atoms_str(self)
        discription += atoms_str
        discription += "Charge = "+str(self.charge)+"\n"
        discription += "Spin = "+str(self.spin)+"\n"
        discription += "Basis Set: "+re.sub('[_.-]', '', self.basis_name).lower()
        return discription  


    def __eq__(self, othr):
        return (isinstance(othr, type(self)) and str(self) == str(othr))

    def __hash__(self):
        return hash(str(self))

    @property
    def id(self):
        if getattr(self, '_id', None) is None:
            digits = 10
            import hashlib
            b = bytes(str(self), 'utf-8')
            encoded = hashlib.sha1(b)
            self._id = encoded.hexdigest()[:digits]
        return self._id






        
   

def test():
    # mol1 = Molecule(linear_dists=[0.7], linear_atoms=['H','H'], charge=0, spin=0, basis_name='aug-cc-pvqz')
    # mol1.write_structure('./HH07.str')
    # mol2 = Molecule(struc_path='./HH07.str')

    # # print(mol1.struc_path, mol2.struc_path)
    # assert mol1._atoms_list == mol2._atoms_list
    # assert mol1._atoms_coords == mol2._atoms_coords
    # assert mol1._atoms_str == mol2._atoms_str
    # assert mol1.charge == mol2.charge
    # assert mol1.spin == mol2.spin

    # # print(mol1.basis_name)
    # mol1.grid = 9
    # # print(mol1.pyscf_mol)
    # # print(mol1.grid)
    # mol1.generate_fixed_matrices()
    # # print(mol1.integrals_3c1e.shape)
    # # print(str(mol1.basis))
    # # print(mol2.basis)

    # # # print(mol1.basis.density_on_grid(mol1.dm_hf, np.random.randn(100,100,1000, 3), package='torch', device='cuda').shape)
    # # # print(mol1.basis.density_on_grid(mol1.dm_hf, np.random.randn(100,100,1000, 3), package='torch', device='cuda').shape)
    # mol1.grid = 9
    # # print(str(mol1))
    # # print(hash(mol1))
    # # # print(mol1.id)
    # # print(mol1.rho('ccsd', grid=mol1.grid.coords, deriv=(0,1), package='torch'))

    # mol1.generate_all(grid_level=3)

    # id = mol1.save()


    # t0 = time.time()
    # with open("./molecules/%s.pkl"%(id), 'rb') as f:
    #     mol2 = pickle.load(f)
    # t1 = time.time()
    # # print("Time for pickle loading: ", t1-t0)
    # # print(mol2.grid)
    # # print(mol2.phi)



    # from molecule import Molecule
    # from gen_molecules import gen_water_mols_sym
    # ang = 104.15 / 180 * np.pi
    # bd = 0.9584
    # mol_h2o = gen_water_mols_sym([ang], [bd], 'aug-cc-pvdz')[0]
    # mol_h2o.grid = 3
    # from pyscf.dft import numint
    # import time

    # t0=time.time()
    # ao_values = numint.eval_ao(mol_h2o.pyscf_mol, np.random.randn(100,100,1000, 3), deriv=0)
    # t1=time.time()
    # print('pyscf-phi: ', t1-t0)

    # rho_pyscf = numint.eval_rho(mol_h2o.pyscf_mol, ao_values, mol_h2o.dm_hf, xctype='lda')
    # t2=time.time()
    # print('pyscf-rho: ', t2-t1)
    

    # mol_h2o.phi
    # t3 = time.time()
    # # print('mol-phi: ', t3-t2)
    # rho = mol_h2o.density_on_grid(mol_h2o.dm_hf, np.random.randn(100,100,1000, 3))
    # t4 = time.time()
    # print('mol-rho: ', t4-t3)


    # mol1 = Molecule(linear_dists=[0.2], linear_atoms=['H','H'], charge=0, spin=0, basis_name='aug-cc-pvqz')
    # mol1.generate_fixed_matrices()
    # print(mol1.S[46:,:46])

    # mol = Molecule(linear_dists=[0.2], linear_atoms=['H','H'], charge=0, spin=0, basis_name='aug-cc-pvdz')
    # import gen_molecules
    # eql_hch = 116.133 / 180 * np.pi
    # eql_ch = 1.111
    # eql_co = 1.205
    # mol = Molecule(struc_dict = gen_molecules.gen_sym_formaldehyde_struc_dict(eql_hch, eql_ch, eql_co), 
    #                basis_name='aug-cc-pvdz')
    
    # occ = 5
    # e,c,dm = mol.solve_KS(mol.H, mol.S, np.array([2]*occ + [0]*(len(mol.S)-occ) )  )
    # e2,c2,dm2 = mol.solve_KS_torch(torch.from_numpy(mol.H), torch.from_numpy(mol.S), np.array([2]*occ + [0]*(len(mol.S)-occ) ))
    # # print(e2.numpy(), c2.numpy(), dm2.numpy())
    # print(np.max(np.abs(dm2.numpy()-dm)))
    # # print(np.max(np.abs(mol.dm_hf2()-mol.dm_hf)))

    # mol = Molecule(linear_dists=[0.7414], linear_atoms=['H','H'], charge=0, spin=0, basis_name='aug-cc-pvqz')
    # mol.build_grid(3, sym_list=[])
    # from pyscf import dft
    # ao_values = dft.numint.eval_ao(mol.pyscf_mol, mol.grid.coords, deriv=1)
    # rho = dft.numint.eval_rho(mol.pyscf_mol, ao_values, mol.dm_ccsd, xctype='gga')

    # mol = Molecule(linear_dists=[0.7414], linear_atoms=['H','H'], charge=0, spin=0, basis_name='aug-cc-pvqz')
    import gen_molecules
    eql_hch = 116.133 / 180 * np.pi
    eql_ch = 1.111
    eql_co = 1.205
    mol = Molecule(struc_dict = gen_molecules.gen_sym_formaldehyde_struc_dict(eql_hch, eql_ch, eql_co), 
                   basis_name='aug-cc-pvdz')
    # print(np.max(np.abs(mol.dm_scf(mol.gen_hartree_H)-mol.dm_hf)))
    mol.grid = 3
    print(np.max(np.abs(mol.rho('hf', package='numpy', device='cpu')-mol.density_on_grid(mol.dm_fixed_H, package='numpy', device='cpu'))))
    
    

if __name__ == '__main__':
    test()
