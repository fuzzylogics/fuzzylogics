import numpy as np
from numpy.lib.financial import npv
import scipy
import pyscf
import pyscf.cc
from pyscf import dft
import tqdm
# constants list
BLK_SIZE = 2000


def solve_KS(f, s, mo_occ):
    e, c = scipy.linalg.eigh(f, s)
    mocc = c[:, mo_occ > 0]
    dm = np.dot(mocc * mo_occ[mo_occ > 0], mocc.T.conj())
    return e, c, dm

def density_on_grids(mol, coords, dm, deriv=0):
    XCTYPE = ['lda', 'gga']
    ao_values = dft.numint.eval_ao(mol, coords, deriv=deriv)
    rho = dft.numint.eval_rho(mol, ao_values, dm, xctype=XCTYPE[deriv])
    return rho

def density_on_grids_blk(mol, coords, dm, deriv=0):
    XCTYPE = ['lda', 'gga']
    total = len(coords)
    n_blk = total // BLK_SIZE
    res = total - n_blk * BLK_SIZE
    if deriv == 0:
        rho = np.empty(total)
    elif deriv == 1:
        rho = np.empty([4, total])
    else:
        raise NotImplementedError
    with tqdm(total=total) as pbar:
        for i_blk in range(n_blk):
            ao_values = dft.numint.eval_ao(
                mol, coords[i_blk * BLK_SIZE: (i_blk + 1) * BLK_SIZE], deriv=deriv)
            rho[:, i_blk * BLK_SIZE: (i_blk + 1) * BLK_SIZE] = dft.numint.eval_rho(
                mol, ao_values, dm, xctype=XCTYPE[deriv])
            pbar.update(BLK_SIZE)
        if res > 0:
            ao_values = dft.numint.eval_ao(mol, coords[-res:], deriv=deriv)
            rho[:, -res:] = dft.numint.eval_rho(mol,
                                                ao_values, dm, xctype=XCTYPE[deriv])
            pbar.update(res)
    return rho


def ccsd(mol, mcc=None, mf=None):
    if mf is None:
        mf = pyscf.scf.RHF(mol)
        mf.kernel()
    c = mf.mo_coeff
    if mcc is None:
        mcc = pyscf.cc.CCSD(mf)
    ecc, t1, t2 = mcc.kernel()
    rdm1 = mcc.make_rdm1()
    rdm1_ao = np.einsum('pi,ij,qj->pq', c, rdm1, c.conj())
    return rdm1_ao

def rhf(mol):
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    rdm1 = mf.make_rdm1()
    return rdm1


def eval_xc_mat(mol, coords, weights, vxc, mat_sz):

    total_size = len(coords)
    n_blk = total_size // BLK_SIZE
    res = total_size - BLK_SIZE * n_blk

    xc_mat = np.zeros(mat_sz) 
    ao_loc = mol.ao_loc_nr()
    shls_slice = (0, mol.nbas)

    for i in range(n_blk):
        idx = slice(BLK_SIZE * i, BLK_SIZE * (i + 1))
        ao = pyscf.dft.numint.eval_ao(mol, coords[idx], deriv=0)
        n_grids, n_ao = ao.shape
        wv = weights[idx] * vxc[idx] * 0.5
        aow = np.einsum('pi,p->pi', ao, wv)
        xc_mat += pyscf.dft.numint._dot_ao_ao(mol, ao, aow, np.ones((n_grids, mol.nbas), dtype=np.int8), shls_slice, ao_loc)
    
    if res > 0:
        ao = pyscf.dft.numint.eval_ao(mol, coords[-res:], deriv=0)
        n_grids, n_ao = ao.shape
        wv = weights[-res:] * vxc[-res:] * 0.5
        aow = np.einsum('pi,p->pi', ao, wv)
        xc_mat += pyscf.dft.numint._dot_ao_ao(mol, ao, aow, np.ones((n_grids, mol.nbas), dtype=np.int8), shls_slice, ao_loc)

    return xc_mat + xc_mat.T




def test():
    from molecule import Molecule

    mol1 = Molecule(linear_dists=[0.7], linear_atoms=['H','H'], charge=0, spin=0, basis_name='aug-cc-pvqz')
    mol1.grid = 3
    rho_old = density_on_grids(mol1.pyscf_mol, mol1.grid.coords, mol1.dm_ccsd, deriv=1)
    rho_new = mol1.rho('ccsd', mol1._grid, deriv=(0,1))
    vxc = np.random.randn(*(mol1.grid.weights.shape))
    V_mat_old = eval_xc_mat(mol1.pyscf_mol, mol1.grid.coords, mol1.grid.weights, vxc, (mol1.basis.n_orbs, mol1.basis.n_orbs))
    V_mat_new = mol1.V_mat(vxc)
    print(np.max(V_mat_new-V_mat_old))
if __name__ == '__main__':
    test()