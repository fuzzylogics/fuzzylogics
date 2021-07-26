# generate molecules
from molecule import Molecule
from numpy import sin, cos, pi
import numpy as np


# hydrogen
#________________________________________________
def gen_H2_struc_dict(d):
    struc_dict = {}
    H1 = ['H', 0.0, 0.0, d/2.0]
    H2 = ['H', 0.0, 0.0, -d/2.0]
    struc_dict['atoms'] = [H1, H2]
    struc_dict['charge'] = 0
    struc_dict['spin'] = 0
    return struc_dict
def H2_set(variations=None, basis='aug-cc-pvdz', return_type='kwargs'):
    eql_hh = 0.7414
    if variations is None:
        variations = [*np.linspace(0.9, 1.1, 5)]
    HHs = [eql_hh*v for v in variations]
    all_kwargs = []
    if return_type == 'mols': mols = []
    idx = 0
    for hh in HHs:
        kwargs = {'struc_dict': gen_H2_struc_dict(hh), 
                  'basis_name': basis,
                  'custom_description': {'index': idx,
                                         'formula': 'H2',
                                         'HH': hh}
                  }                   
        all_kwargs += [kwargs]
        if return_type == 'mols':
            mol = Molecule(**kwargs)
            print(str(mol))
            mols += [mol]
        idx += 1
    if return_type == 'kwargs': return all_kwargs          
    if return_type == 'mols': return mols
def train_H2_set(basis='aug-cc-pvdz', return_type='kwargs'):
    variations = [*np.linspace(0.5/0.7424, 0.9/0.7424, 10, endpoint=True)]
    return H2_set(variations=variations, basis=basis, return_type=return_type)
def val_H2_set(basis='aug-cc-pvdz', return_type='kwargs'):
    variations = [*np.linspace(0.4, 1.0, 2, endpoint=True)]
    return H2_set(variations=variations, basis=basis, return_type=return_type)
def test_H2_set(basis='aug-cc-pvdz', return_type='kwargs'):
    # variations = [*np.linspace(0.504/0.7414, 0.896/0.7414, 50, endpoint=True)]
    variations = [*np.linspace(0.3/0.7414, 1.2/0.7414, 20, endpoint=True)]
    return H2_set(variations=variations, basis=basis, return_type=return_type)
#________________________________________________


def gen_glycine_struc_dict(d_cc_stretch):
    struc_dict = {}
    C1 = ['C',  0.00000000e+00,  0.00000000e+00,  0.00000000e+00]
    O2 = ['O', -6.93256222e-01,  9.87124744e-01,  0.00000000e+00]
    O3 = ['O', -4.97888769e-01, -1.25674005e+00,  0.00000000e+00]
    C4 = ['C',  1.52369290e+00+d_cc_stretch,  1.18712769e-16,  0.00000000e+00]
    N5 = ['N',  2.13479990e+00+d_cc_stretch,  1.32123292e+00,  0.00000000e+00]
    H6 = ['H', -1.46308638e+00, -1.15726907e+00,  0.00000000e+00]
    H7 = ['H',  1.85971777e+00+d_cc_stretch, -5.76049244e-01,  8.81300000e-01]
    H8 = ['H',  1.85971777e+00+d_cc_stretch, -5.76049244e-01, -8.81300000e-01]
    H9 = ['H',  1.76231300e+00+d_cc_stretch,  1.83545263e+00,  8.03200000e-01]
    H10= ['H',  1.76231300e+00+d_cc_stretch,  1.83545263e+00, -8.03200000e-01]
    struc_dict['atoms'] = [C1, O2, O3, C4, N5, H6, H7, H8, H9, H10]
    struc_dict['charge'] = 0
    struc_dict['spin'] = 0
    return struc_dict
def glycine_set(variations=None, basis='aug-cc-pvdz', return_type='kwargs'):    
    eql_cc = 1.52369290
    if variations is None:
        variations = [*np.linspace(0.9, 1.1, 3)]
    CCs = [eql_cc*v for v in variations]
    all_kwargs = []
    if return_type == 'mols': mols = []
    idx = 0
    for cc in CCs:
        kwargs = {'struc_dict': gen_glycine_struc_dict(cc - eql_cc), 
                  'basis_name': basis,
                  'custom_description': {'index': idx,
                                         'formula': 'H2NCH2COOH',
                                         'CC': cc}
                  } 
        all_kwargs += [kwargs]
        if return_type == 'mols':
            mol = Molecule(**kwargs)
            print(str(mol))
            mols += [mol]
        idx += 1
    if return_type == 'kwargs': return all_kwargs
    if return_type == 'mols': return mols
def train_glycine_set(basis='aug-cc-pvdz', return_type='kwargs'):
    variations = [*np.linspace(0.8, 1.1, 4, endpoint=True)]
    return glycine_set(variations=variations, basis=basis, return_type=return_type)
def val_glycine_set(basis='aug-cc-pvdz', return_type='kwargs'):
    variations = [*np.linspace(0.5, 2.0, 4, endpoint=True)]
    return glycine_set(variations=variations, basis=basis, return_type=return_type)
def test_glycine_set(basis='aug-cc-pvdz', return_type='kwargs'):
    variations = [*np.linspace(0.6, 1.4, 3, endpoint=True)]
    return glycine_set(variations=variations, basis=basis, return_type=return_type)



def gen_water_mols_sym(angs, bds, basis):
    mols = []
    for ang in angs:
        for bd in bds:
            struc_dict = {}
            # struc_dict['atoms'] = []
            O = ['O', 0.0, 0.0, 0.0]
            H1 = ['H', -sin(ang/2.0)*bd, cos(ang/2.0)*bd, 0.0]
            H2 = ['H', sin(ang/2.0)*bd, cos(ang/2.0)*bd, 0.0]
            struc_dict['atoms'] = [O, H1, H2]
            struc_dict['charge'] = 0
            struc_dict['spin'] = 0
            mol = Molecule(struc_dict=struc_dict, basis_name=basis)
            mols += [mol]
            print(str(mol))
    return mols

# def gen_water_mols_sym(angs, bds, basis):
#     mols = []
#     for ang in angs:
#         for bd in bds:
#             struc_dict = {}
#             # struc_dict['atoms'] = []
#             O = ['O', 0.0, 0.0, 0.0]
#             H1 = ['H', -sin(ang/2.0)*bd, 0.0, cos(ang/2.0)*bd]
#             H2 = ['H', sin(ang/2.0)*bd, 0.0, cos(ang/2.0)*bd]
#             struc_dict['atoms'] = [O, H1, H2]
#             struc_dict['charge'] = 0
#             struc_dict['spin'] = 0
#             mol = Molecule(struc_dict=struc_dict, basis_name=basis)
#             mols += [mol]
#             print(str(mol))
#     return mols

def simple_water_set(basis='aug-cc-pvdz'):
    eql_ang = 104.15 / 180 * pi
    eql_bd = 0.9584
    angs = [eql_ang*0.9, eql_ang, eql_ang*1.1]
    bds = [eql_bd*0.9, eql_bd, eql_bd*1.1]
    # angs = [eql_ang]
    # bds = [eql_bd]
    mols = gen_water_mols_sym(angs, bds, basis=basis)
    return mols

# formaldehyde
def gen_sym_formaldehyde_struc_dict(hch, ch, co):
    struc_dict = {}
    # struc_dict['atoms'] = []
    C = ['C', 0.0, 0.0, 0.0]
    O = ['O', 0.0, -co, 0.0]
    H1 = ['H', -sin(hch/2.0)*ch, cos(hch/2.0)*ch, 0.0]
    H2 = ['H', sin(hch/2.0)*ch, cos(hch/2.0)*ch, 0.0]
    struc_dict['atoms'] = [C, O, H1, H2]
    struc_dict['charge'] = 0
    struc_dict['spin'] = 0
    return struc_dict
def train_formaldehyde_set(basis='aug-cc-pvdz', return_type='kwargs'):
    eql_hch = 116.133 / 180 * pi
    eql_ch = 1.111
    eql_co = 1.205
    # variations = [0.9, 1.0, 1.1]
    # HCHs = [eql_hch*v for v in variations]
    # COs = [eql_co*v for v in variations]
    # CHs = [eql_ch*v for v in variations]
    variations = [*np.linspace(0.8, 1.2, 3)]
    HCHs = [eql_hch]
    CHs = [eql_ch]
    COs = [eql_co*v for v in variations]
    all_kwargs = []
    if return_type == 'mols': mols = []
    idx = 0
    for hch in HCHs:
        for ch in CHs:
            for co in COs:
                kwargs = {'struc_dict': gen_sym_formaldehyde_struc_dict(hch, ch, co), 
                          'basis_name': basis,
                          'custom_description': {'index': idx,
                                                 'formula': 'CH2O',
                                                 'HCH': hch, 'CH': ch, 'CO': co}
                          }                   
                all_kwargs += [kwargs]
                if return_type == 'mols':
                    mol = Molecule(**kwargs)
                    print(str(mol))
                    mols += [mol]
                idx += 1
    if return_type == 'kwargs': return all_kwargs          
    if return_type == 'mols': return mols
def val_formaldehyde_set(basis='aug-cc-pvdz', return_type='kwargs'):
    eql_hch = 116.133 / 180 * pi
    eql_ch = 1.111
    eql_co = 1.205
    variation = [*np.linspace(0.9, 1.1, 2)]
    # HCHs = [eql_hch*v for v in variation]
    # COs = [eql_co*v for v in variation]
    # CHs = [eql_ch*v for v in variation]
    HCHs = [eql_hch]
    COs = [eql_co*v for v in variation]
    CHs = [eql_ch]
    all_kwargs = []
    if return_type == 'mols': mols = []
    idx = 0
    for hch in HCHs:
        for ch in CHs:
            for co in COs:
                kwargs = {'struc_dict': gen_sym_formaldehyde_struc_dict(hch, ch, co), 
                          'basis_name': basis,
                          'custom_description': {'index': idx,
                                                 'formula': 'CH2O',
                                                 'HCH': hch, 'CH': ch, 'CO': co}
                          }                   
                all_kwargs += [kwargs]
                if return_type == 'mols':
                    mol = Molecule(**kwargs)
                    print(str(mol))
                    mols += [mol]
                idx += 1
    if return_type == 'kwargs': return all_kwargs          
    if return_type == 'mols': return mols
def test_formaldehyde_set(basis='aug-cc-pvdz', return_type='kwargs'):
    eql_hch = 116.133 / 180 * pi
    eql_ch = 1.111
    eql_co = 1.205
    variation = [*np.linspace(0.4, 1.4, 20, endpoint=False)]
    # HCHs = [eql_hch*v for v in variation]
    # COs = [eql_co*v for v in variation]
    # CHs = [eql_ch*v for v in variation]
    HCHs = [eql_hch]
    COs = [eql_co*v for v in variation]
    CHs = [eql_ch]
    all_kwargs = []
    if return_type == 'mols': mols = []
    idx = 0
    for hch in HCHs:
        for ch in CHs:
            for co in COs:
                kwargs = {'struc_dict': gen_sym_formaldehyde_struc_dict(hch, ch, co), 
                          'basis_name': basis,
                          'custom_description': {'index': idx,
                                                 'formula': 'CH2O',
                                                 'HCH': hch, 'CH': ch, 'CO': co}
                          }                   
                all_kwargs += [kwargs]
                if return_type == 'mols':
                    mol = Molecule(**kwargs)
                    print(str(mol))
                    mols += [mol]
                idx += 1
    if return_type == 'kwargs': return all_kwargs          
    if return_type == 'mols': return mols


if __name__ == '__main__':
    # eql_ang = 104.15 / 180 * pi
    # eql_bd = 0.9584
    # angs = [eql_ang*0.9, eql_ang, eql_ang*1.1]
    # bds = [eql_bd*0.9, eql_bd, eql_bd*1.1]
    # mols = gen_water_mols_sym(angs, bds, basis='aug-cc-pvdz')
    # for mol in mols:
    #     print(str(mol))
    print(glycine_set(return_type='mols'))
    