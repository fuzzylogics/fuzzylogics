import scipy

def solve_KS(F, S, mo_occ):
    e, C = scipy.linalg.eigh(F, S)
    mocc = C[:, mo_occ > 0]
    dm = np.dot(mocc * mo_occ[mo_occ > 0], mocc.T.conj())
    return e, C, dm