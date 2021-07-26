import numpy as np
import matplotlib.pyplot as plt
import find_coords as fc
import pickle

def load_data(i):

    with open('coords%d.pkl'%(i), 'rb') as f_c:
        coords = pickle.load(f_c)
    with open('rho_diff%d.pkl'%(i), 'rb') as f_rho:
        rho_diff = pickle.load(f_rho)
    with open('vxc%d.pkl'%(i), 'rb') as f_v:
        vxc = pickle.load(f_v)
    return coords, rho_diff, vxc

def load_appended_data(i):
    with open('coords%d.pkl'%(i), 'rb') as f_c:
        coords = pickle.load(f_c)
    rho_diff = []
    with open('rho_diff%d.pkl'%(i), 'rb') as f_rho:
        try:
            while True:
                rho_diff.append(pickle.load(f_rho))
        except EOFError:
            pass
    vxc = []
    with open('vxc%d.pkl'%(i), 'rb') as f_v:
        try:
            while True:
                vxc.append(pickle.load(f_v))
        except EOFError:
            pass
    # rho_diff = rho_diff[-10:]
    # vxc = vxc[-10:]
    return coords, rho_diff, vxc

for idx in range(9):
    c, rd, v = load_appended_data(idx)
    ixyz, xyz = fc.find_idx(c, 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    rg = [*range(len(rd))]
    print(rg)
    # rg = [*range(10)]
    plt.xlim([-3, 2])
    plt.ylim([-0.5, 0.5])
    for i in rg:
        print(i)
        ax.plot(xyz, rd[i][ixyz], linewidth=3)
    fig.savefig('rho_diff_%d.png' %(idx), dpi=300)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.xlim([-3, 2])
    plt.ylim([-1, 1])
    for i in rg:
        ax.plot(xyz, v[i][ixyz])
    fig.savefig('vxc_%d.png' %(idx), dpi=300)
#    plt.show()
    plt.close()