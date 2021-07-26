import numpy as np
import matplotlib.pyplot as plt
import find_coords as fc
import pickle

A2Bohr = 1.889726124625

def load_data():
    coords = []
    with open('coords.pkl', 'rb') as f_c:
        try:
            while True:
                coords.append(pickle.load(f_c))
        except EOFError:
            pass
    rho_diff = []
    with open('rho_diff.pkl', 'rb') as f_rho:
        try:
            while True:
                rho_diff.append(pickle.load(f_rho))
        except EOFError:
            pass
    vxc = []
    with open('vxc.pkl', 'rb') as f_v:
        try:
            while True:
                vxc.append(pickle.load(f_v))
        except EOFError:
            pass
    return coords, rho_diff, vxc


c, rd, v = load_data()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
rg = [i for i in range(len(rd))]
plt.xlim([-2, 2])
plt.ylim([-0.2, 0.2])

# for CO in CH2O only
eql_co = 1.205
variation = [*np.linspace(0.8, 1.2, 9)]
COs = [eql_co*v for v in variation]
print(COs)
#__________________________________


for i in rg:
    ixyz, xyz = fc.find_idx(c[i], 2)
    xyz = xyz / A2Bohr
    # xyz_i = xyz + COs[i]
    print(rg)
    ax.plot(xyz, rd[i][ixyz], linewidth=3)
fig.savefig('rho_diff.png', dpi=300)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.xlim([0, 5])
# plt.ylim([-10, 10])
for i in rg:
    ixyz, xyz = fc.find_idx(c[i], 2)
    xyz = xyz / A2Bohr
    # xyz_i = xyz + COs[i]
    ax.plot(xyz, v[i].cpu().numpy()[ixyz])
fig.savefig('vxc.png', dpi=300)
plt.close()