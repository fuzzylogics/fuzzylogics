def load_data(f):
    import numpy as np
    data = np.load(f)
    coords = data['coords']
    rho_diff = data['rho_diff']
    return coords, rho_diff

def find_coords(coords, values):
    import numpy as np
    # coords: (n, 3)
    # values: (n, )
    k = 0
    for c in coords:
        if abs(c[0]) < 1.e-8 and abs(c[1]) < 1.e-8:
            k += 1
    
    data = np.zeros([k, 2])
    k = 0
    for i, c in enumerate(coords):
        if abs(c[0]) < 1.e-8 and abs(c[1]) < 1.e-8:
            data[k][0] = c[2]
            data[k][1] = values[i]
            k += 1

    data = np.array(sorted(data.tolist(), key=lambda a:a[0]))
    return data


def find_idx(c, ax):
    import numpy as np
    all_ax = [0,1,2]
    all_ax.remove(ax)
    inz = [[i, ci[ax]] for i, ci in enumerate(c) if abs(ci[all_ax[0]])<1.e-8 and abs(ci[all_ax[1]])<1.e-8]
    inz = [ele for ele in sorted(inz, key=lambda x:x[1])]
    inz = np.asarray(inz).transpose()
    print(c.shape)
    i, z = inz[0], inz[1]
    i = i.astype(int)
    return i, z

