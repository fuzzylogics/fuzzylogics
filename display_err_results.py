# display error history log generated during training
import numpy as np

# from numpy.lib.function_base import disp

tr_err = np.load('tr_err_log.npy')
tr_err_mean = tr_err[0,:,-1,:]
log_tr_err_mean = np.log10(tr_err_mean)
tr_err_max = tr_err[1,:,-1,:]
log_tr_err_max = np.log10(tr_err_max)

val_err = np.load('val_err_log.npy')
val_err_mean = val_err[0,:,0,:]
log_val_err_mean = np.log10(val_err_mean)
val_err_max = val_err[1,:,0,:]
log_val_err_max = np.log10(val_err_max)

def plot_err_results(tr_labels, val_labels, catag_not_xy=False,
                     max_not_mean=False, lin_not_log=False,
                     display=True, save=True):
    import matplotlib.pyplot as plt
    w0 = '' if lin_not_log else 'log_'
    w1_tr, w1_val = 'tr_', 'val_'
    w2 = 'err_'
    w3 = 'max' if max_not_mean else 'mean'
    w_tr, w_val = w0+w1_tr+w2+w3, w0+w1_val+w2+w3
    var_tr, var_val = eval(w_tr), eval(w_val)
    n = var_tr.shape[0]
    m_tr, m_val = var_tr.shape[1], var_val.shape[1]

    if tr_labels is None: tr_labels = ['tr'+str(x) for x in range(m_tr)]
    if val_labels is None: val_labels = ['val'+str(x) for x in range(m_val)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if not lin_not_log: plt.ylim(-3, 0)
    if not catag_not_xy:
        ax.plot([float(x) for x in tr_labels], var_tr[-1,:], '-ob', label='Train')
        ax.plot([float(x) for x in val_labels], var_val[-1,:], '-or', label='Validate')
    else:
        ax.plot(tr_labels, var_tr[-1,:], 'ob', label='Train')
        ax.plot(val_labels, var_val[-1,:], 'or', label='Validate')
    # ax.axis('equal')
    ax.legend(loc='upper right', frameon=True)
    plt.xlabel("different structures")
    plt.ylabel(w0+w2+w3)

    if save: fig.savefig('%s_%d.png' %(w0+w2+w3, n), dpi=300)
    if display: plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tr_labels", type=str, nargs='+',
                        help="labels for training results")
    parser.add_argument("-v", "--val_labels", type=str, nargs='+',
                        help="labels for validation results",)
    parser.add_argument("-x", "--max", help="max rather than mean",
                        action="store_true")
    parser.add_argument("-l", "--lin", help="lin rather than log",
                        action="store_true")    
    parser.add_argument("-nd", "--no_display", help="don't display plot",
                        action="store_true")
    parser.add_argument("-s", "--save", help="don't save plot",
                        action="store_true")
    parser.add_argument("-ca", "--catagory", help="catagorical plot",
                        action="store_true")
    args = parser.parse_args()
    print(args)
    plot_err_results(args.tr_labels, args.val_labels, catag_not_xy=args.catagory,
                     max_not_mean=args.max, lin_not_log=args.lin,
                     display=(not args.no_display), save=args.save)
