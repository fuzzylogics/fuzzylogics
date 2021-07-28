# display error history log generated during training
import numpy as np

# from numpy.lib.function_base import disp

tr_err = np.load('tr_err_log.npy')
tr_err_mean = tr_err[0,:,0,:]
log_tr_err_mean = np.log10(tr_err_mean)
tr_err_max = tr_err[1,:,0,:]
log_tr_err_max = np.log10(tr_err_max)
tr_err_I = tr_err[2,:,0,:]
log_tr_err_I = np.log10(tr_err_I)

val_err = np.load('val_err_log.npy')
val_err_mean = val_err[0,:,0,:]
log_val_err_mean = np.log10(val_err_mean)
val_err_max = val_err[1,:,0,:]
log_val_err_max = np.log10(val_err_max)
val_err_I = val_err[2,:,0,:]
log_val_err_I = np.log10(val_err_I)

def plot_err_history(err_type, val_not_tr=False, lin_not_log=False,
                     display=True, save=True):
    import matplotlib.pyplot as plt
    w0 = '' if lin_not_log else 'log_'
    w1 = 'val_' if val_not_tr else 'tr_'
    w2 = 'err_'
    # w3 = 'max' if max_not_mean else 'mean'
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if type(err_type) is not list: err_type = [err_type]
    for w3 in ['mean', 'max', 'I']:
        if w3 in err_type:
            w = w0+w1+w2+w3
            var = eval(w)
            n = len(var)
            ax.plot(var)

    if save: fig.savefig('%s_%d.png' %(w, n), dpi=300)
    if display: plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--err_type", type=str, nargs='+',
                        default='mean',
                        help="err types to display")
    # parser.add_argument("-x", "--max", help="max rather than mean",
    #                     action="store_true")
    parser.add_argument("-v", "--val", help="val rather than tr",
                        action="store_true")
    parser.add_argument("-l", "--lin", help="lin rather than log",
                        action="store_true")    
    parser.add_argument("-nd", "--no_display", help="don't display plot",
                        action="store_true")
    parser.add_argument("-s", "--save", help="don't save plot",
                        action="store_true")
    args = parser.parse_args()
    print(args)
    plot_err_history(args.err_type, val_not_tr=args.val, lin_not_log=args.lin,
                     display=(not args.no_display), save=args.save )
