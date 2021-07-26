# display error history log generated during training
import numpy as np

# from numpy.lib.function_base import disp

def plot_err_results(test_labels,
                     err_type,
                     models,
                     scf=False,
                     catag_not_xy=False, 
                     lin_not_log=False,
                     display=True, save=True):
    
    results = {}
    for model in models:
        results[model] = {}
        test_err = np.load('test_err_log%s.npy'%(model))
        results[model]['test_err_mean'] = test_err[0]
        results[model]['log_test_err_mean'] = np.log10(test_err[0])
        results[model]['test_err_max'] = test_err[1]
        results[model]['log_test_err_max'] = np.log10(test_err[1])
        results[model]['test_err_I'] = test_err[2]
        results[model]['log_test_err_I'] = np.log10(test_err[2])
        results[model]['test_scf_err_mean'] = test_err[3]
        results[model]['log_test_scf_err_mean'] = np.log10(test_err[3])
        results[model]['test_scf_err_max'] = test_err[4]
        results[model]['log_test_scf_err_max'] = np.log10(test_err[4])
        results[model]['test_scf_err_I'] = test_err[5]
        results[model]['log_test_scf_err_I'] = np.log10(test_err[5])
    
    
    import matplotlib.pyplot as plt
    w0 = '' if lin_not_log else 'log_'
    w1 = 'test_'
    w2 = 'scf_' if scf else ''
    w3 = 'err_'
    
    w = w0+w1+w2+w3
    var, legend = [], []
    if type(err_type) is not list: err_type = [err_type]
    for w5 in models:
        for w4 in ['mean', 'max', 'I']:
            if w4 in err_type:
                var += [results[w5][w+w4]]
                legend += [w4+w5]
    print(legend)
    var = np.stack(var, 1)
    print(var.shape)
    m = var.shape[0]

    if len(test_labels) == 0:
        test_labels = ['test'+str(x) for x in range(m)]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # if not lin_not_log: plt.ylim(-4.5, -2)
    if not catag_not_xy:
        ax.plot([float(x) for x in test_labels], var, '-o', label='l')
    else:
        ax.plot(test_labels, var, 'o', label='l')
    # ax.axis('equal')
    ax.legend(legend, loc='upper right', frameon=True)
    plt.xlabel("different structures")
    plt.ylabel(w0+w1+w2+w3)

    if save: fig.savefig('%s.png' %(w0+w1+w2+w3), dpi=300)
    if display: plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test_labels", type=str, nargs='+',
                        help="err types to display")
    parser.add_argument("-t", "--err_type", type=str, nargs='+',
                        default='mean',
                        help="err types to display")
    parser.add_argument("-m", "--models", type=str, nargs='+',
                        default=[''], help='models to be evaled')
    parser.add_argument("-r", "--scf", help="results after scf",
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
    if not args.catagory and len(args.test_labels)==3:
        test_labels = np.linspace(float(args.test_labels[0]), 
                                  float(args.test_labels[1]),
                                  int(args.test_labels[2]),
                                  endpoint=False)
    else: test_labels = args.test_labels
    models = [x if x=='' else "_"+x for x in args.models]
    plot_err_results(test_labels,
                     args.err_type,
                     models,
                     args.scf,
                     lin_not_log = args.lin,
                     catag_not_xy = args.catagory, 
                     display = (not args.no_display),
                     save = args.save)
