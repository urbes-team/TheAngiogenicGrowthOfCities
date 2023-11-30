import pandas as pd
import numpy as np
import pickle
import sys
from SALib.analyze import sobol

# TODO: add files from Myriad - .sh file, result files, wherever
#  plot_sensitivity_analysis() is used
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    val = array[idx]
    return idx, val


def find_optimum_params(var, results, params):
    print(var)
    if type(results) == str:
        results = pickle.load(open(results, 'rb'))
    if type(results) == list:
        print('IS LIST')
        results = np.array(results)
    if type(params) == str:
        params = pd.read_csv(params, index_col=0, sep=' ',
                             names=['py', 'file', 'D', 'kappa', 'tau', 'mu',
                                    'nu', 'r_R',
                                    'beta', 'ygross', 'run_no']).drop(
            columns=['py', 'file'])

    if 'r2' in var:
        opt_idx, opt_val = find_nearest(array=results, value=1)

    else:
        opt_idx = np.argmin(results)
        opt_val = np.min(results)

    opt_params = params.loc[opt_idx + 1]

    return opt_params, opt_val


def sensitivity_analysis(results, savepath=None):
    params = pd.read_csv('./Results/London/SAphaseall/SA_allphase_saltelli.txt',
                         index_col=0, sep=' ',
        names=['py', 'file', 'D', 'kappa', 'tau', 'mu', 'nu',
               'r_R', 'beta', 'ygross', 'run_no']).drop(
        columns=['py', 'file'])

    prob = {'num_vars': 8,
            'names': ['D', 'kappa', 'tau', 'mu', 'nu',
                      'r_R', 'beta', 'ygross'],
            'bounds': [[0.01, 0.1],
                       [1e-4, 3e-4],
                       [0.9, 1.1],
                       [0.264, 0.396],
                       [1.86, 2.8],
                       [0.0072, 0.0108],
                       [1.6, 2.4],
                       [4.5, 5.5]]}

    X = params[['D', 'kappa', 'tau', 'mu', 'nu',
                'r_R', 'beta', 'ygross']].values

    Y = results

    Si = sobol.analyze(prob, Y, print_to_console=True)

    if savepath is not None:
        pickle.dump(Si, open(savepath, 'wb'))

    return Si


if __name__ == "__main__":
    sensitivity_analysis(results=sys.argv[1],
                         savepath=None)
