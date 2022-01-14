""" calculate metrics """

import numpy as np

def rmse(mod2_sol, mod2_pred):
    """
    input: prediction / ans
    output: rmse
    """
    tmp = mod2_sol - mod2_pred
    rmse_out = np.sqrt(tmp.power(2).mean())
    return rmse_out
