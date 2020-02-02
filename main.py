from pymlmc import mlmc_test
import numpy as np
from math import log, exp


def mlmc_expectation(l, n_0, lambd, n_path=100):
    """
    :param :
    :param :
    :param :
    :param :
    :param :
    :param :

    :return:
    numpy array "sums"e such that
      sums[0] = sum(Pf-Pc)
      sums[1] = sum((Pf-Pc)**2)
      sums[2] = sum((Pf-Pc)**3)
      sums[3] = sum((Pf-Pc)**4)
      sums[4] = sum(Pf)
      sums[5] = sum(Pf**2)

      with Pf and Pc being the estimator at the fine respectively coarse level.
    """
    # Arg checks
    assert(l >=0 and lambd >=0)



    if l == 0:
        # 1. Setup
        n =
        h =
        t =

        # 2. Run

    else:
        # 1. Setup (f,c denotes "fine/coarse" resp. )
        # terminal time t
        t_f = (l + 1) * log(2) / lambd
        t_c = t_f / l

        # time mesh
        h_f = 2 ** (-l)
        h_c = h_f / 2

        # number time steps
        n_f = 1 / h_f
        h_c = n_f * 2

        for _ in range(n_path):
            # use itetools!!!!!!!!!!!!!!!!!!!
            "for coarse and fine"
                # gen bm
                dw = np.random.randn(n_f) * sqrt(h_f)
                x = x * (1-h_f) + dw



    pass


# Script
if __name__ == "__main__":
    # 1. Test expectation
    pass

    # 2. Compare with pymlmc package
    mlmc_test()