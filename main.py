from pymlmc import mlmc_test
import numpy as np
from math import log, exp, sqrt


def mlmc_expectation(level, n_path, n_0=1000, ref_fac=2, lambd=2):
    """
    Computation of expectation of X_t**2 driven by an SDE using MLMC.

    :param l: level
    :param n_0: number of path to do the expectation on for l=0
    :param lambd: lambda parameter
    :param ref_fac: refinement factor per level
    :param n_path: number of paths for expectation
    :return:
    numpy array "sums" such that
      sums[0] = sum(Pf-Pc)
      sums[1] = sum((Pf-Pc)**2)
      sums[2] = sum((Pf-Pc)**3)
      sums[3] = sum((Pf-Pc)**4)
      sums[4] = sum(Pf)
      sums[5] = sum(Pf**2)

      with Pf and Pc being the estimator at the fine respectively coarse level.
    """
    # Arg checks
    assert(level >= 0 and n_path > 0 and n_0 > 0 and ref_fac > 0 and lambd > 0)

    # 1. Setup (f,c, d denotes "fine/coarse" resp. )
    # terminal time t
    t_f = (level + 1) * log(ref_fac) / lambd
    t_c = t_f * level / (level + 1) if level != 0 else 0
    delta_t = t_f - t_c if level != 0 else t_f

    # time mesh
    n_f = n0 * ref_fac ** level
    n_c = n_f / ref_fac

    # mesh
    h_f = 1 / n_f
    h_c = 1 / n_c

    # placeholder for parallel values
    if level == 0:
        n_path = n_0

    x_f = np.zeros(n_path)
    x_c = np.zeros(n_path)

    # 2. Compute fine path on delta_t
    for _ in range(round(delta_t / h_f)):
        dw_f = np.random.randn(1, n_path) * sqrt(h_f)
        x_f = (1 - h_f) * x_f + dw_f

    # 3. Compute both paths
    if level != 0:
        for _ in range(int(n_c)):
            dw_c = np.zeros(n_path)

            # Drive fine path per coarse update
            for _ in range(ref_fac):
                dw_f = np.random.randn(1, n_path) * sqrt(h_f)
                x_f = (1 - h_f) * x_f + dw_f
                # Update coarser increment
                dw_c = dw_c + dw_f

            # Compute coarse values
            x_c = (1 - h_c) * x_c + dw_c

    else:
        x_c = x_f * 0

    x_c = x_c * 2
    x_f = x_f ** 2

    return np.array([np.sum(x_f - x_c),
            np.sum((x_f - x_c) ** 2),
            np.sum((x_f - x_c) ** 3),
            np.sum((x_f - x_c) ** 4),
            np.sum(x_f),
            np.sum(x_f ** 2)])


# Test expectation and compare with pymlmc package
if __name__ == "__main__":
    fn = "log_compute.txt"
    logf = open(fn, "w")

    # Setting params for test
    M = 2  # Refinement factor.
    N = 20000  # Number of paths
    L = 5  # Levels
    Eps = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]  # Accuracy array
    N0 = 10  # Paths for l=0
    Lmin = 2  # Minimum number of levels
    Lmax = 100  # Maximum number of levels
    n0 = 9  # Initial time steps

    def test_func(L,N):
        return mlmc_expectation(L, N, n_0=N0, ref_fac=M, lambd=2)

    mlmc_test(test_func, M, N, L, N0, Eps, Lmin, Lmax, logf)
    del logf

    # Plot
    mlmc_plot(fn, nvert=3)
    plt.savefig(fn.replace('.txt', '.eps'))