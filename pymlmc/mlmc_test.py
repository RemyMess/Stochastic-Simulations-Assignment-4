import timeit
import numpy
from math import sqrt
import sys

from .mlmc import mlmc

def mlmc_test(mlmc_fn, M, N, L, N0, Eps, Lmin, Lmax, logfile):
    """
    Multilevel Monte Carlo test routine.

    mlmc_fn: the user low-level routine. Its interface is
      sums = mlmc_fn(l, N)
    with inputs
      l = level
      N = number of paths
    and a numpy array of outputs
      sums[0] = sum(Pf-Pc)
      sums[1] = sum((Pf-Pc)**2)
      sums[2] = sum((Pf-Pc)**3)
      sums[3] = sum((Pf-Pc)**4)
      sums[4] = sum(Pf)
      sums[5] = sum(Pf**2)

    M: refinement cost factor (2**gamma in general MLMC theorem)

    N: number of samples for convergence tests
    L: number of levels for convergence tests

    N0: initial number of samples for MLMC calculations
    Eps: desired accuracy array for MLMC calculations
    """

    # First, convergence tests

    write(logfile, "\n")
    write(logfile, "**********************************************************\n")
    write(logfile, "*** Convergence tests, kurtosis, telescoping sum check ***\n")
    write(logfile, "**********************************************************\n")
    write(logfile, "\n l   ave(Pf-Pc)    ave(Pf)   var(Pf-Pc)    var(Pf)")
    write(logfile, "    kurtosis     check \n-------------------------")
    write(logfile, "--------------------------------------------------\n")

    del1 = []
    del2 = []
    var1 = []
    var2 = []
    kur1 = []
    chk1 = []
    cost = []

    for l in range(0, L+1):
        tic = timeit.default_timer()
        sums = mlmc_fn(l, N)
        toc = timeit.default_timer()
        cost.append(toc - tic)
        sums = sums/N

        if l == 0:
            kurt = 0.0
        else:
            kurt = (     sums[3]
                     - 4*sums[2]*sums[0]
                     + 6*sums[1]*sums[0]**2
                     - 3*sums[0]*sums[0]**3 ) / (sums[1]-sums[0]**2)**2

        del1.append(sums[0])
        del2.append(sums[4])
        var1.append(sums[1]-sums[0]**2)
        var2.append(max(sums[5]-sums[4]**2, 1.0e-10)) # fix for cases with var = 0
        kur1.append(kurt)

        if l == 0:
            check = 0
        else:
            check =          abs(       del1[l]  +      del2[l-1]  -      del2[l])
            check = check / ( 3.0*(sqrt(var1[l]) + sqrt(var2[l-1]) + sqrt(var2[l]) )/sqrt(N))
        chk1.append(check)

        write(logfile, "%2d   %8.4e  %8.4e  %8.4e  %8.4e  %8.4e  %8.4e \n" % \
                      (l, del1[l], del2[l], var1[l], var2[l], kur1[l], chk1[l]))

    if kur1[-1] > 100.0:
        write(logfile, "\n WARNING: kurtosis on finest level = %f \n" % kur1[-1]);
        write(logfile, " indicates MLMC correction dominated by a few rare paths; \n");
        write(logfile, " for information on the connection to variance of sample variances,\n");
        write(logfile, " see http://mathworld.wolfram.com/SampleVarianceDistribution.html\n\n");

    if max(chk1) > 1.0:
        write(logfile, "\n WARNING: maximum consistency error = %f \n" % max(chk1))
        write(logfile, " indicates identity E[Pf-Pc] = E[Pf] - E[Pc] not satisfied \n\n")

    L1 = int(numpy.ceil(0.4*L));
    L2 = L+1;
    pa    = numpy.polyfit(range(L1+1, L2+1), numpy.log2(numpy.abs(del1[L1:L2])), 1);  alpha = -pa[0];
    pb    = numpy.polyfit(range(L1+1, L2+1), numpy.log2(numpy.abs(var1[L1:L2])), 1);  beta  = -pb[0];
    gamma = numpy.log2(cost[-1]/cost[-2]);

    write(logfile, "\n******************************************************\n");
    write(logfile, "*** Linear regression estimates of MLMC parameters ***\n");
    write(logfile, "******************************************************\n");
    write(logfile, "\n alpha = %f  (exponent for MLMC weak convergence)\n" % alpha);
    write(logfile, " beta  = %f  (exponent for MLMC variance) \n" % beta);
    write(logfile, " gamma = %f  (exponent for MLMC cost) \n" % gamma);

    # Second, MLMC complexity tests

    write(logfile, "\n");
    write(logfile, "***************************** \n");
    write(logfile, "*** MLMC complexity tests *** \n");
    write(logfile, "***************************** \n\n");
    write(logfile, "  eps   mlmc_cost   std_cost  savings     N_l \n");
    write(logfile, "----------------------------------------------- \n");

    gamma = numpy.log2(M)
    theta = 0.25

    for eps in Eps:
       (P, Nl) = mlmc(Lmin, Lmax, N0, eps, mlmc_fn, alpha, beta, gamma)
       l = len(Nl) - 1
       mlmc_cost = (1 + 1.0/M)*sum(Nl * M**numpy.arange(0, l+1))
       std_cost  = sum(var2[-1]*M**numpy.arange(0, l+1))/((1.0 -theta)*eps**2)

       write(logfile, "%.4f  %.3e  %.3e  %7.2f " % (eps, mlmc_cost, std_cost, std_cost/mlmc_cost))
       write(logfile, " ".join(["%9d" % n for n in Nl]))
       write(logfile, "\n")

    write(logfile, "\n")

def write(logfile, msg):
    """
    Write to both sys.stdout and to a logfile.
    """
    logfile.write(msg)
    sys.stdout.write(msg)

