#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy

def mlmc_plot(filename, nvert):

    file = open(filename, "r")

    # Declare lists for data
    del1 = []
    del2 = []
    var1 = []
    var2 = []
    kur1 = []
    chk1 = []
    l    = []

    epss = []
    mlmc_cost = []
    std_cost = []
    Ns = []

    for line in file:
        # Recognise convergence test lines from the fact that line[1] is an integer
        if line[0] == ' ' and '0' <= line[1] <= '9':
            splitline = [float(x) for x in line.split()]
            l.append(splitline[0])
            del1.append(splitline[1])
            del2.append(splitline[2])
            var1.append(splitline[3])
            var2.append(splitline[4])
            kur1.append(splitline[5])
            chk1.append(splitline[6])

        # Recognise MLMC complexity test lines from the fact that line[0] is an integer
        if '0' <= line[0] <= '9':
            splitline = [float(x) for x in line.split()]
            epss.append(splitline[0])
            mlmc_cost.append(splitline[1])
            std_cost.append(splitline[2])
            Ns.append(splitline[4:])

    plt.figure(figsize=(8, 6))

    plt.subplot(nvert, 2, 1)
    plt.plot(l,     numpy.log2(var2),     '*-',  label=r'$P_l$')
    plt.plot(l[1:], numpy.log2(var1[1:]), '*--', label=r'$P_l - P_{l-1}$')
    plt.xlabel('level $l$')
    plt.ylabel(r'$\mathrm{log}_2(\mathrm{variance})$')
    plt.legend(loc='lower left', fontsize='x-small')

    plt.subplot(nvert, 2, 2)
    plt.plot(l,     numpy.log2(numpy.abs(del2)),     '*-',  label=r'$P_l$')
    plt.plot(l[1:], numpy.log2(numpy.abs(del1[1:])), '*--', label=r'$P_l - P_{l-1}$')
    plt.xlabel('level $l$')
    plt.ylabel(r'$\mathrm{log}_2(|\mathrm{mean}|)$')
    plt.legend(loc='lower left', fontsize='x-small')

    if nvert == 3:
        plt.subplot(nvert, 2, 3)
        plt.plot(l[1:], chk1[1:], '*--')
        plt.xlabel('level $l$')
        plt.ylabel(r'consistency check')
        axis = plt.axis(); plt.axis([0, max(l), axis[2], axis[3]])

        plt.subplot(nvert, 2, 4)
        plt.plot(l[1:], kur1[1:], '*--')
        plt.xlabel('level $l$')
        plt.ylabel(r'kurtosis')
        axis = plt.axis(); plt.axis([0, max(l), axis[2], axis[3]])

    styles = ['o--', 'x--', 'd--', '*--', 's--']
    plt.subplot(nvert, 2, 2*nvert-1)
    for (eps, N, style) in zip(epss, Ns, styles):
        plt.semilogy(N, style, label=eps)
    plt.xlabel('level $l$')
    plt.ylabel('$N_l$')
    plt.legend(loc='upper right', frameon=True, fontsize='x-small')

    eps = numpy.array(epss)
    std_cost = numpy.array(std_cost)
    mlmc_cost = numpy.array(mlmc_cost)
    plt.subplot(nvert, 2, 2*nvert)
    plt.loglog(eps, eps**2 * std_cost,  '*-',  label='std MC')
    plt.loglog(eps, eps**2 * mlmc_cost, '*--', label='MLMC')
    plt.xlabel(r'accuracy $\epsilon$')
    plt.ylabel(r'$\epsilon^2$ cost')
    plt.legend(fontsize='x-small')
    axis = plt.axis(); plt.axis([min(eps), max(eps), axis[2], axis[3]])

    plt.subplots_adjust(wspace=0.3)

if __name__ == "__main__":
    import sys

    mlmc_plot(sys.argv[1], nvert=3)
    plt.savefig(sys.argv[1].replace(".txt", ".eps"))
