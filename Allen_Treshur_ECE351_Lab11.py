# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                   #
# Treshur Allen                                     #
# ECE 351 - 52                                      #
# Lab 11                                            #
# November 18, 2021                                 #
# This PY file contains the code for lab 11         #
#                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #  

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack
import math
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams

#origninal function:
#  y[k] = 2x[k] − 40x[k − 1] + 10y[k − 1] − 16y[k − 2]

#---------------TASK 1ish TRANSFER FUNCTION------------------

#z =  np.linspace(0,10 ,10)

num = np.array([2, -40])

den = np.array([1, -10, 16])

sys = sig.TransferFunction(num,den)

print('this is the transfer function:', sys)


#---------------TASK 3 PARTIAL FRACTION------------------

partial = sig.residuez(num, den) 

print('this is the partial froaction:', partial)


#-----------TASK 4 ZPLANE----------------------------

def zplane(b,a,filename=None):
    """Plot the complex z-plane given a transfer function.
    """

    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=1.0,
              markeredgecolor='k', markerfacecolor='g')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=12.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    

    return z, p, k


zplane(num,den)

#-------------TASK 5 PLOT FREQZ------------------------

w, h = sig.freqz(num, den, whole = True)


plt.figure()
plt.grid()
plt.semilogx(w, h)
plt.ylabel('w')
plt.xlabel('h')
plt.title('Task 5')
plt.show()










