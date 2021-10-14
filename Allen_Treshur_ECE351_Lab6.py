# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                   #
# Treshur Allen                                     #
# ECE 351 - 52                                      #
# Lab 6                                             #
# October 14  , 2021                                #
# This PY file contains the code for lab 6          #
#                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #  

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math

steps = 1e-2 #step size
t = np.arange(0,2 +steps, steps)
#---------------------Part 1 task 1 FUNCTIONS --------------------------
def stepFunc(t):
   
    u = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] <= 0:  
            u[i] = 0
        else:
            u[i] = 1
    return u

def prelab_y(t):
    y = 6.93*np.sin(t + 89.8) * stepFunc(t) 
    return y

#---------------------Part 1 task 2 FUNCTIONS --------------------------
num = [1, 10, 24]
den = [1, 6, 12]
#H = sig.TransferFunction(num,den)
#h = sig.step(num,den)

w,h = sig.freqs(num, den)

#---------------------PLOTS-------------------------

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, prelab_y(t))
plt.grid()
plt.ylabel('y(t)')
plt.title('Part 1 Task 1')


plt.subplot(2 , 1, 2)
plt.plot(w, h)
plt.grid()
plt.ylabel('H(s)')
plt.title('Part 1 Task 2')
plt.show()

#------------------------Part 1 Task 3--------------------
print('\nThe coefficients from partial fraction Part 1:\n', sig.residue(num,den))

#---------------------Part 2 Task 1---------------------------

denom = [0, 0, 0, 0, 0, 25250]
numer = [1, 18, 218, 2036, 9085, 25250]

print('\nThe coefficients from partial fraction Part 2:\n', sig.residue(numer,denom))

#---------------------part 2 task 2------------------------

d,g = sig.freqs(numer, denom)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(d,g)
plt.grid()
plt.ylabel('H(s)')
plt.title('Part 2 Task 2')

#--------------------------------Part 2 task 3-------------------------

h = sig.step(numer,denom)


plt.subplot(2, 1, 2)
plt.plot(t,h)
plt.grid()
plt.ylabel('H(s)')
plt.title('Part 2 Task 3')



