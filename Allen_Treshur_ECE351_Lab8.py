# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                   #
# Treshur Allen                                     #
# ECE 351 - 52                                      #
# Lab 8                                             #
# October  28 , 2021                                #
# This PY file contains the code for lab 8          #
#                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #  

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math

steps = 1e-2 #step size
t = np.arange(0,20 +steps, steps)
#---------------------Part 1 task 1 FUNCTIONS --------------------------

T = 1
P = 3.14159

w = 2*P

ak = 0

def b(k):
    b = (2 / (k * 3.14159)) * (1 - np.cos(k * 3.14159))
    return b
    
# ------------------ALL AK VALUES PART 1 TASK 1-----------------

print('\n\nak values: ')

print('\na0 = ', ak)
print('\na1 = ', ak)

#------------------ALL BK VALUES 1 2 3 TASK 1-------------------

print('\n\nbk values: ')

print('\nb0 = 0' )  #division by zero 
print('\nb1 = ', b(1))
print('\nb2 = ', b(2))
print('\nb3 = ', b(3))


#------------TASK 2 PLOTS 1-3----------------
# x is 1000 equally spaced points from 0 to 20, inclusive
#to provide proper resolution
x = np.linspace(0, 20, 1000) 
Psi1 = 0*x # now Psi is an array of zeros

N = 1
# second input of range is N+1 since our index n satisfies 1 <= n < N+1
# third input makes n increment by 2 each loop instead of the default 1
for n in range(1, N+1, 1):
    Psi1 += (2 / (n * 3.14159)) * (1 - np.cos(n * 3.14159*x))


plt.figure(figsize = (10, 10))

plt.subplot(3, 1, 1)
plt.plot(x, Psi1 )
plt.grid()
plt.ylabel('N = 1')
plt.title('Sumation plots 1-3')

Psi2 = 0*x
M = 3
for m in range(1, M+1, 1):
    Psi2 += (2 / (m * 3.14159)) * (1 - np.cos(m * 3.14159*x))

plt.subplot(3, 1, 2)
plt.plot(x,Psi2)
plt.grid()
plt.ylabel('N = 3')

Psi3 = 0*x
L = 15
for l in range(1, L+1, 1):
    Psi3 += (2 / (l * 3.14159)) * (1 - np.cos(l * 3.14159*x))

plt.subplot(3, 1, 3)
plt.plot(x, Psi3)
plt.grid()
plt.ylabel('N = 15')

#----------------------TASK 2 PLOTS 4-6----------------
Psi4 = 0*x # now Psi is an array of zeros
Z = 50
# second input of range is N+1 since our index n satisfies 1 <= n < N+1
# third input makes n increment by 2 each loop instead of the default 1
for z in range(1, Z+1, 1):
    Psi4 += (2 / (z * 3.14159)) * (1 - np.cos(z * 3.14159*x))


plt.figure(figsize = (10, 10))

plt.subplot(3, 1, 1)
plt.plot(x, Psi4 )
plt.grid()
plt.ylabel('N = 50')
plt.title('Sumation plots 4-6')

Psi5 = 0*x
V = 150
for v in range(1, V+1, 1):
    Psi5 += (2 / (v * 3.14159)) * (1 - np.cos(v * 3.14159*x))

plt.subplot(3, 1, 2)
plt.plot(x,Psi5)
plt.grid()
plt.ylabel('N = 150')

Psi6 = 0*x
Y = 1500
for y in range(1, Y+1, 1):
    Psi6 += (2 / (y * 3.14159)) * (1 - np.cos(y * 3.14159*x))

plt.subplot(3, 1, 3)
plt.plot(x, Psi6)
plt.grid()
plt.ylabel('N = 1500')









