# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                   #
# Treshur Allen                                     #
# ECE 351 - 52                                      #
# Lab 4                                             #
# September 30, 2021                                #
# This PY file contains the code for lab 4          #
#                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #  

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import math

steps = 1e-2 #step size
t = np.arange(-10,10 +steps, steps)
#---------------------Part 1 FUNCTIONS --------------------------
def stepFunc(t):
   
    u = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] <= 0:  
            u[i] = 0
        else:
            u[i] = 1
    return u

def h1(t):
    h = np.exp(-2*t) * (stepFunc(t) - stepFunc(t - 3))
    return h

def h2(t):
    h = stepFunc(t - 2) - stepFunc(t - 6)
    return h

def h3(t):
    h = np.sin((2*math.pi* 0.25) * t) * stepFunc(t)
    return h

#---------------------PLOTS-------------------------

plt.figure(figsize = (10, 10))
plt.subplot(3, 1, 1)
plt.plot(t, h1(t))
plt.grid()
plt.ylabel('H1')
plt.title('Part 1 Task 2')


plt.subplot(3 , 1, 2)
plt.plot(t, h2(t))
plt.grid()
plt.ylabel('H2')

plt.subplot(3 , 1, 3)
plt.plot(t, h3(t))
plt.grid()
plt.ylabel('H3')
plt.xlabel('t')
#plt.show()

#-----------PART 2 FUNCTIONS/CONVOLUTION ----------------
steps = 1e-2 #step size
time = np.arange(2*t[0], 2*t[len(t)-1]+steps, steps)

def conv(f1,f2):
  Nf1 = len(f1)
  Nf2 = len(f2)
  f1Extended = np.append(f1,np.zeros((1,Nf2-1)))
  f2Extended = np.append(f2,np.zeros((1,Nf1-1))) 
  #these append function make it so the arrays have the same number
  #of elements to avoid errors 
  result = np.zeros(f1Extended.shape) #shape is the array

  for i in range(Nf2+Nf1-2):
        result[i] = 0
        for j in range(Nf1):
            if(i-j+1>0):
                try:
                    result[i] += f1Extended[j]*f2Extended[i-j+1]
                except:
                        print(i,j)
  return result


h1 = h1(t)
h2 = h2(t)
h3 = h3(t)
u = stepFunc(t)

#scipy.signal.convolve(h1,u)


plt.figure(figsize = (10, 10))
plt.subplot(3, 1, 1)
plt.plot(time, conv(h1, u))
plt.grid()
plt.ylabel('H1 Convolution')
plt.title('Part 2 Task 2')
#plt.show()

plt.subplot(3 , 1, 2)
plt.plot(time, conv(h2, u))
plt.grid()
plt.ylabel('H2 Convolution')
plt.xlabel('t')

plt.subplot(3 , 1, 3)
plt.plot(time,conv(h3, u))
plt.grid()
plt.ylabel('H3 Convolution')
plt.xlabel('t')
#plt.show()

#-----------------HAND CALCULATED INTEGREAL CHECK------------------------

def rampFunc(t):
    r = np.zeros(t.shape)
    
    for i in range(len(t)):
        if i < 0:
            r[i] = 0;
        else:
            r[i] = i
    return r

def test1(t):
    f = -0.5*np.exp(-2 * t) * stepFunc(t) + rampFunc(t) + 0.5*np.exp(-2*t) - rampFunc(t - 3)
    return f

def test2(t):
    y = scipy.signal.convolve(h2,u)
    return y

def test3(t):
    h = scipy.signal.convolve(h3,u)
    return h

time3 = np.arange(2*t[0], 2*t[len(t)-1]+steps, steps)

plt.figure(figsize = (10, 10))
#plt.subplot(3, 1, 1)
plt.plot(time3,test1(t))
plt.grid()
plt.ylabel('H1 Convolution')
plt.title('Part 2 Task 1')
plt.show()








