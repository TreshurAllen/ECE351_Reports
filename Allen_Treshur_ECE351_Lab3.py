# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                   #
# Treshur Allen                                     #
# ECE 351 - 52                                      #
# Lab 2                                             #
# September 23, 2021                                #
# This file contains the code for lab 3             #
#                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #  

import numpy as np
import matplotlib.pyplot as plt

steps = 1e-2 #step size
t = np.arange(0, 20+steps, steps)
#---------------------STEP AND RAMP FUNCTIONS-----------------------------------
def stepFunc(t):
    u = np.zeros(t.shape)
    
    for i in range(len(t)):
        if t[i] <= 0:  
            u[i] = 0
        else:
            u[i] = 1
    return u

def rampFunc(t):
    r = np.zeros(t.shape)
    
    for i in range(len(t)):
        if i < 0:
            r[i] = 0;
        else:
            r[i] = i
    return r
#-------------------------------PLOTS--------------------------------------------

u = stepFunc(t) #run the function first 

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, u)
plt.grid()
plt.ylabel('Step Function (u(t))')
plt.title('Plots Part 1 Task 1')

r = rampFunc(t)

plt.subplot(2 , 1, 2)
plt.plot(t, r)
plt.grid()
plt.ylabel('Ramp Function (r(t))')
plt.xlabel('t')


#-------------------------------PART 1 FUNCTIONS--------------------------------

def Func1(t):
    f = stepFunc(t - 3) - stepFunc(t-9)
    return f

def Func2(t):
    f = np.exp(-t) * stepFunc(t)
    return f

def Func3(t):
    f = rampFunc(t-2)*(stepFunc(t-2) - stepFunc(t-3)) + rampFunc(4-t)*(
        stepFunc(t-3) - stepFunc(t-4)) 
    return f

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, Func1(t))
plt.grid()
plt.ylabel('F1')
plt.title('Part 1 Task 2')


plt.subplot(3 , 1, 2)
plt.plot(t, Func2(t))
plt.grid()
plt.ylabel('F2')
plt.xlabel('t')

plt.subplot(3 , 1, 3)
plt.plot(t, Func3(t))
plt.grid()
plt.ylabel('F3')
plt.xlabel('t')

#--------------------------------PART 2 TASK 1---------------------------------

def conv(f1,f2):
  Nf1 = len(f1)
  Nf2 = len(f2)
  f1Extended = np.append(f1,np.zeros((1,Nf2-1)))
  f2Extended = np.append(f2,np.zeros((1,Nf1-1))) 
  #these append function make it so the arrays have the same number
  #of elements to avoid errors 
  result = np.zeros(f1Extended.shape) #shape is the array

  for i in range(Nf1+Nf2-2):
        result[i] = 0
        for j in range(Nf1):
            if(i-j+1>0):
                try:
                    result[i] += f1Extended[j]*f2Extended[i-j+1]
                except:
                        print(i,j)
  return result

steps = 1e-2 #step size

t = np.arange(0, 50+steps, steps) 
f1 = Func1(t)
f2 = Func2(t)
#both = conv(f1, f2)    
    
plt.figure(figsize = (10, 10))
plt.plot(t,conv(f1, f2))
plt.grid()
plt.ylabel('F(t)')
plt.xlabel('t')
plt.title('Plots for self convolution')
plt.show()

p1 = np.signal.convolve(f1 , f2)

p2 = np.signal.convolve(f1 , f2)

p3 = np.signal.convolve(f1 , f2)


plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t, p1)
plt.grid()
plt.ylabel('F1')
plt.title('Part 2 Task 2-4')


plt.subplot(3 , 1, 2)
plt.plot(t, p2)
plt.grid()
plt.ylabel('F2')
plt.xlabel('t')

plt.subplot(3 , 1, 3)
plt.plot(t, p3)
plt.grid()
plt.ylabel('F3')
plt.xlabel('t')
plt.show()


