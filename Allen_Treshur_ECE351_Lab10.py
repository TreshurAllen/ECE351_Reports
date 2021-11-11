# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                   #
# Treshur Allen                                     #
# ECE 351 - 52                                      #
# Lab 10                                            #
# November 11, 2021                                 #
# This PY file contains the code for lab 10         #
#                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #  

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack
import math

"""
def dB(amp):
    return 20*math.log(amp)
"""
def RCLH(w):
    R = 1000
    L = .027j*w
    C = 1/( (100j*10**-9) * w)
    
    mag = 20*np.log((w / (R*C)) / np.sqrt( w**4 + ((1/R*C)**2 - (2/C*C))*w**2 + (1/L*C)**2))
    
    phase = (np.pi / 2) - np.arctan( (w /R*C)/(-w**2 + 1/L*C))
    
    return mag, phase

#----------------PART 1--------Task 1--------------------------------
w = np.linspace(10000, 10000000, 100000) #10^3 < w < 10^6 with 10^4 steps

mag1, phase1 = RCLH(w)

phi = np.angle(phase1)

#mag1dB = dB(mag1)

plt.figure()
plt.subplot(2,1,1)
plt.semilogx(w, mag1)
#plt.grid()
plt.ylabel('magnitude in dB')
plt.title('Task 1 bode')

plt.subplot(2,1,2)
plt.semilogx(w, phi)
#plt.plot(w,phi)
#plt.grid()
plt.ylabel('Phase in degrees')

#---------------------PART 1 ------Task 2 ------------------------------

r = 1000
l = 27*10**-3
c = 100*10**-9

num = [1/r*c,0]
den = [1, 1/r*c, 1/l*c]

sys = sig.TransferFunction(num,den)

w2, mag2, phase2 = sig.bode(sys)

plt.figure()
plt.subplot(2,1,1)
plt.semilogx(w2, mag2)
plt.ylabel('magnitude')
plt.title('Task 2 bode')
plt.subplot(2,1,2)
plt.semilogx(w2, phase2)
plt.ylabel('phase')


#----------------PART 1--------Task 3-------------------------------

import control as con #this is the one we were supposed to get in L0


sys2 = con.TransferFunction(num,den)
#_ = con.bode(sys2, omega, dB = True, deg - True, Plot = True)
#  use  _=... to suppress the output

w3, mag3, phase3 = con.bode(sys2)

plt.figure()
plt.subplot(2,1,1)
plt.semilogx(w3, mag3)
plt.ylabel('magnitude')
plt.title('Task 3 bode')
plt.subplot(2,1,2)
plt.semilogx(w3, phase3)
plt.ylabel('phase')
plt.show()

#-------------PART 2----------Task 1-----------------------------
fs = 10000 #10^3
steps = 1/ fs #step size
t = np.arange(0, 0.01 +steps, steps)
#t = np.linspace(0.0, 1/100, 1/10000)
x2 = np.cos(2*np.pi*100*t) + np.cos(2*np.pi*3024*t) + np.sin(2*np.pi*50000*t)

sig.bilinear()










