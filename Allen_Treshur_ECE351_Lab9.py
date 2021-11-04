# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                   #
# Treshur Allen                                     #
# ECE 351 - 52                                      #
# Lab 9                                             #
# November 4, 2021                                  #
# This PY file contains the code for lab 9          #
#                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #  

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.fftpack

fs = 100

#-------------Function for Tasks 1-3------------

def Fastft(x, fs):
    N = len(x) #find length of signal
    X_fft = scipy.fftpack.fft(x) # perform the fast fourier transform
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) #shift zero freq components
                                                  #to the center of the spectrum
    
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
    # signal , (fs is the sampling frequency and
    # needs to be defined previously in your code
    
    X_mag = np.abs(X_fft_shifted)/N #compute the magnitudes of the signal
    X_phi = np.angle(X_fft_shifted) #compute the phases of the signal 
    return freq, X_mag, X_phi
    #--------- END OF THE FUNCTION--------------


#-------------Function for Task 4------------
"""
def Fastft(x, fs):
    N = len(x) #find length of signal
    X_fft = scipy.fftpack.fft(x) # perform the fast fourier transform
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) #shift zero freq components
                                                  #to the center of the spectrum
    
    freq = np.arange(-N/2, N/2)*fs/N # compute the frequencies for the output
    # signal , (fs is the sampling frequency and
    # needs to be defined previously in your code
    
    X_mag = np.abs(X_fft_shifted)/N #compute the magnitudes of the signal
    
    #for i in range(1, N+1, 1):
    if X_mag[0] < 1e-10:
        X_phi = 0
    else:
        X_phi = np.angle(X_fft_shifted) #compute the phases of the signal 
    return freq, X_mag, X_phi
    #--------- END OF THE FUNCTION--------------
"""
t = np.linspace(0, 2, 100)
x1 = np.cos(2*np.pi*t)

freq1, X_mag1, X_phi1 = Fastft(x1, fs)



# need to use stem to get these plots to be correct

fig = plt.figure(figsize = (10, 10))
gs = plt.GridSpec(nrows=3, ncols=2)


fig.add_subplot(gs[0, :])
#plt.subplot(5, 1, 1)
plt.plot(t, x1 )
plt.grid()
plt.ylabel('x2(t)')
plt.xlabel('t[s]')
plt.title('Task 1 - cos(2pit)')

fig.add_subplot(gs[1, 0])
#plt.subplot(5, 1, 2)
plt.stem(freq1, X_mag1)
plt.grid()
plt.ylabel('|x(f)|')

fig.add_subplot(gs[2, 0])
#plt.subplot(5, 1, 3)
plt.stem(freq1, X_phi1)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

fig.add_subplot(gs[1, 1])
plt.xlim(-5,5)
plt.stem(freq1, X_mag1)
plt.grid()
#plt.ylabel('|x(f)|')   ZOOMED MAG

fig.add_subplot(gs[2, 1])
plt.xlim(-5,5)
plt.stem(freq1, X_phi1)
plt.grid()
#plt.ylabel('/_x(f)')   ZOOMED PHASE
plt.xlabel('f[Hz]')

#------------------- END OF TASK 1---------------------------


x2 = 5 * np.cos(2*np.pi*t)

freq2, X_mag2, X_phi2 = Fastft(x2, fs)



# need to use stem to get these plots to be correct

fig = plt.figure(figsize = (10, 10))
gs = plt.GridSpec(nrows=3, ncols=2)


fig.add_subplot(gs[0, :])
#plt.subplot(5, 1, 1)
plt.plot(t, x2 )
plt.grid()
plt.ylabel('x2(t)')
plt.xlabel('t[s]')
plt.title('Task 2 - 5sin(2pit)')

fig.add_subplot(gs[1, 0])
#plt.subplot(5, 1, 2)
plt.stem(freq2, X_mag2)
plt.grid()
plt.ylabel('|x(f)|')

fig.add_subplot(gs[2, 0])
#plt.subplot(5, 1, 3)
plt.stem(freq2, X_phi2)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

fig.add_subplot(gs[1, 1])
plt.xlim(-5,5)
plt.stem(freq2, X_mag2)
plt.grid()
#plt.ylabel('|x(f)|')   ZOOMED MAG

fig.add_subplot(gs[2, 1])
plt.xlim(-5,5)
plt.stem(freq2, X_phi2)
plt.grid()
#plt.ylabel('/_x(f)')   ZOOMED PHASE
plt.xlabel('f[Hz]')

#------------------- END OF TASK 2---------------------------

x3 = 2 * np.cos((2 * np.pi * 2 * t) - 2) + (np.sin((2*np.pi*6*t) +3))**2

freq3, X_mag3,X_phi3 = Fastft(x3, fs)



# need to use stem to get these plots to be correct

fig = plt.figure(figsize = (10, 10))
gs = plt.GridSpec(nrows=3, ncols=2)


fig.add_subplot(gs[0, :])
#plt.subplot(5, 1, 1)
plt.plot(t, x3 )
plt.grid()
plt.ylabel('x3(t)')
plt.xlabel('t[s]')
plt.title('Task 3 - 5sin(2pit)')

fig.add_subplot(gs[1, 0])
#plt.subplot(5, 1, 2)
plt.stem(freq3, X_mag3)
plt.grid()
plt.ylabel('|x(f)|')

fig.add_subplot(gs[2, 0])
#plt.subplot(5, 1, 3)
plt.stem(freq3, X_phi3)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

fig.add_subplot(gs[1, 1])
plt.xlim(-5,5)
plt.stem(freq3, X_mag3)
plt.grid()
#plt.ylabel('|x(f)|')   ZOOMED MAG

fig.add_subplot(gs[2, 1])
plt.xlim(-5,5)
plt.stem(freq3, X_phi3)
plt.grid()
#plt.ylabel('/_x(f)')   ZOOMED PHASE
plt.xlabel('f[Hz]')

#------------------- END OF TASK 3---------------------------
"""

Psi3 = 0*x
L = 15
for l in range(1, L+1, 1):
    Psi3 += (2 / (l * 3.14159)) * (1 - np.cos(l * 3.14159*x))
"""
    
    
x4 = (2 / (15 * 3.14159)) * (1 - np.cos(15 * 3.14159*t))

freq4, X_mag4,X_phi4 = Fastft(x4, fs)



# need to use stem to get these plots to be correct

fig = plt.figure(figsize = (10, 10))
gs = plt.GridSpec(nrows=3, ncols=2)


fig.add_subplot(gs[0, :])
#plt.subplot(5, 1, 1)
plt.plot(t, x4 )
plt.grid()
plt.ylabel('x4(t)')
plt.xlabel('t[s]')
plt.title('Task 5 from lab 8')

fig.add_subplot(gs[1, 0])
#plt.subplot(5, 1, 2)
plt.stem(freq4, X_mag4)
plt.grid()
plt.ylabel('|x(f)|')

fig.add_subplot(gs[2, 0])
#plt.subplot(5, 1, 3)
plt.stem(freq4, X_phi4)
plt.grid()
plt.ylabel('/_x(f)')
plt.xlabel('f[Hz]')

fig.add_subplot(gs[1, 1])
plt.xlim(-5,5)
plt.stem(freq4, X_mag4)
plt.grid()
#plt.ylabel('|x(f)|')   ZOOMED MAG

fig.add_subplot(gs[2, 1])
plt.xlim(-5,5)
plt.stem(freq4, X_phi4)
plt.grid()
#plt.ylabel('/_x(f)')   ZOOMED PHASE
plt.xlabel('f[Hz]')