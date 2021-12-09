# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                   #
# Treshur Allen                                     #
# ECE 351 - 52                                      #
# Lab 12 - FINAL LAB                                #
# December 8, 2021                                  #
# This PY file contains the code for lab 11         #
#                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #  

#other packages here
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
#from lab 9:
import scipy.fftpack
import math
#from lab manual:
import pandas as pd
#from lab 10:
import control as con


#---------------Part 1------------------
#  
#  My beginning appraoch to these tasks is to add the given code from the 
#  manual so they are ready to go when I need them then:
#  T1: identify the noise magnitudes and frequencies 
#  T2: create a filter 
    
## This is copied from the lab handout 
#  to test the example signal 

#load input signal
df = pd.read_csv('NoisySignal.csv')

t = df['0'].values
sensor_sig = df['1'].values

#this is the sample freq
plt.figure(figsize = (10,6))
plt.plot(t, sensor_sig)
plt.grid()
plt.title('Noisy Input Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude [V]')
plt.show()


#  this is also from the manual, this is the 
#  .stem() workaround which is better to use when the sampling freq
#  is higher than what we've worked with in the past labs. 

def make_stem(ax ,x,y,color='k',style='solid',label='',linewidths =2.5 ,** kwargs):
    ax.axhline(x[0],x[-1],0, color='r')
    ax.vlines(x, 0 ,y, color=color , linestyles=style , label=label , linewidths= linewidths)
    ax.set_ylim ([1.05 * y.min(), 1.05 * y.max()])
    
    return ax
# ^^ I modified the given code to return the ax value to be used later 

#-------------Part 1.5---THE FILTER-----------
"""
I chose to model my filter after a band pass becasuse the frequency range of 
1.8kHz to 2.0kHz would fit within a -parabola type shape, thus a band pass
when looking at the RLC filter I worked with in lab 10 the number will be simiar 
when dividing the ratio of the range by 1m. so that is how I came up with the 
values I wanted to use for my RLC below, the calulations will be included in
the report under the calculations section.
"""



def RCLH(w):
    R = 1000000 #1M
    L = 27*w
    C = 1/(( 1j*10**-4)*w)
    
    mag = 20*np.log((w / (R*C)) / np.sqrt( w**4 + ((1/R*C)**2 - (2/C*C))*w**2 + (1/L*C)**2))
    
    phase = (np.pi / 2) - np.arctan( (w /R*C)/(-w**2 + 1/L*C))
    
    return mag, phase

w = np.linspace(11309, 10000, 12566) #1.8k to 2.0k with 10krad/s step size 

mag1, phase1 = RCLH(w)

phi = np.angle(phase1)

#----------------Part 2-----------------------
#
#  My plan for this part is to perfom the rest of the requirements for 
#  task 2 which is:
#  1: type transferfunction
#  2: plot ^ 
#  My plan for this part is to complete task 3 which is to 
#  generate the bode plot. following what we have done in lab 10

r = 1000000 #1M
l = 27
c = 1e-4

num = [1/r*c,0]
den = [1, 1/r*c, 1/l*c]

sys = sig.TransferFunction(num,den)

w2, mag2, phase2 = sig.bode(sys)

plt.figure()
plt.subplot(2,1,1)
plt.semilogx(w2, mag2)
plt.ylabel('magnitude')
plt.title('Task 3 bode')
plt.subplot(2,1,2)
plt.semilogx(w2, phase2)
plt.ylabel('phase')


#  FFT from lab 9 to demistrate the requirements of task 2

#set sampling freq

fs = 1e6

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

#to satisfy requirements, here I am running through the signal with FFT


freq1, X_mag1, X_phi1 = Fastft(sensor_sig, fs)

fig = plt.figure(figsize = (10, 10))

plt.subplot(3, 1, 1)
plt.plot(t, sensor_sig )
plt.grid()
plt.ylabel('x2(t)')
plt.xlabel('t[s]')
plt.title('Task 1 - cos(2pit)')

plt.subplot(3, 1, 2)
plt.stem(freq1, X_mag1)
plt.grid()
plt.ylabel('mag')

plt.subplot(3, 1, 3)
plt.stem(freq1, X_phi1)
plt.grid()
plt.ylabel('phase')


#--------------Part 4 the final filter through------------

from numpy import sin, cos, pi, arange
from numpy.random import randint

fs = 1e6
Ts = 1/fs
t_end = 50e-3

t = arange(0,t_end-Ts,Ts)

f1 = 1.8e3
f2 = 1.9e3
f3 = 2e3
f4 = 1.85e3
f5 = 1.87e3
f6 = 1.94e3
f7 = 1.92e3

info_signal = (2.5*cos(2*pi*f1*t) + 1.75*cos(2*pi*f2*t) + 2*cos(2*pi*f3*t) + 
               2*cos(2*pi*f4*t) + 1*cos(2*pi*f5*t) + 1*cos(2*pi*f6*t) +
               1.5*cos(2*pi*f7*t))

N = 25
my_sum = 0

for i in range(N+1):
    noise_amp     = 0.075*randint(-10,10,size=(1,1))
    noise_freq    = randint(-1e6,1e6,size=(1,1))
    noise_signal  = my_sum + noise_amp * cos(2*pi*noise_freq*t)
    my_sum = noise_signal

f6 = 1000000    #50e3    #50kHz
f7 =  27   #49.9e3
f8 =  1e-4   #51e3

pwr_supply_noise = 1.5*sin(2*pi*f6*t) + 1.25*sin(2*pi*f7*t) + 1*sin(2*pi*f8*t)

f9 = 60

low_freq_noise = 1.5*sin(2*pi*f9*t)

total_signal = info_signal + noise_signal + pwr_supply_noise + low_freq_noise
total_signal = total_signal.reshape(total_signal.size)

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t,info_signal)
plt.grid(True)
"""
plt.subplot(3,1,2)
plt.plot(t,info_signal+pwr_supply_noise)
plt.grid(True)
"""
plt.subplot(2,1,2)
plt.plot(t,total_signal)
plt.grid(True)
plt.show()

df = pd.DataFrame({'0':t,
                   '1':total_signal})

df.to_csv('NoisySignal.csv')