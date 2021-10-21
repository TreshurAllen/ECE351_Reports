# # # # # # # # # # # # # # # # # # # # # # # # # # # 
#                                                   #
# Treshur Allen                                     #
# ECE 351 - 52                                      #
# Lab 7                                             #
# October  21 , 2021                                #
# This PY file contains the code for lab 7          #
#                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # #  

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math

#--------------------Part 1 Task 1-----
#this will have my hand calculated equations of z, p, k

#--------------------Part 1 Task 2-----------------------------

def G():
    num = [0, 0, 1, 9]
    den = [1, -2, -40, -64]
    print('\nG(s) zero, poll, gain: ', sig.tf2zpk(num, den))
    return 

def A():
    num = [0, 1, 4]
    den = [1, 4, 3]
    print('\nA(s) zero, poll, gain: ', sig.tf2zpk(num, den))
    return 
    
def B():
    p = [1, 26, 168]
    print('\nB(s) zeros: ', np.roots(p))
    return  

g = G()
a = A()
b = B()

#---------------------Part 1 Task 3-------------------------------------

#on paper we have the factored form of the two functions multiplied 

#----------------------Part 1 Task 4-------------------------

#Explination of the open loop being unstable

#----------------------Part 1 Task 5--------------------------
#convolve 

numer = sig.convolve([1, 4], [1, 9])
print('\nNmerator = ', numer)
denom = sig.convolve([1, 4, 3], [1, -2, -40, -64])
print('\nDenominator = ', denom)

resp = [numer, denom]

t,h = sig.step(resp)
    
plt.plot(t, h)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel('Time[s]')
plt.title('Part 1 Task 5')
plt.show()


#-------------------Part 2 Task 1--------------------------------
#hand written symbolic closed loop equation

#-----------------Part 2 Task 2--------------------------------

numA = [1, 4]

denA = [1, 4, 3]

numG = [1, 9]

denG = [1, -2, -40, -64]

numB = [1, 26, 168]

denB = [1]

numHT = sig.convolve(numA, numG)
numHB = sig.convolve(denA, denG)

top = sig.convolve(numHT, numHB)

denHT = sig.convolve(numB, numG)
denHB = sig.convolve(denB, denG)

bot = 1 + (sig.convolve(denHT, denHB))

newH = [bot, top]

print('\nTHIS IS PART 2: \n')

print('\nhere is the new H function top: ', top )
print('\nhere is the new H function bot: ', bot )

print('\nHere is the new H function:', newH)

#print('\nNEW zero, poll, gain: ', sig.tf2zpk(top, bot))
#---------------Part 2 Task 4-------------------------

n, m = sig.step(newH)
    
plt.plot(n,m)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel('Time[s]')
plt.title('Part 2 Task 4')
plt.show()





