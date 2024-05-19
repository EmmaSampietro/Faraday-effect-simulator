# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:11:19 2024

@author: clem, sascha, emma
"""

import numpy as np
import matplotlib.pyplot as plt
import math

e_0 = 8.8541878128e-12 #vacuum permittivity (change if we are in other medium)
mu_0 = 1.256637062e-6 #vacuum permeability

c = 1/np.sqrt(e_0*mu_0) #speed of light
m_e = 9.1e-31 #electron mass
q_e = 1.6e-19 #electron charge
B_0 = 1e3  #magnitude external B field

m_max = 600 #steps we want to perform (space)
n_max = 1000 #steps we want to perform (time)

m_source = 1

#Constants for the source
lambda_source = 550e-9 #wavelength of the source
w = 2*np.pi*c/lambda_source
tau = 10e-15 #
t0 = tau*2 #center of the Gaussian pulse

#step size
dx = 25e-9 #step size on position
dt = 0.9*dx/c #lambda_source/10  #step size on time

#Constants we will use in our code
Sc = c*dt/dx #Courant number
Z_0 = 1  #np.sqrt(mu_0/e_0) #impedance of free space
n = 1e27 #update, density of electrons so that j=-nev
gamma = 0 #update
Sc_o_Z0=Sc/Z_0  #to have less constants in our code
Z0Sc=Z_0*Sc   #to have less constants in our code

w_p = np.sqrt(n/(m_e*e_0)) * q_e 
w_c = (q_e*B_0)/m_e

#Initializing the arrays
E_z = np.zeros(m_max) #we create arrays of zeros of length m_max
E_y = np.zeros(m_max)
H_y = np.zeros(m_max)
H_z = np.zeros(m_max)
j_y = np.zeros(m_max)
j_z = np.zeros(m_max)

E_z_previous = np.zeros(m_max)
E_y_previous = np.zeros(m_max)
H_y_previous = np.zeros(m_max)
H_z_previous = np.zeros(m_max)
j_y_previous = np.zeros(m_max)
j_z_previous = np.zeros(m_max)

angle_rec=np.zeros(n_max)

#Function for our gaussian wavepacket source
def Source_Function(t):
  E = np.exp(-(t-t0)**2/tau**2)*np.sin(w*t) #gaussian wavepacket
  return E


#Function to compute the angle of polarization in the y-z plane from the z axis
def polarization_angle(E_y,E_z):
    if E_z==0:
        degree_angle = np.pi/2
    else:
        radians_angle = math.atan(E_y/E_z)
        degree_angle = radians_angle *180/np.pi
    return degree_angle


for n in range(n_max):
  #recall n is time index
  H_y[m_max-1]=H_y[m_max-2] #updating magnetic field boundaries=> our Absorbing boundary conditions no longer work
  H_z[m_max-1]=H_z[m_max-2]
  
  j_y[m_max-1]=j_y[m_max-2]
  j_z[m_max-1]=j_z[m_max-2]
  
  for m in range(m_max-1):
    #space iteration (recall m is space index)
    H_y[m] = H_y_previous[m] + Sc_o_Z0 *(E_z[m+1]-E_z[m]) #updating magnetic field
    H_y_previous[m] = H_y[m]
    
    H_z[m] = H_z_previous[m] + Sc_o_Z0 *(E_y[m+1]-E_y[m]) #updating magnetic field
    H_z_previous[m] = H_z[m]

  #Magnetic field source
  tn=n*dt
  H_y[m_source-1] -= Source_Function(tn)/Z_0
  H_y_previous[m_source-1] = H_y[m_source-1]

  #Field polarized along the E_z direction so the source is only in H_y
  #H_z[m_source-1] -= Source_Function(n)/Z_0
  H_z_previous[m_source-1] = H_z[m_source-1]
  
  
  for m in range(m_max-1): #staggered grid so we must iterate starting at +1/ finish at m+1 
    #updating the current j
    j_y[m]=(j_y_previous[m]*math.cos(w_c*dt)-j_z_previous[m]*math.sin(w_c*dt))*math.exp(-gamma*dt) + E_y[m]
    j_z[m]=(j_y_previous[m]*math.sin(w_c*dt)+j_z_previous[m]*math.cos(w_c*dt))*math.exp(-gamma*dt) + E_z[m]
  
    j_y_previous[m]=j_y[m]
    j_z_previous[m]=j_z[m]



  E_z[0]=E_z_previous[1] #updating electric field boudaries
  E_y[0]=E_y_previous[1]
  

  for m in range(1,m_max): #staggered grid so we must iterate starting at +1/ finish at m+1 
  #updating electric field
    E_z[m] = E_z_previous[m] + Z0Sc * (H_y[m] - H_y[m-1]) - (w_p*dt)**2 * (j_z[m])  
    E_z_previous[m] = E_z[m]
    
    E_y[m] = E_y_previous[m] + Z0Sc * (H_z[m] - H_z[m-1]) - (w_p*dt)**2 * (j_y[m])
    E_y_previous[m] = E_y[m]


  #Electric field source
  tnp1=(n+1)*dt
  E_z[m_source] += Source_Function(tnp1)
  E_z_previous[m_source] = E_z[m_source]

  #Field polarized in the E_z direction initially so no E_y source
  #E_y[m_source] += Source_Function(n+1)
  E_y_previous[m_source] = E_y[m_source]

    
  #angle_rec[n] = polarization_angle((E_y[499]), (E_z[499]))
  angle_rec[n] = polarization_angle(max(E_y), max(E_z))
  
  
  if n==n_max-1:
    final_rotation=angle_rec[n_max-1]-angle_rec[0]
    print("The final rotation angle is of degrees ", final_rotation)
    
  #we plot it
  if n%40 == 0:
    # Plotting the first plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2 row, 2 columns
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    axs[0][0].plot(E_z)
    axs[0][0].set_title('$E_z$ field')
    axs[0][0].set_xlabel('X-axis (step)')
    axs[0][0].set_ylabel('Z-axis')
    axs[0][0].set_ylim(-2.5, 2.5)
    axs[0][0].set_xlim(0, 600)

    axs[1][0].plot(H_y, color='green')
    axs[1][0].set_title('$H_y$ field')
    axs[1][0].set_xlabel('X-axis (step)')
    axs[1][0].set_ylabel('Y-axis')
    axs[1][0].set_ylim(-2.5, 2.5)
    axs[1][0].set_xlim(0, 600)

    axs[0][1].plot(E_y)
    #axs[0][1].plot(np.vectorize(polarization_angle(E_y, E_z)))
    axs[0][1].set_title('$E_y$ field')
    axs[0][1].set_xlabel('X-axis (step)')
    axs[0][1].set_ylabel('Y-axis')
    axs[0][1].set_ylim(-2.5, 2.5)
    axs[0][1].set_xlim(0, 600)

    axs[1][1].plot(H_z, color='green')
    axs[1][1].set_title('$H_z$ field')
    axs[1][1].set_xlabel('X-axis (step)')
    axs[1][1].set_ylabel('Z-axis')
    axs[1][1].set_ylim(-2.5, 2.5)
    axs[1][1].set_xlim(0, 600)

    plt.show()



plt.figure()
plt.plot(angle_rec)
plt.title('Polarization angle with respect to the z-axis (in the y-z plane)')
plt.xlabel('time step')
plt.ylabel('angle (degrees)')