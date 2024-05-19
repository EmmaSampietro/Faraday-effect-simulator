# -*- coding: utf-8 -*-
"""
Created on Sat May 11 16:11:19 2024

@author: clem, sascha, emma
"""
import numpy as np
import matplotlib.pyplot as plt
import math

e_0 = 8.8541878128e-12 #vacuum permittivity
mu_0 = 1.256637062e-6  #vacuum permeability

c = 1/np.sqrt(e_0*mu_0) #speed of light
m_e = 9.1e-31     #electron mass
q_e = 1.6e-19     #electron charge
B_0 = 5e2         #magnitude external B field

m_max = 1300       #steps we want to perform (space)
n_max = 1850      #steps we want to perform (time)

m_source = 1

#Constants for the source
lambda_source = 550e-9 #wavelength of the source
w = 2*np.pi*c/lambda_source  #corresponding angular frequency
tau = 10e-15
t0 = tau*2         #center of the Gaussian pulse

#step sizes
dx = 25e-9         #step size on position
dt = 0.9*dx/c      #step size on time

#Constants we will use in our code
Sc = c*dt/dx       #Courant number
Z_0 = 1            #impedance of free space
n = 1e27           #density of electrons in the medium
gamma = 1          #damping coefficient
Sc_o_Z0=Sc/Z_0     #to minimize constants in our code
Z0Sc=Z_0*Sc        #to minimize constants in our code

w_p = np.sqrt(n/(m_e*e_0)) * q_e   #plasma frequency
w_c = (q_e*B_0)/m_e                #cyclotron frequency


#Initializing the arrays
E_z = np.zeros(m_max)   #arrays of zeros of length m_max
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

angle_rec=np.zeros(n_max)  #array of zeros of length n_max to record rotation of polarization angle in time
angle_rec_H=np.zeros(n_max)  



#Function for our gaussian wavepacket source
def Source_Function(t):
  E = np.exp(-(t-t0)**2/tau**2)*np.sin(w*t)     #gaussian wavepacket
  return E


#Function to compute the angle of polarization in the y-z plane
def polarization_angle(E_y,E_z):
    if E_y==0:
        degree_angle = np.pi/2
    else:
        radians_angle = math.atan(E_z/E_y)
        degree_angle = radians_angle*180/np.pi
    return degree_angle


#main loop
for n in range(n_max):
  #time iteration (recall n is time index)
  H_y[m_max-1]=H_y[m_max-2] #updating magnetic field boundaries
  H_z[m_max-1]=H_z[m_max-2]

  j_y[m_max-1]=j_y[m_max-2]
  j_z[m_max-1]=j_z[m_max-2]

  for m in range(m_max-1):
    #space iteration (recall m is space index)
    H_y[m] = H_y_previous[m] + Sc_o_Z0 *(E_z[m+1]-E_z[m]) #updating magnetic field y axis
    H_y_previous[m] = H_y[m]

    H_z[m] = H_z_previous[m] + Sc_o_Z0 *(E_y[m+1]-E_y[m]) #updating magnetic field z axis
    H_z_previous[m] = H_z[m]


  #Magnetic field source (along y axis since we have a z polarized wave)
  tn=n*dt
  H_z[m_source-1] -= Source_Function(tn)/Z_0
  H_z_previous[m_source-1] = H_z[m_source-1]


  for m in range(m_max-1): 
    #space iteration
    j_y[m]=(j_y_previous[m]*math.cos(w_c*dt)-j_z_previous[m]*math.sin(w_c*dt))*math.exp(-gamma*dt) + E_y[m]     #updating the current j
    j_z[m]=(j_y_previous[m]*math.sin(w_c*dt)+j_z_previous[m]*math.cos(w_c*dt))*math.exp(-gamma*dt) + E_z[m]

    j_y_previous[m]=j_y[m]
    j_z_previous[m]=j_z[m]


  E_z[0]=E_z_previous[1]  #updating electric field boudaries
  E_y[0]=E_y_previous[1]


  for m in range(1,m_max): #staggered grid so we must iterate starting at +1/ finish at m+1
    #space iteration
    E_z[m] = E_z_previous[m] + Z0Sc * (H_y[m] - H_y[m-1]) - (w_p*dt)**2 * (j_z[m])     #updating electric field
    E_y[m] = E_y_previous[m] + Z0Sc * (H_z[m] - H_z[m-1]) - (w_p*dt)**2 * (j_y[m])
    
    E_z_previous[m] = E_z[m]
    E_y_previous[m] = E_y[m]


  #Electric field source (along z axis since we have a z polarized wave)
  tnp1=(n+1)*dt
  E_y[m_source] += Source_Function(tnp1)
  E_y_previous[m_source] = E_y[m_source]


  #We record the polarization angle at this moment in time
  angle_rec[n] = polarization_angle(max(E_y),max(E_z))
  angle_rec_H[n] = polarization_angle(max(H_z),max(H_y))

  #If we are at the end of the simulation, we record the final polarization rotation angle 
  if n==n_max-1:
    final_rotation=angle_rec[n_max-1]-angle_rec[0]
    print("The final rotation angle from the z axis is of degrees ", final_rotation)


    final_rotation_H=angle_rec_H[n_max-1]-angle_rec_H[0]
    print("The final rotation angle from the z axis is of degrees ", final_rotation_H)


  #we plot the wavepacket
  if n%40 == 0:
    # Plotting the first plot
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 2 row, 2 columns
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    axs[0][0].plot(E_y)
    axs[0][0].set_title('$E_y$ field')
    axs[0][0].set_xlabel('X-axis (step)')
    axs[0][0].set_ylabel('Z-axis')
    axs[0][0].set_ylim(-2.5, 2.5)
    axs[0][0].set_xlim(0, 1000)

    axs[1][0].plot(H_z, color='green')
    axs[1][0].set_title('$H_z$ field')
    axs[1][0].set_xlabel('X-axis (step)')
    axs[1][0].set_ylabel('Y-axis')
    axs[1][0].set_ylim(-2.5, 2.5)
    axs[1][0].set_xlim(0, 1000)

    axs[0][1].plot(E_z)
    #axs[0][1].plot(np.vectorize(polarization_angle(E_y, E_z)))
    axs[0][1].set_title('$E_z$ field')
    axs[0][1].set_xlabel('X-axis (step)')
    axs[0][1].set_ylabel('Y-axis')
    axs[0][1].set_ylim(-2.5, 2.5)
    axs[0][1].set_xlim(0, 1000)

    axs[1][1].plot(H_y, color='green')
    axs[1][1].set_title('$H_y$ field')
    axs[1][1].set_xlabel('X-axis (step)')
    axs[1][1].set_ylabel('Z-axis')
    axs[1][1].set_ylim(-2.5, 2.5)
    axs[1][1].set_xlim(0, 1000)

    plt.show()
    


plt.figure()
plt.plot(angle_rec)
plt.title('Polarization angle with respect to the z-axis (in the y-z plane)')
plt.xlabel('time step')
plt.ylabel('angle (degrees)')


plt.figure()
plt.plot(angle_rec_H)
plt.title('Polarization angle with respect to the z-axis (in the y-z plane)')
plt.xlabel('time step')
plt.ylabel('angle (degrees)')