import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import fractions
import time
import cProfile
import re
from astropy.io import fits
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
from scipy.special import genlaguerre, gamma, hyp2f1, factorial, gammaln, gamma, zeta



'''
We want to create a module that takes a text file of values given by class as a function of redshift.
For example, we first want take in a redshift from a compilation of text files. 

'''


# Constants to be used in the cgs unit system (centimeters-grams-seconds)

c = 3e10 # Speed of light ( cm/s )
h = 6.626e-27 # Planck constant ( erg*s or cm^2 g / s )
hbar = h / (2*np.pi) # Reduced Planck constant in erg*s
e0 = 4.8032e-10 # One unit of electric charge   ( cm^3/2 * g^1/2 / s )
G_Newton = 6.6743e-8 # Newton's constant (cm^3 g^-1 s^-2)
m_electron = 9.10938e-28 # Electron rest mass ( grams )
m_proton = 1.673e-24 # Proton mass (g)
eV_to_ergs = 1.60218e-12 # eV to ergs conversion 
kB = 1.3807e-16 # Boltzmann constant (cm^2 g / s^2 K)
#baryon_to_photon_ratio = 5.9e-10 # Number of Baryons per photons
#baryon_to_photon_ratio = 6.13027e-10 
He_abund = 0.25 # Primordial Helium Abundance

mag_field = 1e-12 # Magnetic Field Strength (Gauss)

prim_curvature_pert = 2e-5 # Primordial curvature perturbation

# Quantities calculated using the above constants of nature

Bohr_radius = hbar**2 / (m_electron * e0**2 ) # Bohr radius ( cm )
larmor_freq = 1.3996e6 * mag_field # Lahmor frequency ( s^-1 )
ion_energy = m_electron*e0**4/(2*hbar**2) 

# Quantum numbers

S = 0.5 # Electron spin quantum number
I = 0.5 # Nuclear spin quantum number

# Values considered when summing over quantum numbers to determine 

                                                                                                          
numN = 3 # Largest N value considered. Will go from 1 to Nmax.
numL = numN # Largest L value considered.
numJ = 2 # Number of allowed J values other than the L=0 case which there is only one.
numK = 2 # Sum from 0 to 2
numF = 2 # Number of allowed F values given a value of J.

# We want to integrate across a wide range of energies. Here is where we set bounds on what
# we are actually integrating across.

numE = 10000 # Pick a number of elements to integrate across
Emax = 100*eV_to_ergs #Pick a maximum energy for free energy of electron


# We want to develop a program where we can index our matrices correctly.


'''
We wish to calculate find the wave number line that is closest to the entered value. This is 
set by the value of k_error which we set to 1e-3.
'''

# We want to upload the current redshift+rho1s values from our text file in master_program

arg_val = np.loadtxt("argv_entrees.txt")


# Input redshift and wave number

#redshift = 1276.835
redshift = arg_val[0]

print(" ")
print("Redshift of this iteration of new_matrices_code.py: ",redshift)
print (" ")

wave_num = 1.042057218184e-05

# Bollian variables for finding redshift and T(k) data

found_kline = False
found_zline = False

# Opens the thermodynamics file and the T(k) file
    
thermo_array = np.loadtxt(open("thermodynamics.dat"))
tk_array = np.loadtxt(open("tk_"+str(redshift)+".dat"))

# Boolian statement to find lines in file that we can use for a given redshift


while found_kline == False:
    for line in range(len(tk_array)):
    
        #print(line)
    
        if tk_array[line][0] == wave_num :
            #print("The wave number in the file is "+str(tk_array[line][0]))
            #print("The wave number is in line "+str(line))
            
            kline = line
            found_kline = True
    
    if found_kline == False:        
        wave_num = input("Wave number not found. Please input a new wave number: ")
    


while found_zline == False:
    for line in range(len(thermo_array)):
        
        if thermo_array[line][1] == redshift:
            #print("The redshift in the file is "+str(thermo_array[line][1]))
            #print("The redshift is in line "+str(line))
            
            zline = line
            found_zline = True
    
    if found_zline == False:        
        redshift = input("Redshift not found. Please input a new redshift: ")


# We acquire all the synchronous gauge variables from the CLASS data that may or may not be used


wave_num = tk_array[kline][0]
delta_g = tk_array[kline][1] * prim_curvature_pert
delta_b = tk_array[kline][2] * prim_curvature_pert
phi = tk_array[kline][7]
psi = tk_array[kline][8]
h_scalar = tk_array[kline][9]
h_prime = tk_array[kline][10]
eta_scalar = tk_array[kline][11]
eta_prime = tk_array[kline][12]
t_g = tk_array[kline][15]
t_b = tk_array[kline][16]
shear_g = tk_array[kline][18]
pol_0 = tk_array[kline][19]
#pol_2 = tk_array[kline][20]




# Monopole and Quadrapole perturbation variables

Theta_0 = delta_g * prim_curvature_pert
Theta_2 = 2*shear_g * prim_curvature_pert

                
# Calculate what the Hubble parameter based on Plack 2018 best fit values. (or CLASS values)


T0 = 2.7255
T = T0*(1+redshift)
T_star = 0.0681 # Kelvin

little_h = 0.67810 # H0 = gamma_100*h
gamma_100 = 3.24078e-18 # gamma_100 = 100 km/s/Mpc

rho_crit_today = 3*gamma_100**2*little_h**2/ (8*np.pi*G_Newton)
rho_rad_today = (np.pi**2/15) * (kB*T0)**4 / (hbar**3 * c**5)

# omega_X = Omega_X * h^2 = (rho_X/rho_crit) * h^2

omega_cdm = 0.1201075
omega_b = 0.02238280
omega_m = omega_b + omega_cdm
omega_r = (rho_rad_today / rho_crit_today)*little_h**2
omega_DE = little_h**2 - omega_m - omega_r 

num_baryon_today = (omega_b*rho_crit_today)/(m_proton*little_h**2)
num_photon_today = (2*zeta(3)/np.pi**2)*(kB*T0/(hbar*c))**3
baryon_to_photon_ratio = num_baryon_today / num_photon_today

# Hubble parameter H(z) where H(z=0) = H0

H0 = little_h*gamma_100
H_param = gamma_100 * np.sqrt(omega_r*(1+redshift)**4 + omega_m*(1+redshift)**3  + omega_DE)

# Converting wave number to units of 1/cm

wave_num = wave_num*little_h/3.086e24

# Calculating the rho_1s matrix elements based on the last iteration


rho_1s_Fis0_val = arg_val[1]
rho_1s_Fis1_val = arg_val[2]
rho_1s_Kis2_val = arg_val[3]


X_1s = rho_1s_Fis0_val + np.sqrt(3)*rho_1s_Fis1_val
X_e = 1-X_1s




#X_e = thermo_array[zline][3]
#X_1s = 1-X_e



print("X_1s:", X_1s)
print("X_e:", X_e)





'''

dipole_bf(Nmax, E_Free, array_type) takes in a maximum upper state(Nmax), takes the energy of 
the free electron (E_free), and then dictates which dipole array you will recieve (array_type).
Note that 

array_type == True will give you g(N,l;k,l+1)
array_type == False will give you g(N,l-1;k,l)

where g(N,l;k,l^{'}) = Bohr_Radius*e0 * integral( Free_electron*wf * r * Bound_wf r^2 dr)

One can show in Burgess et. al 1965 that the dipole integrals obey a recursion relation
which we impliment in this code.
'''

def wigner_3j_explicit(a,b,c,d,e,f):

    if c==0 and f==0:
        term = (-1)**(a-d) / np.sqrt(2*a+1)
    elif c==2 and a==2 and a==b:
        term = -np.sqrt(2/35)
    else:
        term = float(wigner_3j(a,b,c,d,e,f))
        
    return term
    
    
def wigner_6j_explicit(a,b,c,d,e,f):
    

    if (a-b)== 1:
        a = b
        b += 1
        
        temp = d
        d= e
        e = temp
                
    if np.abs(a-b) <= c and a+b >= c and np.abs(a-e) <= f and a+e >= f and np.abs(d-e) <= c and d+e >= c and np.abs(b-d) <= f and b+d >= f:   
        if a == b and e==d and c==0:
    
            term = (-1)**(a+e+f) / (np.sqrt(2*a+1)*np.sqrt(2*d+1))
    
        elif a==b and d==e and c==1:
    
            term = (-1)**(a+e+f+1)/2
            term *= a*(a+1) + e*(e+1) - f*(f+1)
            term *= 1/np.sqrt(a*(a+1)*(2*a+1)*e*(e+1)*(2*e+1))
        
        elif b==(a+1) and e==(d-1) and c==1:
     
            term = (-1)**(a+e+f)/2
            term *= np.sqrt((a+e+f+3)*(a+e+f+2)*(a+e-f+2)*(a+e-f+1))
            term *= 1/np.sqrt((a+1)*(2*a+1)*(2*a+3)*(e+1)*(2*e+1)*(2*e+3))
        
        elif b==(a+1) and d==e and c==1:
     
            term = (-1)**(a+e+f+1)/2
            term *= np.sqrt((a+e+f+2)*(a-e+f+1)*(a+e-f+1)*(f-a+e))
            term *= 1/np.sqrt((a+1)*(2*a+1)*(2*a+3)*e*(e+1)*(2*e+1))

        elif b==(a+1) and e==(d+1) and c==1:
     
            term = (-1)**(a+e+f)/2
            term *= np.sqrt((f-a+e)*(c-a+e-1)*(a-e+f+2)*(a-e+f+1))
            term *= 1/np.sqrt((a+1)*(2*a+1)*(2*a+3)*e*(2*e-1)*(2*e+1))
            
        else:
            term = float(wigner_6j(a,b,c,d,e,f))
    else:
        term = 0

    return term
    
def wigner_9j_explicit(a,b,c,d,e,f,g,h,i):

    
    if i==0 and c==f and g==h:
        term = (-1)**(b+c+d+g) / (np.sqrt(2*c+1)*np.sqrt(2*g+1))
        term *= wigner_6j_explicit(a,b,c,e,d,g)
    else:
        term = float(wigner_9j(a,b,c,d,e,f,g,h,i))
        
    return term
    
    
   
      
       
    
         
    
def energy_noF(N, L, J):

    # Unperturbed energy of the Hydrogen atom
    energy_0th = -ion_energy / N**2

    # NOTE: Need to add perturbed terms to this quantity later.

    return energy_0th



'''

The function dipole_element takes input from the principal quantum number, n, the azimuthal 
angular momentum, l or L, and the total angular momentum of the electron, J = L + S, for two
separate levels. It will compute the reduced matrix element

< n0, l0, J0 || \vec{d} || n1, l1, J1 >

and return the numerical value as output. In principal, we are interested in the absolute value
squared of this quantity which this function does not return. However, you can easily take the
absolute value squared once you call this function for some set of (n0, l0, J0) and (n1, l1, J1).
'''

def dipole_element(n0, n1, L0, L1, J0, J1):

    # Need to determine which L is larger.

    if L0 == L1-1 and n0 != n1 and np.abs(J0-J1) <= 1 :

        #print("first one")
    
        l = L1
        n_prime = n0 
        n = n1
        
        #print("step 1")
        term = np.sqrt(l) / np.sqrt(2*l-1)
        #print("step 2", term)
        term *= (-1)**(n_prime-1)/(4*factorial(2*l-1))
        #print("step 3", term)
        term *= np.exp( 0.5*gammaln(n+l+1) + 0.5*gammaln(n_prime+l) - 0.5*gammaln(n-l) - 0.5*gammaln(n_prime-l+1) )
        #print("step 4", term)
        term *= (4*n*n_prime)**(l+1)*(n - n_prime)**(n+n_prime-2*l-2) / (n+n_prime)**(n+n_prime)
        #print("step 5", term)
        term *= ( hyp2f1(-n+l+1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2 ) - (n-n_prime)**2/(n+n_prime)**2 * hyp2f1(-n+l-1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2) )
        
        
        

    elif L0 == L1+1 and n0 != n1 and np.abs(J0-J1) <= 1:

        l = L1+1
        n_prime = n1
        n = n0

        #print("step 1")
        term = np.sqrt(l) / np.sqrt(2*l+1)
        #print("step 2", term)
        term *= (-1)**(n_prime-1)/(4*factorial(2*l-1))
        #print("step 3", term)
        term *= np.exp( 0.5*gammaln(n+l+1) + 0.5*gammaln(n_prime+l) - 0.5*gammaln(n-l) - 0.5*gammaln(n_prime-l+1) )
        #print("step 4", term)
        term *= (4*n*n_prime)**(l+1)*(n - n_prime)**(n+n_prime-2*l-2) / (n+n_prime)**(n+n_prime)
        #print("step 5", term)
        term *= ( hyp2f1(-n+l+1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2 ) - (n-n_prime)**2/(n+n_prime)**2 * hyp2f1(-n+l-1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2) )

    else:

        term = 0

    return Bohr_radius*e0*term 


def Hubble_pert(K, h_scalar, h_prime, eta_scalar, baryon_vel):
        
    #cov_velocity = baryon_vel/3 - Hubble_param*Phi - Psi_dot
    cov_velocity = baryon_vel/3 + h_prime/6
    shear_33 = 2*(h_scalar + 6*eta_scalar)/3
        
    if K == 0:
        term = cov_velocity
    elif K == 2:
        term = shear_33
    else:
        term = 0
    
    return term
    
 
def B_Einstein(N0,N1,L0,L1,J0,J1):
 
     term = 32*np.pi**4/(3*h**2*c)*dipole_element(N0,N1,L0,L1,J0,J1)**2
     term *= (2*L0+1)*(2*J1+1)*wigner_6j_explicit(L1,L0,1,J0,J1,0.5)**2
     
     return term
     
def A_Einstein_coeff(N0,N1,L0,L1,J0,J1):

    freq = (energy_noF(N0,L0,J0) - energy_noF(N1,L1,J1))/h
    freq = np.abs(freq)
    
    term = 64*np.pi**4*freq**3/(3*h*c**3) * dipole_element(N0,N1,L0,L1,J0,J1)**2
    term *= (2*L0+1)*(2*J1+1)*wigner_6j_explicit(L1,L0,1,J0,J1,0.5)**2
    
    return term
    
       


# Defining the Lya frequency and Einstein coefficient to calculate the Sobolev optical depth

freq_lya = (energy_noF(2,1,0.5) - energy_noF(1,0,0.5))/h

#A_2p = 64*np.pi**4*freq_lya**3/(3*h*c**3) * dipole_element(2,1,1,0,0.5,0.5)**2
A_2p = A_Einstein_coeff(2,1,1,0,0.5,0.5)

# Number Density Information

num_H_tot = (1-He_abund)*num_baryon_today*(1+redshift)**3 # hydrogen number density
optical_depth = 3*np.pi**2 * A_2p * num_H_tot*(1-X_e)*c**3/ ( H_param * (2*np.pi*freq_lya)**3 ) # Lya optical depth

# Escape Probability

P_esc = 1/optical_depth 

# 2 photon Einstein rate

A_2photon = 8.22 # s^{-1}
    
    
# Einstein coefficients and Phase Space factors

freq_2p1s = (energy_noF(2,1,0.5)-energy_noF(1,0,0.5))/h
freq_3s2p = (energy_noF(3,0,0.5) - energy_noF(2,1,0.5))/h
freq_3p2s = (energy_noF(3,1,0.5) - energy_noF(2,0,0.5))/h
freq_3d2p = (energy_noF(3,2,1.5) - energy_noF(2,1,0.5))/h


'''
B_1s2p = 32*np.pi**4/(3*h**2*c) * dipole_element(1,2,0,1,0.5,0.5)**2
B_2s3p = 32*np.pi**4/(3*h**2*c) * ( dipole_element(2,3,0,1,0.5,0.5)**2 + dipole_element(2,3,0,1,0.5,1.5)**2 )
B_2p3s = 32*np.pi**4/(3*h**2*c) * dipole_element(2,3,1,0,0.5,0.5)**2
B_2p_1half_3d = 32*np.pi**4/(3*h**2*c) * dipole_element(2,3,1,2,0.5,1.5)**2
B_2p_3half_3d = 32*np.pi**4/(3*h**2*c) * (dipole_element(2,3,1,2,1.5,1.5)**2 + dipole_element(2,3,1,2,1.5,2.5)**2 )


A_2p1s = 64*np.pi**4*freq_2p1s**3/(3*h*c**3) * dipole_element(2,1,1,0,0.5,0.5)**2
A_3s2p = 64*np.pi**4*freq_3s2p**3/(3*h*c**3) * dipole_element(3,2,0,1,0.5,0.5)**2
A_3p2s = 64*np.pi**4*freq_3p2s**3/(3*h*c**3) * dipole_element(3,2,1,0,1.5,0.5)**2
A_3d2p = 64*np.pi**4*freq_3d2p**3/(3*h*c**3) * dipole_element(3,2,2,1,1.5,1.5)**2

phase_space_21 = 1/(np.exp(h*freq_2p1s/(kB*T))-1)
phase_space_32 = 1/(np.exp(h*freq_3p2s/(kB*T))-1)
'''






def rad_field_tensor_BB(N0,N1,L0,L1,J0,J1,K0,K1,Kr,F0,F1,F2,F3,T,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,pert_index):

    
    # Blackbody function stuff
    
    x = h*freq_lya/(kB*T)
    weight = 2*h*freq_lya**3/c**2
        
    if x > 0:
        phase_space = np.exp(-x)/(1-np.exp(-x))
        phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2
    
    # IF/then arguments for different parts of the radiation field tensor

    term = 0
       
    if (N0==1 and N1==2 and L0==0 and L1==1 and J0==0.5) or (N0==2 and N1==1 and L0==1 and L1==0 and J1==0.5) or (N0==1 and N1==1 and L0==0 and L1==0 and J0==0.5 and J1==0.5):
        
        if  Kr == 0 and pert_index == False:
        
            # Blackbody term: P_esc*B_P(freq_lya,T)
        
            term = P_esc*weight*phase_space
        
        elif K0 == 0 and K1 == 0 and Kr == 0 and pert_index == True:
            
            # Perturbed K_r=0 Blackbody term
            
            term = P_esc*weight*phase_space*Hubble_pert(0, h_scalar, h_prime, eta_scalar, t_b)
            term += -2*P_esc*(kB*T)**3*x**4*phase_deriv*Theta_0/(h*c)**3
            
        elif K0 == 2 and K1==0 and Kr==2 and pert_index == True:
            
            # Perturbed K_r=2 Blackbody term
            
            term = P_esc*weight*phase_space*Hubble_pert(2, h_scalar, h_prime, eta_scalar, t_b)
            term += np.sqrt(2)*P_esc*(kB*T)**3*x**4*phase_deriv*Theta_2/(h*c)**2
            
        else:
            
            term = 0
            
    return term


def Lambda_lymann_lines_BB(N0,N1,L0,L1,J0,J1,K0,K1,Kr,F0,F1,F2,F3,T,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,pert_index):
       
    
    # Blackbody function stuff
    
    x = h*freq_lya/(kB*T)
    weight = 2*h*freq_lya**3/c**2
    #print(weight)
    
    phase_space = np.exp(-x)/(1-np.exp(-x))
    phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2
    
    # IF/then arguments for different parts of the radiation field tensor

    # Calculating the Einstein Coefficients for every calculation

    B_Einstein_abs = B_Einstein(1,2,0,1,0.5,J1)
    B_Einstein_stim = B_Einstein(2,1,1,0,J0,0.5)
    A_Einstein = A_Einstein_coeff(2,1,1,0,J0,0.5)
    
    
    #print("Absorption B coefficient:", B_Einstein_abs)
    #print("Stimulared emission A coefficient:", A_Einstein)
    
    # All the T_A (transmission absorbtion terms) are listed below:

    term = 0

    if N0 == 2 and L0 == 1 and N1 == 1 and L1 == 0 and J1 == 0.5:
        
        if Kr==0 and K0==K1 and pert_index==False:
            
            term = P_esc*A_Einstein*phase_space
            term *= (2*J0+1)*np.sqrt(3*(2*F2+1)*(2*F3+1)*(2*F0+1)*(2*F1+1)*(2*K0+1)) 
            #print(term)
            #term *= float( wigner_9j(F0, F2, 1, F1, F3, 1, K0, K0, 0) * wigner_6j(J0,0.5,1,F2, F0, 0.5) * wigner_6j(J0, 0.5, 1, F3, F1, 0.5) )
            term *= wigner_9j_explicit(F0,F2,1,F1,F3,1,K0,K0,0)*wigner_6j_explicit(J0,0.5,1,F2,F0,0.5)*wigner_6j_explicit(J0,0.5,1,F3,F1,0.5)
            
        elif Kr==0 and K0==0 and K1==0 and F0==F1 and F2==F3 and pert_index==True:
            
            term = P_esc*A_Einstein
            #term *= (2*J0+1)*np.sqrt((2*F0+1)*(2*F2+1)) * float( wigner_6j(J0, 0.5, 1, F0, F2, 0.5) )**2
            term *= (-x*phase_deriv *Theta_0 + phase_space*Hubble_pert(0, h_scalar, h_prime, eta_scalar, t_b))
            term *= (2*J0+1)*np.sqrt((2*F0+1)*(2*F2+1)) * wigner_6j_explicit(J0,0.5,1,F0,F2,0.5)**2
            
        elif Kr==2 and K0==2 and K1==0 and F2==F3 and pert_index==True:
            
            term = 3*P_esc*A_Einstein*x*phase_deriv*Theta_2
            term *= (2*J0+1)*(2*F2+1)*np.sqrt(30*(2*F0+1)*(2*F1+1))
            term *= wigner_9j_explicit(F0,F2,1,F1,F2,1,2,0,2) 
            #term *= float( wigner_6j(J0,J1,1,F2,F0,0.5)*wigner_6j(J0,J1,1,F2,F1,0.5) )
            term *= wigner_6j_explicit(J0,J1,1,F2,F0,0.5) * wigner_6j_explicit(J0,J1,1,F2,F1,0.5)
        else:
            term = 0


    # All the R_A (relaxation absorption terms) are listed below:

    if N0==1 and L0==0 and N1==1 and L1==0 and J0==J1 and J0==0.5:       
        if Kr==0 and K0==K1 and F0==F2 and F1==F3 and F0==F1 and pert_index==False:
                
            term = -3*P_esc*phase_space*A_Einstein
            #print(term)
                

        if Kr==0 and K0==K1 and F0==F2 and F1==F3 and F0==F1 and pert_index==True:
            
             term = -3*P_esc*phase_space*A_Einstein*(-x*phase_deriv *Theta_0 + phase_space*Hubble_pert(0, h_scalar, h_prime, eta_scalar, t_b))
         
                         
    return term

       
    

def Lambda_lymann_lines_source(N0,N1,L0,L1,J0,J1,K0,K1,Kr,F0,F1,F2,F3,T,X_e,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,pert_index):

    
    # Escape probability
    
    P_esc = 1/optical_depth
    
    # Blackbody function stuff
    
    x = h*freq_lya/(kB*T)
    weight = 2*h*freq_lya**3/c**2
        
    phase_space = np.exp(-x)/(1-np.exp(-x))
    phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2
    
    # IF/then arguments for different parts of the radiation field tensor

    # Depending on whether N0>N1 or N0<=N1 we will have different coefficients
    # that we need to take account of. 
        
    # To 0th order, we can estimate the population in the 1s state given the CMB and spin temperature

    T_star = 0.0681 # Kelvin
    
    rho_1s = np.zeros(3)
    
    '''
    X_1s = 1-X_e
    X_1s_singlet = X_1s/(1+3*np.exp(-T_star/T)) # F=0 state
    X_1s_triplet = 3*X_1s_singlet*np.exp(-T_star/T) # F=1 state
       
    rho_1s[0] = X_1s_singlet
    rho_1s[1] = X_1s_triplet/np.sqrt(3)
    rho_1s[2] = 0
    '''
    
    
    
    rho_1s[0] = float(rho_1s_Fis0_val)
    rho_1s[1] = float(rho_1s_Fis1_val)
    rho_1s[2] = float(rho_1s_Kis2_val)
    
    
    #print("rho[0]:", rho_1s[0])
    #print("rho[1]:", rho_1s[1])
    
   
    # Einstein coefficients on whether N0>N1 or N0<N1

    '''
    B_Einstein_abs = 32*np.pi**4/(3*h**2*c)
    B_Einstein_abs *= np.abs( dipole_element(1,2,0,1,J0,J1))**2    

        
    B_Einstein_stim = 32*np.pi**4/(3*h**2*c)
    B_Einstein_stim *= np.abs( dipole_element(2,1,1,0,J0,J1))**2 
    A_Einstein = weight*B_Einstein_stim
    '''
    
    B_Einstein_abs = B_Einstein(1,2,0,1,0.5,J1)
    B_Einstein_stim = B_Einstein(2,1,1,0,J0,0.5)
    A_Einstein = A_Einstein_coeff(2,1,1,0,J0,0.5)   
    #print("A_Einstein:", A_Einstein)
       
    #print("Probability:", P_esc)
    #print("Einstein coefficient:", B_Einstein_abs)
        
    
    # Listing all the source terms related to T_A

    term = 0
    #print("term_original:", term+1)            

   
    if N0==N1 and N0==2 and L0==L1 and L0==1:
    
        if Kr==0 and K0==0 and K1==0 and F0==F1 and F2==F3 and pert_index == False:
        
            term = (1-P_esc)*A_Einstein
            term *= np.sqrt((2*F0+1)*(2*F2+1) )/12
        
            '''
            term = (1-P_esc)*A_Einstein 
            term *= (2*J0+1)*np.sqrt((2*F0+1)*(2*F2+1) )
            #print("Term:", term)
            
            sum_1s = 0
            
            for F_l in range(2):
            
                sum_1s += (2*F_l+1) * float(wigner_6j(J0, 0.5, 1, F_l, F0, 0.5) )**2 /4
                
            #print("sum_1s:", sum_1s)    
            term *= sum_1s
            #print("term:", term)
            '''
                
        elif Kr==0 and K0==0 and K1==0 and F2==F3 and pert_index == True:
        
            term = P_esc*A_Einstein 
            term *= np.sqrt((2*F0+1)*(2*F2+1) )/12
            term *= delta_b - Hubble_pert(0, h_scalar, h_prime, eta_scalar,t_b)
            
            '''
            sum_1s = 0
            
            for F_l in range(2):
            
                sum_1s += (2*F_l+1)*float(wigner_6j(J0, 0.5, 1, F_l, F0, 0.5) )**2 /4
                
            term *= sum_1s 
            
            '''
            
            
                               
    
        elif Kr == 2 and K0==2 and K1==0 and F2==F3 and pert_index == True:
        
                    
            term1 = (1-P_esc)*A_Einstein 
            term1 *= 3*np.sqrt(15)*(2*J0+1)*np.sqrt((2*F0+1)*(2*F1+1)*(2*F2+1))
            #term1 *= float( wigner_9j(F0,1,1,F1,1,1,2,2,0)*wigner_6j(J0,0.5,1,1,F0,0.5)*wigner_6j(J0,0.5,1,1,F1,0.5))
            term1 *= wigner_9j_explicit(F0,1,1,F1,1,1,2,2,0)*wigner_6j_explicit(J0,0.5,1,1,F0,0.5)*wigner_6j_explicit(J0,0.5,1,1,F1,0.5)
            
            term1 *= rho_1s[2]/X_1s
            
            term2 = -P_esc*A_Einstein*Hubble_pert(2,h_scalar, h_prime, eta_scalar,t_b)
            term2 *= (2*J0+1)*np.sqrt(15*(2*F0+1)*(2*F1+1)*(2*F2+1))
            
            sum_1s = 0
                        
            for F_l in range(2):
            
                sum_1s += (2*F_l+1)*wigner_9j_explicit(F0,F_l,1,F1,F_l,1,2,0,2)
                sum_1s *= wigner_6j_explicit(J0,0.5,1,1,F0,0.5)*wigner_6j_explicit(J0,0.5,1,1,F1,0.5)
                sum_1s *= rho_1s[F_l]/X_1s
                
            term2 *= sum_1s
            
            term = term1 + term2
               
        
        else:
            term = 0


    # All the R_A (relaxation absorption terms) are listed below:
    
    if N0==1 and L0==0 and N1==2 and L1==1:       
            
        if Kr==0 and K0==K1 and K0 == 0 and F0==F1 and F2==F3 and pert_index==False:

            #print("F0", F0)
            #print("rho_1s:", rho_1s)             
            #term = -3*(1-P_esc)*np.sqrt(2*F2+1)*A_Einstein
            term = -(1-P_esc)*A_Einstein*np.sqrt((2*F2+1)*(2*F0+1))/4
            
        elif Kr==0 and K0==K1 and F0==F1 and F2==F3 and pert_index==True:
                                       
            #term *= -(1-P_esc)*np.sqrt(2*F2+1)*rho_1s[int(F0)] / X_1s
            #term *= ( delta_b/X_1s - Hubble_pert(0, h_scalar, h_prime, eta_scalar, t_b) )/3

            term = -P_esc*np.sqrt((2*F2+1)*(2*F0+1)) *A_Einstein*( delta_b - Hubble_pert(0, h_scalar, h_prime, eta_scalar, t_b) )/4        
            # Numerical factor depending on value of F0==0,1

        elif Kr==2 and K0==2 and K1==0 and F0==1 and F1==1 and F2==F3 and pert_index==True:

            term = -(1-P_esc)*A_Einstein*np.sqrt(2*F2+1)
            term *= rho_1s[2]/X_1s    

        else:
            term = 0
               

    return term    
        

def dipole_bf(Nmax, E_free, array_type):
    
    # The wave number in cgs units is given by
    
    k = np.sqrt(E_free/ion_energy)/Bohr_radius
    
    # Define a dimensionless variable x = k*Bohr_radius
    
    x = k*Bohr_radius
        
    # 2D array that has dimensions of (N,N-1)
    
    dipole_bf_array = np.zeros( (Nmax+1,Nmax))
    
    #######################################################################
    ########################    Recursion Relation    #####################
    #######################################################################
    
    
    # We want to calculate the g(n,n-1;k,n) term first.
    
    if array_type == True:
        
        # Calculating the original term 
        
        
        for N in range(1,Nmax+1,1):
            
                               
            # Redundant information, but this is the conversion factor to get Landau's normalization
            # compared to Burgess et. al 1965. We multiply by np.sqrt()
                                
            g_first = np.sqrt(2*x/np.pi) * N**2 # constant to go from Burgess g to Landau g
            
            g_first *= 4*np.sqrt(np.pi/2)
            g_first *= 10**5
            
            g_first *= np.exp(-np.log(10**5))
            g_first *= np.exp(N*np.log(4*N) - 2*N - 0.5*gammaln(2*N) )
            g_first *= np.exp(2*N - 2*np.arctan(N*x)/x - (N+2)*np.log(1+N**2*x**2))
            g_first *= np.exp( - 0.5*np.log(1-np.exp(-2*np.pi/x)))
            
            for s in range(1, N+1, 1):
                
                g_first *= np.exp(0.5*np.log(1+s**2*x**2))
            
            
            '''
            g_first = 4*np.sqrt(np.pi/(2*np.math.factorial(2*N-1))) * (4*N)**N * np.exp(-2*N)
            #print(g_first)
    
    
            g_first *= 1/np.sqrt(1- np.exp(-2*np.pi/k))
            g_first *= np.exp(2*N - 2*np.arctan(N*k)/k) / (1+N**2*k**2)**(N+2)
    
            # Adding on the normalization constant
    
            for s in range(1,N+1,1):
        
                g_first *= np.sqrt(1+s**2*k**2)
                
            '''

            # We want to calculate the g(n,n-2; k, n-1) term next
        
            g_second = 0.5*np.sqrt( (2*N-1)*(1+N**2 * x**2)) * g_first
    
            # We want to assign values to our array as output


            if N-2 > -1:    
                dipole_bf_array[N, N-1] = g_first
                dipole_bf_array[N, N-2] = g_second
            elif N-1 > -1:
                dipole_bf_array[N, N-1] = g_first
        
            
    

           # Recursion relation which loops backwards to find each next value.
    
            for L in range(N-1,1,-1):        

                g_term = ( 4*N**2 - 4*L**2 + L*(2*L-1)*(1+ N**2*x**2))
                g_term *= g_second
                g_term += -2*N*np.sqrt( (N**2-L**2)*( 1 + (L+1)**2*x**2))*g_first
                g_term = g_term / ( 2*N*np.sqrt( (N**2 - (L-1)**2)*(1+L**2*x**2)))
        
                dipole_bf_array[N, L-2] = g_term
        
                g_first = g_second
                g_second = g_term
                
    else:
        
        for N in range(1,Nmax+1,1):
            

            g_first = np.sqrt(2*x/np.pi) * N**2 # constant to go from Burgess g to Landau g
            
            g_first = 4*np.sqrt(np.pi/2)
            g_first *= 10**5
            
            g_first *= np.exp(-np.log(10**5))*np.exp(N*np.log(4*N) - 2*N - 0.5*gammaln(2*N) )
            g_first *= np.exp(2*N - 2*np.arctan(N*x)/x - (N+2)*np.log(1+N**2*x**2))
            g_first *= np.exp( - 0.5*np.log(1-np.exp(-2*np.pi/x)))
            
            for s in range(1, N+1, 1):
                
                g_first *= np.exp(0.5*np.log(1+s**2*x**2))
       
            
            
            '''
            g_first = 4*np.sqrt(np.pi/(2*np.math.factorial(2*N-1))) * (4*N)**N * np.exp(-2*N)
            print(g_first)
    
    
            g_first *= 1/np.sqrt(1- np.exp(-2*np.pi/k))
            g_first *= np.exp(2*N - 2*np.arctan(N*k)/k) / (1+N**2*k**2)**(N+2)
    
            # Adding on the normalization constant
    
            for s in range(1,N+1,1):
        
                g_first *= np.sqrt(1+s**2*k**2)
                
            '''

            # We want to calculate the g(n,n-1; k, n-2) term first
        
            g_first *= np.sqrt((1+N**2*x**2)/(1 + (N-1)**2*x**2)) / (2*N)
            
            
            # We want to now calculate the g(n,n-2;k,n-3) term
                
            g_second = (4 + (N-1)*(1+N**2*x**2))/(2*N)
            g_second *= np.sqrt( (2*N-1)/(1+(N-2)**2*x**2))
            g_second *= g_first

                
            # We want to assign values to our array as output.
            # Note that we are assigning to the array for the L's in the Bound state.
            # Therefore, the L = N-2 term will correlate with the [N,N-1] element, etc.
                
            if N-3>-1:
                dipole_bf_array[N,N-1] = g_first
                dipole_bf_array[N,N-2] = g_second
            elif N-2>-1:
                dipole_bf_array[N,N-1] = g_first
        

    

            # Recursion relation which loops backwards to find each next value.
    
            for counter in range(N-3, 0, -1):
                
                L = counter + 1  # The array element and the actual L are off by 1.         
        
                g_term = ( 4*N**2 - 4*L**2 + L*(2*L+1)*(1+ N**2*x**2))
                g_term *= g_second
                g_term += -2*N*np.sqrt( (N**2 - (L+1)**2)*(1+x**2*L**2)) * g_first
                g_term = g_term / ( 2*N*np.sqrt( (N**2 - L**2)*(1+(L-1)**2*x**2)))
                
                
                dipole_bf_array[N, counter] = g_term
        
                g_first = g_second
                g_second = g_term
                                                        
        
    return e0*Bohr_radius*dipole_bf_array




def source_boundfree_spontaneous(N, L, j, k, f0, f1, energy_array):

    term = 0
    
    # Computes total atomic and electron angular momentum
    J = np.arange( np.abs(L-S), L+S+1, 1)
    F = np.arange( np.abs(J[j]-I), J[j]+I+1, 1)
    #print(J)
    #print(F)
    

    # Prefactors
    
    prefactor = np.sqrt(2*F[f0]+1)
    prefactor *= 1/(2*J[j]+1)
    prefactor *= np.sqrt(m_electron/(2*hbar**2))
    prefactor *= X_e  / 2 
    #prefactor *= 1/2
    prefactor *= Bohr_radius
    #print("Prefactor before loop:", prefactor)
    
    # Number density of baryons (cm^{-3})
    
    '''
    num_den_proton = (2*zeta(3)/np.pi**2)
    num_den_proton *= (kB*T/ (hbar*c))**3
    num_den_proton *= baryon_to_photon_ratio*(1-He_abund)*X_e
    '''
    
    num_den_proton = (1-He_abund)*X_e*num_baryon_today*(1+redshift)**3
    
    #print ("Proton Number Density:", num_den_proton) 

    # Electron chemical potential (ergs)   
    
    #chemical_potential = m_electron*c**2 + kB*T*np.log(num_den_proton/2) 
    chemical_potential = kB*T*np.log(num_den_proton/2) 
    chemical_potential += 1.5*kB*T*np.log( 2*np.pi*hbar**2 / (m_electron*kB*T) )
    #print("Chemical Potential (no rest energy):", chemical_potential)
    
    # Computes the value if the bound state has L=0
    
    if L == 0:
             
        Lu = L + 1
                            
        Ju = np.arange(np.abs(Lu-S), Lu+S+1, 1)  
                
        for ju in range(len(Ju)):
                                                                                                                                                                                                                                                                                                                                                                                                                  
            temp = (2*Ju[ju] + 1)*prefactor
                    
            #hstep = np.abs(energy_array[1]-energy_array[0])
            #print(hstep)

            integral = hstep*np.sum(A_Einstein_array1[:,N,L,j,ju]*np.exp(-( energy_array - chemical_potential) /(kB*T)) / np.sqrt(energy_array))
               
            
            temp *= integral           
            term += temp
            
            if F[f0] < np.abs(J[j]-S) or F[f0] > J[j]+S or k==1 or f0 != f1:
                term = 0
    
    # Computes the bound state if L is not zero
        
    elif L > 0:
        
        for lu in range(2):
            
            Lu = L - 1 + 2*lu
            Ju = np.arange(np.abs(Lu-S), Lu+S+1, 1)  

            for ju in range(len(Ju)): 
                

                temp = (2*Ju[ju] + 1)*prefactor
                #print("Temp", temp)
                    
                #hstep = np.abs(energy_array[1]-energy_array[0])
                #print(hstep)
                
                if L < Lu:
                    
                    integral = hstep*np.sum(A_Einstein_array1[:,N,L,j,ju]*np.exp(-( energy_array - chemical_potential) /(kB*T)) / np.sqrt(energy_array))   
                    #print("Integral:",integral)
                    temp *= integral
                    term += temp
                    
                    '''
                    for i in range(1,numE-1):
                        
                        integral += hstep*A_Einstein_array1[i,N,L]*np.exp(-energy_array[i]/(kB*T))/np.sqrt(energy_array[i])
                
                        #print(integral) 
                
                    temp *= integral 
            
                    term += temp                 
                    '''
                                
                elif L > Lu:
                    
                    #integral = 0.5*hstep*A_Einstein_array2[0,N,L]*np.exp(-energy_array[0]/(kB*T))/np.sqrt(energy_array[0])
            
                    #integral += 0.5*hstep*A_Einstein_array2[numE-1,N,L]*np.exp(-energy_array[numE-1]/(kB*T))/np.sqrt(energy_array[numE-1]) 
                  
                    #print(integral)
                    

                    integral = hstep*np.sum(A_Einstein_array2[:,N,L,j,ju]*np.exp(-( energy_array - chemical_potential) /(kB*T)) / np.sqrt(energy_array))
                     
                    temp *= integral
                    term += temp
                    
                    '''
                    for i in range(1,numE-1):
                        
                        integral += hstep*A_Einstein_array2[i,N,L]*np.exp(-energy_array[i]/(kB*T))/np.sqrt(energy_array[i])
                
                        print(integral) 
                
                    temp *= integral 
            
                    term += temp   
                    '''
        
        # Making sure F is in the correct range
        
        if F[f0] < np.abs(J[j]-S) or F[f0] > J[j]+S or k==1 or f0 != f1:
            term = 0
        
    return term
    

def boundfree_photoionization(N,L,J,I,K,K_prime, Kr, F0,F1,F2,F3,X_e,pert_index):



    term = 0
    
    if L == 0 and F0==F2 and F1==F3 and F0==F1 and K==K_prime:
        
        Lu = L + 1
        
        Ju = np.arange( np.abs(Lu-S), Lu+S+1, 1)
        
        if J == 0.5:
            j = 0
            
        for ju in range(len(Ju)):
        
            prefactor = Bohr_radius*np.sqrt(m_electron/(2*hbar**2))
            #prefactor *= (2*Ju[ju]+1)/(2*J+1) # Conversion from bound-unbound to bound-unbound.
            prefactor *= (2*Ju[ju]+1)/(2*J+1)
            
            freq_bf = (energy_array- energy(N,L,J,I,F0))/h
            
            x = h*freq_bf/(kB*T)
            weight = 2*h*freq_bf**3/c**2            
            
            phase_space = np.exp(-x)/(1-np.exp(-x))
            phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2
              
            if pert_index == False:  
                integral = hstep*np.sum( A_Einstein_array1[:,N,L,j,ju]*phase_space/np.sqrt(energy_array))    
            else:
                integral = -hstep*np.sum( A_Einstein_array1[:,N,L,j,ju]*x*phase_deriv*delta_g/np.sqrt(energy_array))                     
                               
            term += prefactor*integral
            

    elif L > 0 and F0==F2 and F1==F3 and  F0==F1 and K==K_prime and pert_index==False:
        
        if J == np.abs(L-S):
            j = 0 # index for the first possible value of J: J = |L-S| and L > 0
        elif J == L+S:
            j = 1 # index for the second possible value of J: J = L+S and L > 0
       
            
        for lu in range(2):
                
            Lu = L-1+2*lu
            Ju = np.arange( np.abs(Lu-S), Lu+S+1, 1)
                
            for ju in range(len(Ju)):
                    
                prefactor = Bohr_radius*np.sqrt(m_electron/(2*hbar**2))
                #prefactor *= (2*Ju[ju]+1)/(2*J+1) # Conversion from bound-unbound to bound-unbound.                 
                prefactor *= (2*Ju[ju]+1)/(2*J+1)                
                
                freq_bf = (energy_array- energy(N,L,J,I,F0))/h
            
                x = h*freq_bf/(kB*T)
                weight = 2*h*freq_bf**3/c**2            

                    
                phase_space = np.exp(-x)/(1-np.exp(-x))
                phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2
                
                if L < Lu:
                                            
              
                    if pert_index == False:  
                        integral = hstep*np.sum( A_Einstein_array1[:,N,L,j,ju]*phase_space/np.sqrt(energy_array))    
                    else:
                        integral = -hstep*np.sum( A_Einstein_array1[:,N,L,j,ju]*x*phase_deriv*delta_g/np.sqrt(energy_array))    
                                                      
                elif L > Lu:
               
              
                    if pert_index == False:  
                        integral = hstep*np.sum( A_Einstein_array2[:,N,L,j,ju]*phase_space/np.sqrt(energy_array))    
                    else:
                        integral = -hstep*np.sum( A_Einstein_array2[:,N,L,j,ju]*x*phase_deriv*delta_g/np.sqrt(energy_array))    
                    
                term += prefactor*integral
                
                #print(integral)
 
     
    return term
         
'''                

def boundfree_photoionization(N, L, J, I, K, K_prime, Kr, F0, F1, F2, F3, X_e, pert_index):
    
   
    term = 0
    
    if L == 0 and F2==F3:
        
        Lu = L + 1
        
        Ju = np.arange( np.abs(Lu-S), Lu+S+1, 1)
        
        if J == 0.5:
            j = 0
        
        
        for ju in range(len(Ju)):
            
            term1 = 0
            term2 = 0
            
            prefactor = Bohr_radius*np.sqrt(m_electron/(2*hbar**2))
            prefactor *= (2*Ju[ju]+1)*np.sqrt(3*(2*K+1)*(2*K_prime+1)*(2*Kr+1))
            prefactor *= (-1)**(1+Ju[ju]-I+ F0)
            prefactor *= wigner_6j(J,J,Kr,1,1,Ju[ju])*wigner_3j(K,K_prime,Kr,0,0,0)
            prefactor = float(prefactor)
            print("Prefactor:", prefactor/(Bohr_radius*np.sqrt(m_electron/(2*hbar**2))))
            
            if F0 == F2:
                
                term1 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
                term1 *= wigner_6j(J,J,Kr,F3,F1,I)*wigner_6j(K,K_prime,Kr,F3,F1,F0)
                term1 = float(term1)
                print("Term1:", term1)
                
            if F1 == F3:
                
                term2 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
                term2 *= (-1)**(F2-F1)
                term2 *= wigner_6j(J,J,Kr,F2,F0,I)*wigner_6j(K,K_prime,Kr,F2,F0,F1)
                term2 = float(term2)
                print("Term2:", term2)
                
            integral = 0
                        
            for i in range(numE):
                
                freq_bf = ( energy_array[i] - energy(N,L,J,I,F0) )/h
                
                
               
                integral += hstep*B_Einstein_array1[i,N,L,j,ju]*rad_field_tensor(Kr, freq_bf, T, Theta_0, Theta_2, pert_index) / np.sqrt(energy_array[i])


                
                
                #print("integral sum:", integral)
                                     
            print ("Integral:", integral)
                
            temp = prefactor*(term1 + term2)*integral
            #print("Temp:", temp)
            
            term += temp
            #print("Term:", term)
            
    elif L > 0 and F0==F1:
        
        if J == np.abs(L-S):
            j = 0 # index for the first possible value of J: J = |L-S| and L > 0
        elif J == L+S:
            j = 1 # index for the second possible value of J: J = L+S and L > 0
       
            
        for lu in range(2):
                
            Lu = L-1+2*lu
            Ju = np.arange( np.abs(Lu-S), Lu+S+1, 1)
                
            for ju in range(len(Ju)):
                    
                term1 = 0
                term2 = 0
                  
                prefactor = Bohr_radius*np.sqrt(m_electron/(2*hbar**2))
                prefactor *= (2*J[ju]+1)*np.sqrt(3*(2*K+1)*(2*K_prime+1)*(2*Kr+1))
                prefactor *= (-1)**(1+Ju[ju]-I+ F0)
                prefactor *= wigner_6j(J,J,Kr,1,1,Ju[ju]) * wigner_3j(K,K_prime,Kr,0,0,0)
                prefactor = float(prefactor)
                print("Prefactor:", prefactor/Bohr_radius*np.sqrt(m_electron/(2*hbar**2)))
                        
                if F0 == F2:
                
                    term1 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
                    term1 *= wigner_6j(J,J,Kr,F3,F1,I)*wigner_6j(K,K_prime,Kr,F3,F1,F0)
                    term1 = float(term1)
                
                if F1 == F3:
                
                    term2 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
                    term2 *= (-1)**(F2-F1+K+K_prime+Kr)
                    term2 *= wigner_6j(J,J,Kr,F2,F0,I)*wigner_6j(K,K_prime,Kr,F2,F0,F1)
                    term2 = float(term2)  
                    
                integral = 0
                
                if L < Lu:
                    
                    for i in range(numE):
                
                        freq_bf = ( energy_array[i] - energy(N,L,J,I,F0) )/h
               
                        integral += hstep*B_Einstein_array1[i,N,L,j,ju]*rad_field_tensor(Kr, freq_bf, T, Theta_0, Theta_2, pert_index) / np.sqrt(energy_array[i])

                                    
                elif L > Lu:
                    
                    for i in range(numE):
                
                        freq_bf = ( energy_array[i] - energy(N,L,J,I,F0) )/h
               
                        integral += hstep*B_Einstein_array2[i,N,L,j,ju]*rad_field_tensor(Kr, freq_bf, T, Theta_0, Theta_2, pert_index) / np.sqrt(energy_array[i])
                
                
                temp = prefactor*(term1 + term2)*integral
                term += temp
                
                #print(integral)
                
    return term
'''                    
                    
                    
                
                
            
            
def source_boundfree_stimulated(N, L, j, k, f0, f1, pert_index, energy_array):

    term = 0
    
    # Computes total atomic and electron angular momentum
    K = 2*k
    J = np.arange( np.abs(L-S), L+S+1, 1)
    F = np.arange( np.abs(J[j]-I), J[j]+I+1, 1)
    

    # Prefactors
    
    if k==0:
        
        prefactor = np.sqrt(2*F[f0]+1)
        prefactor *= 1/(2*J[j]+1)
        prefactor *= np.sqrt(m_electron/(2*hbar**2))
        prefactor *= X_e  / 2
        prefactor *= Bohr_radius
        
    elif k==1:
        
        prefactor = np.sqrt( 3*(2*F[f0]+1)*(2*F[f1]+1) )
        prefactor *= np.sqrt(m_electron/(2*hbar**2))
        prefactor *= X_e / 2
        prefactor *= Bohr_radius
        
        
    #print("Prefactor before loop:", prefactor)
    
    # Number density of baryons (cm^{-3})
    
    '''
    num_den_proton = (2*zeta(3)/np.pi**2)
    num_den_proton *= (kB*T/ (hbar*c))**3
    num_den_proton *= baryon_to_photon_ratio * (1-He_abund)*X_e
    '''
    
    num_den_proton = (1-He_abund)*X_e*num_baryon_today*(1+redshift)**3
    #print ("Proton Number Density:", num_den_proton) 

    # Electron chemical potential (ergs)   
    
    #chemical_potential = m_electron*c**2 + kB*T*np.log(num_den_proton/2) 
    chemical_potential = kB*T*np.log(num_den_proton/2) 
    chemical_potential += 1.5*kB*T*np.log( 2*np.pi*hbar**2 / (m_electron*kB*T) )
    #print("Chemical Potential (no rest energy):", chemical_potential)
    
    # Computes the value if the bound state has L=0
    
    if L == 0 and K==0 and pert_index==False:
             
        Lu = L + 1
                            
        Ju = np.arange(np.abs(Lu-S), Lu+S+1, 1)  
                
        for ju in range(len(Ju)):
            
            if k==0:
                temp = (2*Ju[ju] + 1)*prefactor
            
            elif k==1:
                
                temp = (-1)**(Ju[ju]+F[f1]+0.5)
                temp *= (2*Ju[ju] + 1)*prefactor
                temp *= wigner_6j_explicit(J[j], J[j], 2, F[f1], F[f0], 0.5)
                temp *= wigner_6j_explicit(J[j], J[j], 2, 1, 1, Ju[ju])
                
            
            #hstep = np.abs(energy_array[1]-energy_array[0])
            #print(hstep)
            
            freq_bf = (energy_array- energy(N,L,J[j],I,F0))/h
            
            x = h*freq_bf/(kB*T)
            weight = 2*h*freq_bf**3/c**2
            
            phase_space = np.exp(-x)/(1-np.exp(-x))
            phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2
            
            integral = hstep*np.sum( A_Einstein_array1[:,N,L,j,ju]*phase_space*np.exp(-(energy_array-chemical_potential)/(kB*T))/np.sqrt(energy_array))
            
            temp *= integral
            
            term += temp
            
            if F[f0] < np.abs(J[j]-I) or F[f0] > J[j]+I:
                term = 0
    
    # Computes the bound state if L is not zero
        
    elif L > 0 and K==0 and pert_index==False:
        
        for lu in range(2):
            
            Lu = L - 1 + 2*lu
            Ju = np.arange(np.abs(Lu-S), Lu+S+1, 1)  

            for ju in range(len(Ju)): 
                

                if k==0:
                    temp = (2*Ju[ju] + 1)*prefactor
            
                elif k==1:
                
                    temp = (-1)**(Ju[ju]+F[f1]+0.5)
                    temp *= (2*Ju[ju] + 1)*prefactor
                    temp *= wigner_6j_explicit(J[j], J[j], 2, F[f1], F[f0], 0.5)
                    temp *= wigner_6j_explicit(J[j], J[j], 2, 1, 1, Ju[ju])
                    #print("temp:",temp)
                

            
                freq_bf = (energy_array- energy(N,L,J,I,F0))/h
            
                x = h*freq_bf/(kB*T)
                weight = 2*h*freq_bf**3/c**2
            
                phase_space = np.exp(-x)/(1-np.exp(-x))
                phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2
            
                           
                if L < Lu:
                    
                    integral = hstep*np.sum( A_Einstein_array1[:,N,L,j,ju]*phase_space*np.exp(-(energy_array-chemical_potential)/(kB*T))/np.sqrt(energy_array))

             
                    #print("Integral:",integral)
                    temp *= integral
                    term += temp

                                
                elif L > Lu:
                    
                                    
                    integral = hstep*np.sum( A_Einstein_array1[:,N,L,j,ju]*phase_space*np.exp(-(energy_array-chemical_potential)/(kB*T))/np.sqrt(energy_array))                     
                    
                    temp *= integral
                    term += temp
                    
                    '''
                    for i in range(1,numE-1):
                        
                        integral += hstep*A_Einstein_array2[i,N,L]*np.exp(-energy_array[i]/(kB*T))/np.sqrt(energy_array[i])
                
                        print(integral) 
                
                    temp *= integral 
            
                    term += temp   
                    '''
        
        # Making sure F is in the correct range
        
        if F[f0] < np.abs(J[j]-I) or F[f0] > J[j]+I:
            term = 0
        
    return term
   

'''

We want to theoretically define the energy of a Hydrogen atom that we will use as input
for other sections of this code. We want to incorporate corrections due to the Hyperfine 
structure of the atom along with other effects. 

Inputs: n, l, J, I, F
Output: Energy of a given leveln

'''




def energy(N, L, J, I, F):

    alpha_e = e0**2/(hbar*c)
    g = 5.56 
    m = 9.109e-31 # electron mass in kg
    M_p = 1.673e-27 # proton mass in kg


    # Unperturbed energy of the Hydrogen atom
    energy_0th = -ion_energy/ N**2 


    energy_1st = alpha_e**2*g*m/M_p
    energy_1st *= F*(F+1) - I*(I+1) - J*(J+1)
    energy_1st *= 1 / ( J*(J+1)*(2*L+1) )
    energy_1st *= 1/N**3
    energy_1st *= ion_energy
    
    energy_1st = 0 # The correction is only a correction to the 5th decimal of energy_0th

    #print(energy_1st)
    #print(energy_0th)

    # NOTE: Need to add perturbed terms to this quantity later.

    return energy_0th + energy_1st




    
'''

The function rad_field_tensor quantifies the temperature and polarization anisotropies, particularly
the monopole and quadrapole terms, as the source of changes in the density matrix due to stimulated
emission and absorbtion.

Input of rad_field_tensor:

    1) Rank K of the perturbation.

    2) Principal quantum number, n0 and n1, and angular momentum, l0 and l1, for two levels.

    3) Temperature of the photons.

    4) Strength of the perturbation (denoted by "rad_field").

Output:

    1) Perturbed phase space distribution (perturbed blackbody spectrum).

    2)Can be used to trace the evolution of the density matrix.


    THIS IS NOT COMPLETE AND IS MISSING THE POLARIZATION TERMS!!!!

'''

def rad_field_tensor(K, freq, T, Theta_0, Theta_2, pert_index):
    


    # Define a blackbody

    weight = 2*h*freq**3/c**2
    x = h*freq/(kB*T) # Ratio of photon energy with thermal energy

    if x > 0:
        phase_space = np.exp(-x)/(1-np.exp(-x))
        phase_deriv = -np.exp(-x)/(1-np.exp(-x))**2

    # The variable "pert_index" let's us know if we want the perturbed or unperturbed value.
    # If pert_index = False, then rad_field_tensor gives unperturbed value while pert_index = True
    # gives the 1st order term. 

    # Also, the value of K gives us which rad_field_Tensor we are interested in. Only the K=0,2 cases
    # are nonzero.

    if K==0 and freq > 0 and pert_index == False:
        rad_field = weight*phase_space

    elif K==0 and freq > 0 and pert_index == True:
        rad_field = weight*x*phase_deriv*Theta_0

    elif K==2 and freq>0 and pert_index == True:
        rad_field = (1/np.sqrt(2)) * weight*phase_space*Theta_2

    else:
        rad_field = 0

    return rad_field  

'''
def rad_field_tensor(K, n0, n1, l0, l1, J0, J1, T, pert_index):

    # Compute the frequency of the spectral line.

    freq = energy_noF(n0, l0, J0) - energy_noF(n1, l1, J1)
    freq = np.abs(freq)/h

    # Strength of perturbation
    
    Theta_0 = 10**-3
    Theta_2 = 10**-5

    # Define a blackbody

    weight = 2*h*freq**3/c**2
    x = h*freq/(kB*T) # Ratio of photon energy with thermal energy

    if x > 0:
        phase_space = np.exp(-x)/(1-np.exp(-x))
        phase_deriv = -np.exp(-x)/(1-np.exp(-x))**2

    # The variable "pert_index" let's us know if we want the perturbed or unperturbed value.
    # If pert_index = False, then rad_field_tensor gives unperturbed value while pert_index = True
    # gives the 1st order term. 

    # Also, the value of K gives us which rad_field_Tensor we are interested in. Only the K=0,2 cases
    # are nonzero.

    if K==0 and freq > 0 and pert_index == False:
        rad_field = weight*phase_space

    elif K==0 and freq > 0 and pert_index == True:
        rad_field = weight*x*phase_deriv*Theta_0

    elif K==2 and freq>0 and pert_index == True:
        rad_field = (1/np.sqrt(2)) * weight*phase_space*Theta_2

    else:
        rad_field = 0

    return rad_field
    
'''        

'''
def rad_field_tensor(K, n0, n1, l0, l1, freq, T):

    # Defines a blackbody

    weight = 2*h*freq**3/c**2
    x = h*freq / (kB*T) # Ratio that is invariant under the expansion of the Universe.

    phase_space = 1/ (np.exp(x)-1)

    phase_deriv = np.exp(x) / ( np.exp(x) - 1)**2

    # Depending on K, we have a different perturbed phase space density.

    if K == 0:
        pert_K0 = 0.001 # Perturbation variable
        rad_field = weight*phase_space + weight*x*phase_deriv*pert_K0
    elif K == 2:
        pert_K2 = 0.001 # Perturbation variable
        rad_field = (1/np.sqrt(2)) * weight * x * phase_deriv * pert_K2
    else:
        rad_field = 0

    return rad_field
'''

        
'''

Despite the massive amounts of dependent variables, the input and output is fairly simple.

Input: 

    1) A nonzero result requires both states have the same alpha = (n,l), I, and J. 

    2) Each quantum state we consider will need a valid value of K and two values of F.

'''


def Nhat(N0,N1,L0,L1,J0,J1,K0,K1,F0,F1,F2,F3):
    
    if N0==N1 and L0==L1 and J0==J1 and K0==K1 and F0==F2 and F1==F3:
        term = energy(N0,L0,J0,I,F0) - energy(N1,L1,J0,I,F1)
        term *= 1/h
        
    else:
        term = 0
        
    return term


'''

Note that all the T functions more-or-less behave the same way as far as input and output are concerned.

Input:

    1) Two sets of (n, l, J, K, F, F'). 

    2) Must obey the laws of physics such as J = L-1/2, L + 1/2.

    3) Frequency since we need to define a radiation field tensor.

Output:

    1) Tracks evolution and coherence between different states.

'''

    
def T_A_unpert_expected(n0,n1,l0,l1,J0,J1,K0,K1,Kr,F0,F1,F2,F3,pert_index):
    
    if energy(n0, l0, J0, I, F0) <= energy(n1, l1, J1, I, F1) and pert_index == False:

        term = 0

    elif energy(n0, l0, J0, I, F0) >= energy(n1, l1, J1, I, F1) and pert_index == False:
        

        n = n0
        l = l0
        J = J0
        K = K0
        F = F0
        F_prime = F1

        n_l = n1
        l_l = l1
        J_l = J1
        K_l = K1
        F_l = F2
        F_lprime = F3

        
        # Frequency of radiation (s^{-1})
        freq = (energy_noF(n, l, J) - energy_noF(n_l, l_l, J_l))/h
        freq = np.abs(freq)
        #print("freq:", freq)
                        
        # Calculating the appropriate Einstein coefficients

        B_Einstein = 32*np.pi**4/ (3*h**2*c)
        B_Einstein *= np.abs( dipole_element(n_l, n, l_l, l, J_l, J) )**2
        #print("B_abs:", B_Einstein)
  
        term = (2*J_l+1)*np.sqrt((2*F+1)*(2*F_l+1))*wigner_6j_explicit(J,J_l,1, F_l,F,0.5)**2
        term *= B_Einstein * (2*h*freq**3/c**2) * 1/(np.exp(h*freq/(kB*T))-1)
    else:
        term = 0
        
    return term
        
def T_A(n0, n1, l0, l1, J0, J1, K0, K1, Kr, F0, F1, F2, F3, pert_index):



    # We need to isolate which state is the lower state.

    if energy(n0, l0, J0, I, F0) <= energy(n1, l1, J1, I, F1):

        term1 = 0
        term2 = 0 

    else:
        

        n = n0
        l = l0
        J = J0
        K = K0
        F = F0
        F_prime = F1

        n_l = n1
        l_l = l1
        J_l = J1
        K_l = K1
        F_l = F2
        F_lprime = F3
                
        # Calculating the appropriate Einstein coefficients
        

        B_Einstein_abs = B_Einstein(n_l,n,l_l,l,J_l,J)
        
        #print("B_abs:", B_Einstein)

        term1 = (2*J_l + 1)*B_Einstein_abs # Prefactor to the sum
        
        # Frequency of radiation (s^{-1})
        freq = (energy_noF(n, l, J) - energy_noF(n_l, l_l, J_l))/h
        freq = np.abs(freq)
        #print("freq:", freq)

        term2 = 0 # Value of the sum across K_r 
    
        # Computing the sum across K_r from 0 to 2 where Q_r=0 is fixed.

        temp = np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_l+1)*(2*F_lprime+1)*(2*K+1)*(2*K_l+1)*(2*Kr+1))
        temp *= (-1)**(F_lprime - F_l)
        
        temp *= wigner_9j_explicit(F,F_l,1,F_prime, F_lprime,1,K,K_l,0)*wigner_6j_explicit(J,J_l,1,F_l,F,0.5)
        temp *= wigner_6j_explicit(J,J_l,1,F_lprime,F_prime,0.5)*wigner_6j_explicit(K,K_l,0,0,0,0)
        temp *= rad_field_tensor(Kr,freq,T,Theta_0, Theta_2, pert_index)

            
        #temp *= rad_field_tensor(Kr, n0, n1, l0, l1, J0, J1, T, pert_index)

        term2 += temp
        #print("term2:", term2)



    return term1*term2




def T_S_unpert_expected(n0, n1, l0, l1, J0, J1, K0, K1, Kr, F0, F1, F2, F3, pert_index):


 # Prefactor to the sum
    term2 = 0 # Value of the sum across K_r 


    # We need to isolate which state is the upper state.

    if energy(n1, l1, J1, I, F1) <= energy(n0, l0, J0, I, F0) and pert_index == False:
    
        term = 0

    elif energy(n1, l1, J1, I, F1) > energy(n0, l0, J0, I, F0) and pert_index == False:
        
        n = n0
        l = l0
        J = J0
        K = K0
        F = F0
        F_prime = F1

        n_u = n1
        l_u = l1
        J_u = J1
        K_u = K1
        F_u = F2
        F_uprime = F3

        # Calculating the appropriate Einstein coefficients
        
        B_Einstein = 32*np.pi**4 / (3*h**2*c)
        B_Einstein *= np.abs( dipole_element(n_u, n, l_u, l, J_u, J) )**2
        #print("B_stim:", B_Einstein)
        
        # Frequency of radiation (s^{-1})
        freq = (energy_noF(n_u, l_u, J_u) - energy_noF(n, l, J))/h
        freq = np.abs(freq)
        #print("freq:", freq)

                        
        # Calculating the appropriate Einstein coefficients

        B_Einstein = 32*np.pi**4/ (3*h**2*c)
        B_Einstein *= np.abs( dipole_element(n_u, n, l_u, l, J_u, J) )**2
        #print("B_abs:", B_Einstein)
  
        term = (2*J_u+1)*np.sqrt((2*F+1)*(2*F_u+1))*wigner_6j_explicit(J,J_u,1, F_u,F,0.5)**2
        term *= B_Einstein * (2*h*freq**3/c**2) * 1/(np.exp(h*freq/(kB*T))-1)
    else:
        term = 0

    

    return term
    
def T_S(n0, n1, l0, l1, J0, J1, K0, K1, Kr, F0, F1, F2, F3, pert_index):


 # Prefactor to the sum
    term2 = 0 # Value of the sum across K_r 


    # We need to isolate which state is the upper state.

    if energy(n1, l1, J1, I, F1) <= energy(n0, l0, J0, I, F0):
    
        term1 = 0
        term2 = 0

    elif energy(n1, l1, J1, I, F1) > energy(n0, l0, J0, I, F0):
        
        n = n0
        l = l0
        J = J0
        K = K0
        F = F0
        F_prime = F1

        n_u = n1
        l_u = l1
        J_u = J1
        K_u = K1
        F_u = F2
        F_uprime = F3

        # Calculating the appropriate Einstein coefficients

        B_Einstein_stim = B_Einstein(n_u,n,l_u,l,J_u,J)
        
        term1 = (2*J_u + 1)*B_Einstein_stim
        
        #print("B_stim:", B_Einstein)
        
        # Frequency of radiation (s^{-1})
        freq = (energy_noF(n_u, l_u, J_u) - energy_noF(n, l, J))/h
        freq = np.abs(freq)
        #print("freq:", freq)


        #print("This is the B Einstein coefficient:", B_Einstein)

        # Computing the sum across K_r from 0 to 2 where Q_r=0 is fixed.


        temp = np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_u+1)*(2*F_uprime+1)*(2*K+1)*(2*K_u+1)*(2*Kr+1))
        temp *= (-1)**(Kr+K_u+F_uprime-F_u)
        
        temp *= wigner_9j_explicit(F,F_u,1,F_prime,F_uprime,1,K,K_u,0)
        temp *= wigner_6j_explicit(J_u,J,1,F,F_u,0.5)*wigner_6j_explicit(J_u,J,1,F_prime,F_uprime,0.5)
        temp *= wigner_3j_explicit(K,K_u,0,0,0,0)
        temp *= rad_field_tensor(Kr,freq,T,Theta_0,Theta_2,pert_index)
        
        term2 += temp
        #print("term2:", temp)

    

    return term1*term2

def T_E_unpert_expected(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3):

    # Compute the frequency of the transition.

    freq = energy_noF(n0, l0, J0) - energy_noF(n1, l1, J1)
    freq = np.abs(freq)/h
    #print("freq:", freq)

    # We need to determine which state is the upper state.

    if energy(n1, l1, J1, I, F1) <= energy(n0, l0, J0, I, F0):
        
        term = 0

    elif energy(n1, l1, J1, I, F1) > energy(n0, l0, J0, I, F0) and K0 != 2 and K1 != 2:

        
        n = n0
        l = l0
        J = J0
        K = K0
        F = F0
        F_prime = F1

        n_u = n1
        l_u = l1
        J_u = J1
        K_u = K1
        F_u = F2
        F_uprime = F3

        # Calculating the appropriate Einstein Coefficients

        A_Einstein = 64*np.pi**4 * freq**3 / (3*h*c**3)
        A_Einstein *= np.abs( dipole_element(n_u, n, l_u, l, J_u, J) )**2
        #print("A Einstein coefficient:", A_Einstein)

        # Computing the term itself
        term = (2*J_u+1)*np.sqrt((2*F+1)*(2*F_u+1))*wigner_6j_explicit(J,J_u,1,F_u,F,0.5)**2*A_Einstein
    else:
        term = 0

    return term
    
    
def T_E(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3):

    # Compute the frequency of the transition.

    freq = energy_noF(n0, l0, J0) - energy_noF(n1, l1, J1)
    freq = np.abs(freq)/h
    #print("freq:", freq)

    # We need to determine which state is the upper state.

    if energy(n1, l1, J1, I, F1) <= energy(n0, l0, J0, I, F0):
        
        term = 0

    if energy(n1, l1, J1, I, F1) > energy(n0, l0, J0, I, F0):

        
        n = n0
        l = l0
        J = J0
        K = K0
        F = F0
        F_prime = F1

        n_u = n1
        l_u = l1
        J_u = J1
        K_u = K1
        F_u = F2
        F_uprime = F3

        # Calculating the appropriate Einstein Coefficients
        
        A_Einstein = A_Einstein_coeff(n_u,n,l_u,l,J_u,J)
        
        #print("A Einstein coefficient:", A_Einstein)

        # Computing the term itself
        term = 0

        if K == K_u:
        
            term = (2*J_u + 1)*A_Einstein
            term *= np.sqrt( (2*F+1)*(2*F_prime+1)*(2*F_u+1)*(2*F_uprime+1) )
            term *= (-1)**(1 + F_prime + F_uprime)
            term *= wigner_6j_explicit(F , F_prime, K, F_uprime, F_u, 1) 
            term *= wigner_6j_explicit(J_u , J, 1, F, F_u, 0.5) 
            term *= wigner_6j_explicit(J_u, J, 1, F_prime, F_uprime, 0.5) 
            #print("term:", term)
    


    return term


'''

Same general idea as what was done with the T functions except the inputs are different.
Will discuss this detail at a later date with my code.

'''

def R_A(n, l, J, I, K, K_prime, Kr, F0, F1, F2, F3, pert_index):


    Nmax = numN # Total number of different values of n we are considering.

    # Define 3 terms to make the calculation easier

    term1 = 0
    term2 = 0
    term3 = 0
    total_term = 0

    for n_level in range(1,Nmax+1):
        for l_level in range(n_level):

            # Composes allowed values of J given a value of L and S.
            # Allowed values: J = L- 1/2, L+ 1/2

            J_level = np.arange( np.abs(l_level - S), l_level + S+1, 1)
            
            for j_index in range(len(J_level)):

                J_u = J_level[j_index] # J_u value

                # Need to determine if a state is more energetic.

                if n_level > n and n>1:
                    if l_level == l-1 or l_level == l or l_level == l+1:

                        #print("Atomic levels (n,l,J_u)")
                        #print(n_level,l_level,J_u)
                        
                        # Frequency of radiation (s^{-1})
                        freq = (energy_noF(n_level, l_level, J_u) - energy_noF(n, l, J))/h
                        freq = np.abs(freq)

                        
                        B_Einstein_abs = B_Einstein(n,n_level,l,l_level,J,J_u)
                        
                        #print("B_abs:",B_Einstein)

                        
                        term1 = B_Einstein_abs
                        term1 *= np.sqrt(3*(2*K+1)*(2*K_prime+1)*(2*Kr+1))
                        term1 *= (-1)**(1+J_u-I+F0)
                        #term1 *= float( wigner_6j(J, J, Kr, 1, 1, J_u) )
                        #term1 *= float( wigner_3j(K, K_prime, Kr, 0,0, 0) )
                        term1 *= (2*J+1)*wigner_3j_explicit(K,K_prime,Kr,0,0,0)*wigner_6j_explicit(J,J,0,1,1,J_u)

                        #print("This is term1")
                        #print(term1)
                        #print("n_level:", n_level)
                        #print("l_level:", l_level)
                        #print("J_u:", J_u)
                        #print("term1", term1)
    
                        if F0 == F2:
                            term2 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1))*wigner_6j_explicit(J,J,0,F3,F1,0.5)*wigner_6j_explicit(K,K_prime,0,F3,F1,F0)

                            #term2 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_u, T, pert_index)
                            term2 *= rad_field_tensor(Kr, freq, T, Theta_0, Theta_2, pert_index)
                            #print("This is term2")
                            #print("term2:",term2)
                        if F1 == F3:
                            term3 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
                            term3 *= (-1)**(F2 - F1)
                            term3 *= wigner_6j_explicit(J,J,0,F2,F0,0.5)*wigner_6j_explicit(K,K_prime,0,F2,F0,F1)

                            #term3 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_u, T, pert_index)
                            term3 *= rad_field_tensor(Kr, freq, T, Theta_0, Theta_2, pert_index)
                            #print("This is term3")
                            #print("term3",term3)

                        total_term += term1*(term2+term3)
                        
                      
                        
                        #print("total_term:", total_term)
                        
                        

    return total_term



    

def R_E(n, l, J, I, K, K_prime, F0, F1, F2, F3):

    Nmax = numN # Total number of principal quantum states being considered.

    A_sum = 0 # Sum of Einstein A coefficients.

    if K == K_prime and F0 == F2 and F1 == F3:

        for n_level in range(1,Nmax+1):
            for l_level in range(n_level):


                # Composes allowed values of J given a value of L and S.
                # Allowed values: J = L- 1/2, L+ 1/2

                J_level = np.arange(np.abs(l_level - S), l_level + S + 1, 1)

                for j_index in range(len(J_level)):

                    J_l = J_level[j_index] # J for the lower state.

                    # Need to determine if a state is less energetic.

                    if n_level < n:
                        if l_level == l-1 or l_level == l or l_level == l+1:

                            # Calculate frequency

                            freq = energy_noF(n, l, J) - energy_noF(n_level, l_level, J_l)
                            freq = np.abs(freq)/h

                            # Compute Einstein A coefficient

                            #print("Atomic levels (n,l,J_l)")
                            #print(n_level,l_level,J_l)
                            
                            if (n>2 and n_level>1) or (n==2 and n_level==1):

                                A_Einstein = A_Einstein_coeff(n,n_level,l,l_level,J,J_l)

                                #print("A Einstein coefficient:", A_Einstein)

                                A_sum += A_Einstein # Sum each allowed Einstein-A coefficient.

    return A_sum



def R_S(n, l, J, I, K, K_prime, Kr, F0, F1, F2, F3, pert_index):

    Nmax = numN+1 # Total number of principal quantum states being considered.

    # Define 3 terms to make the calculations easier.

    term1 = 0
    term2 = 0
    term3 = 0
    total_term = 0

    for n_level in range(1,Nmax+1):
        for l_level in range(n_level):

            # Composes allowed values of J given a value of L and S.
            # Allowed values: J = L- 1/2, L+ 1/2

            J_level = np.arange(np.abs(l_level - S), l_level + S + 1, 1)

            for j_index in range(len(J_level)):

                J_l = J_level[j_index] # J value for the lower level.


                # Need to determine if a state is less energetic.

                if n_level < n and n != 2:
                    if l_level == l-1 or l_level == l or l_level == l+1:

                        #print("Atomic levels (n,l,J_l)")
                        #print(n_level,l_level,J_l)
                        
                        if n>2 and n_level>1:


                            # Calculate frequency

                            freq = energy_noF(n, l, J) - energy_noF(n_level, l_level, J_l)
                            freq = np.abs(freq) / h

                            # Calculating allowed Einstein-B coefficients

                            
                            B_Einstein_stim = B_Einstein(n,n_level,l,l_level,J,J_l)
                            
                            #print("B_stim:",B_Einstein)
                        
                                                
                            term1 = B_Einstein_stim
                            term1 *= np.sqrt( 3*(2*K+1)*(2*K_prime+1)*(2*Kr+1) )
                            term1 *= (-1)**(1+J_l - I + F0 +Kr)
                            term1 *= wigner_6j_explicit(J, J, Kr, 1, 1, J_l)*wigner_3j_explicit(K,K_prime,Kr,0,0,0)
                            term1 *= (2*J+1) #Prefactor
                            #print("This is term1")
                            #print(term1)
                            #print("n_level:", n_level)
                            #print("l_level:", l_level)
                            #print("J_l:", J_l)
                            #print("F2:", F2)
                            #print("F3:", F3)
                            #print("term1", term1)
        
                            if F0 == F2:
                                #print("we are in term2")
                                term2 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
                                term2 *= wigner_6j_explicit(J,J,0,F3,F1,0.5)*wigner_6j_explicit(K,K_prime,0,F3,F1,F0)
                                term2 *= rad_field_tensor(Kr, freq, T, Theta_0, Theta_2,pert_index)
                                
                                #print("This is term2")
                                #print("term2:",term2)    
                            if F1 == F3:
                                #print("we are in term3")
                                term3 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
                                term3 *= (-1)**(F2 - F1 + K + K_prime + Kr)
                                term3 *= wigner_6j_explicit(J,J,0,F2,F0,0.5)*wigner_6j_explicit(K,K_prime,0,F2,F0,F1)
                                term3 *= rad_field_tensor(Kr, freq, T, Theta_0, Theta_2, pert_index)
                                #print("This is term3")
                                #print("term3",term3)
                        
                            total_term += term1*(term2+term3) # Summing each term.
                            #print("total_term:", total_term)

    
    return total_term


# We want to make a mask for correct values.

def mask_allowed(N_index, l_index, j_index, k_index, f0_index, f1_index):
    

    '''
    if N_index > l_index and N_index > 1:
        if l_index > 0:
            physical_val = True
        elif l_index == 0 and j_index == 0:
            physical_val = True
        else:
            physical_val = False
    elif l_index > N_index-1:
        physical_val = False
        
        
        
    if k_index == 0 and f0_index != f1_index:
        physical_val = False
    elif k_index == 1 and l_index == 0  and f0_index + f1_index < 2:
        physical_val = False
    elif k_index == 1 and l_index == 1 and j_index == 0  and f0_index + f1_index < 2:
        physical_val = False
    else:
        physical_val = True

        
    if N_index == 1 and l_index == 0 and k_index == 0 and f0_index == f1_index:
        physical_val = True
    elif N_index == 1 and l_index == 0 and k_index == 0 and f0_index + f1_index == 2:
        physical_val = True
    elif N_index == 1:
        physical_val = False

    
    return physical_val 
    '''
    

    physical_val = False
    
    if N_index > l_index and N_index > 1 and k_index == 0 and f0_index == f1_index:
        if l_index > 0:
            physical_val = True
        elif l_index == 0 and j_index == 0:
            physical_val = True

    elif N_index > l_index and N_index > 1 and k_index == 1:
        if l_index == 0 and j_index == 0 and f0_index + f1_index == 2:
             physical_val = True
        elif l_index == 1 and j_index == 0 and f0_index + f1_index == 2:
            physical_val = True
        elif l_index == 1 and j_index == 1:
            physical_val = True
        elif l_index > 1:
            physical_val = True

    elif N_index == 1 and l_index == 0 and j_index == 0 and k_index == 0 and f0_index == f1_index:
        physical_val = True
    elif N_index == 1 and l_index == 0 and j_index == 0 and k_index == 1 and f0_index + f1_index == 2:
        physical_val = True



    
    return physical_val 

def Lambda_2photon(N0, N1, L0, L1, J0, J1, K0, K1, Kr, F0, F1, F2, F3, T, delta_g, pert_index):


    
    x = h*freq_lya/(kB*T)
    weight = 2*h*freq_lya**3/c**2
        
    if x > 0:
        phase_space = np.exp(-x)/(1-np.exp(-x))
        phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2
    
    
    
    
    term = 0
    
    if K0==K1 and Kr==0 and F0==F1 and F2==F3 and pert_index==False:
        if (N0==1 and L0==0 and J0==0.5 and F0==F2 and F1==F3) and (N1==1 and L1==0 and J1==0.5 and F0==F2 and F1==F3):
            term = -A_2photon*phase_space
            #term = 0
        elif (N0==1 and L0==0 and J0==0.5) and (N1==2 and L1==0 and J1==0.5):
            term = A_2photon*2*np.sqrt((2*F0+1)*(2*F2+1))*wigner_6j_explicit(0.5,0.5,1,F2,F0,0.5)**2
        elif (N0==2 and L0==0 and J0==0.5) and (N1==1 and L1==0 and J1==0.5):
            term = A_2photon*phase_space*2*np.sqrt((2*F0+1)*(2*F2+1))*wigner_6j_explicit(0.5,0.5,1,F0, F2, 0.5)**2
            #term=0
        elif (N0==2 and L0==0 and J0==0.5 and F0==F2 and F1==F3) and (N1==2 and L1==0 and J1==0.5 and F0==F2 and F1==F3):
            term = - A_2photon
        else:
            term = 0
    elif K0==K1 and Kr==0 and F0==F1 and F2==F3 and pert_index==True:
        if (N0==1 and L0==0 and J0==0.5 and F0==F2 and F1==F3) and (N1==1 and L1==0 and J1==0.5 and F0==F2 and F1==F3):
            term = -A_2photon*x*phase_deriv*delta_g
        elif (N0==2 and L0==0 and J0==0.5) and (N1==1 and L1==0 and J1==0.5):
            term = A_2photon*phase_deriv*delta_g
            term *= 2*np.sqrt((2*F0+1)*(2*F2+1))*wigner_6j_explicit(0.5,0.5,1,F0, F2, 0.5)**2
        else:
            term = 0
    else:
        term = 0
        
        
        
    
    '''
    term = 0
    if K0==K1 and Kr==0 and F0==F2 and F1==F3 and F0==F1 and pert_index==False:
        if (N0==1 and L0==0 and J0==0.5) and (N1==1 and L1==0 and J1==0.5) :
            term = -A_2photon*np.exp(-h*freq_lya/(kB*T))
        elif (N0==2 and L0==0 and J0==0.5) and (N1==1 and L1==0 and J1==0.5):
            term = A_2photon*np.exp(-h*freq_lya/(kB*T))             
        elif (N0==2 and L0==0 and J0==0.5)  and (N1==2 and L1==0 and J1==0.5):
            term = - A_2photon
        elif (N0==1 and L0==0 and J0==0.5) and (N1==2 and L1==0 and J1==0.5):
            term = A_2photon  
        else:
            term = 0
     '''
            
        
    return term

    
'''
Arrays for the unperturbed evolution matrix (Lambda0), the perturbed evolution matrix corresponding to K=0 (L0),
and the perturbed evolution matrix corresponding to K=2 (L2).

The structure of each matrix is the following:

Matrix(N0, L0, J0, K0, F0, F1, N1, L1, J1, K1, F2, F3)

'''


Lambda0 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN+1, numL, numJ, numK, numF, numF), dtype = complex)

L0 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN+1, numL, numJ, numK, numF, numF), dtype = complex)

L2 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN+1, numL, numJ, numK, numF, numF), dtype = complex)

mask_array = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN+1, numL, numJ, numK, numF, numF), dtype = complex)






'''
We want to define the matrix structure for the density matrix and the source function so that we can compute matrix
multiplication.
'''

density_matrix = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = complex)
source_matrix = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = complex)





# Einstein arrays for both cases

A_Einstein_array1 =  np.zeros( (numE, numN+1, numN, numJ, numJ) )
A_Einstein_array2 =  np.zeros( (numE, numN+1, numN, numJ, numJ) )

B_Einstein_array1 =  np.zeros( (numE, numN+1, numN, numJ, numJ) )
B_Einstein_array2 =  np.zeros( (numE, numN+1, numN, numJ, numJ) )

# Array with energy values
energy_array = np.zeros(numE)

# Acquiring energy and A coefficients

hstep = Emax/numE # Step size

# Setting up midpoint method for integration

energy_array = np.linspace(hstep/2, (numE - 1/2)*hstep, numE)


for i in range(numE):
    
       
    bf_dipole_array1 = dipole_bf(numN, energy_array[i], True) # Bound state l < Free state l
    bf_dipole_array2 = dipole_bf(numN, energy_array[i], False) # Bound state l > Free state l

    for N in range(1,len(bf_dipole_array1),1):
        for L in range(N):
            
                L_u = L+1
                
                J = np.arange( np.abs(L-S), L+S+1, 1)
                J_u = np.arange(np.abs(L_u-S), L_u+S+1, 1)
                
                for j in range(len(J)):
                    for j_u in range(len(J_u)):
                        
                        # Angular prefactor as discussed in the Overleaf
                        
                        ang_prefactor = (-1)**(L_u + S + J[j] + 1)
                        ang_prefactor *= np.sqrt( (2*J[j]+1)*(2*L_u+1) )
                        #ang_prefactor *= float(wigner_6j(L_u,L,1,J[j],J_u[j_u],S))
                        ang_prefactor *= wigner_6j_explicit(L_u,L,1,J[j],J_u[j_u],0.5)

                        ang_prefactor *= np.sqrt(L_u / (2*L_u+1))
                                                
    
                        freq = (energy_array[i] - energy_noF(N, L, J[j]))/h #Hydrogen energy is negative, so total is positive.

        
                        A_Einstein_array1[i,N,L,j,j_u] = 64*np.pi**4/(3*h*c**3) * freq**3 * ang_prefactor**2 * bf_dipole_array1[N,L]**2
                        B_Einstein_array1[i,N,L,j,j_u] = 32*np.pi**4/(3*h**2*c) * ang_prefactor**2 * bf_dipole_array1[N,L]**2


    for N in range(1,len(bf_dipole_array2),1):
        for L in range(1,N,1):
            
                L_u = L-1
                
                J = np.arange( np.abs(L-S), L+S+1, 1)
                J_u = np.arange(np.abs(L_u-S), L_u+S+1, 1)
                
                for j in range(len(J)):
                    for j_u in range(len(J_u)):
                        
                        # Angular prefactor as discussed in the Overleaf
                        
                        ang_prefactor = (-1)**(L_u + S + J[j] + 1)
                        ang_prefactor *= np.sqrt( (2*J[j]+1)*(2*L_u+1) )
                        #ang_prefactor *= float(wigner_6j(L_u,L,1,J[j],J_u[j_u],S))
                        ang_prefactor *= wigner_6j_explicit(L_u,L,1,J[j],J_u[j_u],0.5 )
                        
                        ang_prefactor *= np.sqrt(L / (2*L_u+1))
                                                
    
                        freq = (energy_array[i] -energy_noF(N, L, J[j])) /h #Hydrogen energy is negative, so total is positive.

        
                        A_Einstein_array2[i,N,L,j,j_u] = 64*np.pi**4/(3*h*c**3) * freq**3 * ang_prefactor**2 * bf_dipole_array2[N,L]**2
                        B_Einstein_array2[i,N,L,j,j_u] = 32*np.pi**4/(3*h**2*c) * ang_prefactor**2 * bf_dipole_array2[N,L]**2


        
output_file = open("output.txt", "w")
        
for N0 in range(1, numN+1):
    for N1 in range(1, numN+1):
        for l0 in range(N0):
            for l1 in range(N1):
                J0 = np.arange(np.abs(l0-S),l0+S+1,1 )
                J1 = np.arange(np.abs(l1-S), l1+S+1,1)
                        
                for j0 in range(len(J0)):
                    for j1 in range(len(J1)):
                        F0 = np.arange(np.abs(J0[j0]-I), J0[j0]+I+1,1)
                        F1 = F0
                        
                        F2 = np.arange(np.abs(J1[j1]-I), J1[j1]+I+1, 1)
                        F3 = F2
                        
                        for k0 in range(numK):
                            
                            K0 = 2*k0
                            
                            for k1 in range(numK):
                                
                                K1 = 2*k1
                                
                                for kr in range(numK):
                                    
                                    Kr = 2*kr
                                    
                                    for f0 in range(len(F0)):
                                        for f1 in range(len(F1)):
                                            for f2 in range(len(F2)):
                                                for f3 in range(len(F3)):
                                                    
                                                    # Assigning true and false values to the masked array
                                                    
                                                    if mask_allowed(N0, l0, j0, k0, f0, f1) == True and mask_allowed(N1, l1, j1, k1, f2, f3) == True:
                                                        mask_array[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] = True
                                                    else:
                                                        mask_array[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] = False
                                                        

                                                    #print("####### New State #####", file = output_file)
                                                    #print(" ", file = output_file)
                                                    #print("N0:"+str(N0), file = output_file)
                                                    #print("N1:"+str(N1), file = output_file)
                                                    #print("L0:"+str(l0), file = output_file)
                                                    #print("L1:"+str(l1), file = output_file)
                                                    #print("J0:"+str(J0[j0]), file = output_file)
                                                    #print("J1:"+str(J1[j1]), file = output_file)
                                                    #print("K0:"+str(K0), file = output_file)
                                                    #print("K1:"+str(K1), file = output_file)
                                                    #print("Kr:"+str(Kr), file = output_file)
                                                    #print("F0:"+str(F0[f0]), file = output_file)
                                                    #print("F1:"+str(F1[f1]), file = output_file)
                                                    #print("F2:"+str(F2[f2]), file = output_file)
                                                    #print("F3:"+str(F3[f3]), file = output_file)
                                                    #print(" ", file = output_file)
                                                    #print("##### Code Index #####", file = output_file)
                                                    #print(" ", file = output_file)
                                                    #print("N0:"+str(N0), file = output_file)
                                                    #print("N1:"+str(N1), file = output_file)
                                                    #print("L0:"+str(l0), file = output_file)
                                                    #print("L1:"+str(l1), file = output_file)
                                                    #print("j0:"+str(j0), file = output_file)
                                                    #print("j1:"+str(j1), file = output_file)
                                                    #print("k0:"+str(k0), file = output_file)
                                                    #print("k1:"+str(k1), file = output_file)
                                                    #print("kr:"+str(kr), file = output_file)
                                                    #print("f0:"+str(f0), file = output_file)
                                                    #print("f1:"+str(f1), file = output_file)
                                                    #print("f2:"+str(f2), file = output_file)
                                                    #print("f3:"+str(f3), file = output_file)
                                                    #print(" ", file = output_file)
                                                    
                                                    #print("Mapping to 2D fits file", file = output_file)
                                                    
                                                    # Mapping from 12D array to 2D array. Where are the elements?
                                                    
                                                    if N1>1:
                                                        element_x = 16*numN*(N1-2) + 16*l1 + 8*j1 + 4*k1 + 2*f2 + f3 + 1
                                                        #print("Nx (excited): "+str(element_x), file = output_file)                                                    
                                                    else:
                                                        element_x = 8*j1 + 4*k1 + 2*f2 + f3 + 1
                                                        #print("Nx (1s): "+str(element_x), file = output_file)
                                                    
                                                    if N0>1:
                                                        element_y = 16*numN*(N0-2) + 16*l0 + 8*j0 + 4*k0 + 2*f0 + f1 + 1
                                                        #print("Ny (excited): "+str(element_y), file = output_file)                                                    
                                                    else:
                                                        element_y = 8*j0 + 4*k0 + 2*f0 + f1 + 1
                                                        #print("Ny (1s): "+str(element_y), file = output_file)                                                        
                                                    
                                                    


                                                    
                                        
                                                
                                                    

                                                    if N0==N1 and l0==l1:
                                                        
                                                        #print(" ", file=output_file)
                                                        #print("R terms (and N hat)", file=output_file)
                                                        #Nhat_total = 0
                                                        
                                                        # Computing the Nhat term
                                                        
                                                        Nhat_total = Nhat(N0,N1,l0, l1, J0[j0], J1[j1], K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])
                                                        Nhat_total *= -2*np.pi*complex(0,1)
                                                        
                                                        #print("Nhat: "+str(Nhat_total), file = output_file)
                                                        
                                                        #Nhat_total = 0
                                                        RA_unpert = 0
                                                        RS_unpert = 0
                                                        RE_total = 0
                                                        RA_pert_0 = 0
                                                        RS_pert_0 = 0
                                                        RA_pert_2 = 0
                                                        RS_pert_2 = 0
                                                                                                                
                                                        Lymann_BB_unpert = 0
                                                        Lymann_BB_pert_0 = 0
                                                        Lymann_BB_pert_2 = 0   
                                                                                                             
                                                        Lymann_source_unpert = 0
                                                        Lymann_source_pert_0 = 0
                                                        Lymann_source_pert_2 = 0   
                                                        
                                                        photo_unpert = 0
                                                        photo_pert0 = 0
                                                        
                                                        two_photon_unpert = 0
                                                        two_photon_pert_0 = 0
                                                                                                                                                                       
                                                        
                                                        
                                                        #if N0>2 and Kr==0 and J0[j0]==J1[j1]:
                                                        if Kr==0:
                                                           
                                                            if (N0  == 2 and l0==1): 

                                                                if J0[j0]==J1[j1]:

                                                                    RE_total = R_E(N0, l0, J0[j0], I, K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])
                                                                    RA_unpert = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)

                                                                Lymann_BB_unpert = Lambda_lymann_lines_BB(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,False)
                                                                Lymann_BB_pert_0 = Lambda_lymann_lines_BB(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,True)

                                                                Lymann_source_unpert = Lambda_lymann_lines_source(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,X_e,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,False)
                                                                Lymann_source_pert_0 = Lambda_lymann_lines_source(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,X_e,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,True)
                                                                                                                                    
                                                            elif (N0==1 and l0==0):
                                                            
                                                                if J0[j0]==J1[j1]:

                                                                    RE_total = R_E(N0, l0, J0[j0], I, K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])                                                                    
                                                                                                                      
   
                                                                Lymann_BB_unpert = Lambda_lymann_lines_BB(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,False)
                                                                Lymann_BB_pert_0 = Lambda_lymann_lines_BB(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,True)

                                                                Lymann_source_unpert = Lambda_lymann_lines_source(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,X_e,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,False)
                                                                Lymann_source_pert_0 = Lambda_lymann_lines_source(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,X_e,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,True)
                                                                two_photon_unpert = Lambda_2photon(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1], F2[f2], F3[f3], T,delta_g, False)                                                            
                                                                two_photon_pert_0 = Lambda_2photon(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1], F2[f2], F3[f3], T,delta_g, True)                                                            
                                                            else:
                                                            
                                                                if J0[j0] == J1[j1]:
                                                                                                            
                                                                    RA_unpert = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)
                                                                    RS_unpert = R_S(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)
                                                                    RA_pert_0 = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                                    RS_pert_0 = R_S(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                                    RE_total = R_E(N0, l0, J0[j0], I, K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])

                                                                two_photon_unpert = Lambda_2photon(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1], F2[f2], F3[f3], T,delta_g, False) 
                                                                two_photon_pert_0 = Lambda_2photon(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1], F2[f2], F3[f3], T, delta_g, True) 
                                                            if J0[j0]==J1[j1]:
                                                                photo_unpert = boundfree_photoionization(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], X_e, False)
                                                                photo_pert_0 = boundfree_photoionization(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], X_e,True)
                                                                                                                            
                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += -RE_total + two_photon_unpert
                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += -RS_unpert                                                   
                                                        

                                                        Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - RA_unpert - RS_unpert - RE_total + two_photon_unpert
                                                        #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += -RA_pert_0 - RS_pert_0
                                                        

                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - RA_unpert - RS_unpert - RE_total 
                                                        #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += -RA_pert_0 - RS_pert_0 - photo_pert_0 

                                                        Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_unpert + Lymann_source_unpert
                                                        L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_pert_0 + Lymann_source_pert_0 + two_photon_pert_0

                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] +=  Lymann_BB_unpert
                                                        #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - Lymann_BB_pert_0
                                                            
                                                                                                                        
                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_source_unpert
                                                        #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_source_pert_0
                                                            
                                                        if N0 != 1 and N1 !=1 and J0[j0]==J1[j1]:
                                                            Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - photo_unpert
                                                            L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - photo_pert_0
                                                            #pass
                                                        
                                                            
                                                        #if N0 > 2 and Kr==2:
                                                        if Kr==2:    
                                                            if (N0  != 2 and l0!=1 and J0[j0]==J1[j1]) and (N0!=1 and l0!=0 and J0[j0]==J1[j1]):                                                         
                                                                RA_pert_2 = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                                RS_pert_2 = R_S(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)

                                                            Lymann_BB_Kr_2 = Lambda_lymann_lines_BB(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,True)  
                                                            Lymann_source_Kr_2 = Lambda_lymann_lines_source(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,X_e,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,True) 
                                                             
                                                            photo_pert_2 = boundfree_photoionization(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], X_e, True)
                                             
                                                            #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - RA_pert_2 - RS_pert_2 - photo_pert_2
                                                            L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - photo_pert_2
                                                            
                                                            if K0 == 2 and K1 == 0:
                                                                L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_Kr_2 + Lymann_source_Kr_2 
                                                                #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_Kr_2                                                           
                                                                #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_source_Kr_2                                                                                                                        
                                                                                                             

                                                    elif N0 != N1:
                                                        
                                                            #print(" ", file=output_file)
                                                            #print("T terms", file=output_file)
                                                            
                                                            TA_unpert = 0
                                                            TS_unpert = 0
                                                            TE_total = 0

                                                            TA_unpert_expected = 0
                                                            TS_unpert_expected = 0
                                                            TE_total_expected = 0
                                                                                                                        
                                                            TA_pert_0 = 0
                                                            TS_pert_0 = 0
                                                            
                                                            TA_pert_2 = 0
                                                            TS_pert_2 = 0
                                                            
                                                            Lymann_BB_unpert = 0
                                                            Lymann_BB_pert_0 = 0
                                                            Lymann_BB_pert_2 = 0
                                                            
                                                            Lymann_source_unpert = 0
                                                            Lymann_source_pert_0 = 0
                                                            Lymann_source_pert_2 = 0
                                                            
                                                                                                                                                                                   
                                                            #if N0 != 1 and N0!= 2 and N1 != 1 and N1 != 2 and Kr == 0:
                                                            if Kr==0:
                                                                if (N0  == 2 and l0==1 and N1 == 1 and l1 == 0) or (N0==1 and l0==0 and N1==2 and l1==1): 

                                                                    TE_total = T_E(N0, N1, l0, l1 , J0[j0], J1[j1], K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])
                                                                    
                                                              
                                                                    Lymann_BB_unpert = Lambda_lymann_lines_BB(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,False)
                                                                    Lymann_BB_pert_0 = Lambda_lymann_lines_BB(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,True)
                                                               
                                                                    Lymann_source_unpert = Lambda_lymann_lines_source(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,X_e,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,False)
                                                                    Lymann_source_pert_0 = Lambda_lymann_lines_source(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,X_e,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,True) 
                                                                    
                                                                elif (N0==2 and l0==0 and N1==1 and l1==0) or (N0==1 and l0==0 and N1==2 and l1==0):
                                                                    two_photon_unpert = Lambda_2photon(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1], F2[f2], F3[f3], T, delta_g,False)
                                                                    two_photon_pert_0 = Lambda_2photon(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1], F2[f2], F3[f3], T, delta_g, True)                                                                   
                                                                else:
                                                                                                                                                                
                                                                    TA_unpert = T_A(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)
                                                                    TS_unpert = T_S(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)
                                                                    TE_total = T_E(N0, N1, l0, l1 , J0[j0], J1[j1], K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])
                                                                    
                                                                    TA_unpert_expected = T_A_unpert_expected(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)
                                                                    TS_unpert_expected = T_S_unpert_expected(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)
                                                                    TE_total_expected = T_E_unpert_expected(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])                                                                 
                                                                
                                                                    TA_pert_0 = T_A(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                                    TS_pert_0 = T_S(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                                     
                                                         
                                                                

                                                                # Version that just gives you Lymann line contribution
                                                             
                                                                 
                                                                if N1 < N0 and K0 == K1:
                                                                    Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_unpert
                                                                    #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_pert_0
                                                                    Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_source_unpert
                                                                    #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_source_pert_0
                                                                    
                                                                    if N0==2 and l0==0 and N1==1 and l1==0:
                                                                        Lambda0[N0,l0,j0,k0,f0,f1,N1,l1,j1,k1,f2,f3] += two_photon_unpert
                                                                        L0[N0,l0,j0,k0,f0,f1,N1,l1,j1,k1,f2,f3] += two_photon_pert_0                                                                        
                                                                        #pass
                                                                    if N0==2 and l0==1 and N1==1 and l1==0:
                                                                        Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_unpert + Lymann_source_unpert
                                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_unpert
                                                                                                                                                 
                                                                        #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_pert_0 + Lymann_source_pert_0
                                                                        #pass
                                                                    elif N1>1:                                                                 
                                                                        Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_unpert
                                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_unpert - TA_unpert_expected                                                                         
                                                                        L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_pert_0 
                                                                        #pass
                                                                        
                                                                        
                                                                elif N1 > N0 and K0 == K1:

                                                                    if N0==1 and l0==0 and N1==2 and l1==0:
                                                                        Lambda0[N0,l0,j0,k0,f0,f1,N1,l1,j1,k1,f2,f3] += two_photon_unpert
                                                                        L0[N0,l0,j0,k0,f0,f1,N1,l1,j1,k1,f2,f3] += two_photon_pert_0                                                                       
                                                                        #pass
                                                                                                                                        
                                                                    if N0==1 and N1==2 and l0==0 and l1==1:
                                                                        Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_unpert+Lymann_source_unpert
                                                                        #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_pert_0+Lymann_source_pert_0
                                                                        Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TE_total
                                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TE_total - TE_total_expected                                                                       
                                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_source_unpert
                                                                        #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_source_pert_0
                                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_unpert
                                                                        L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_pert_0 + Lymann_source_pert_0 
                                                                        #pass                                                                        
                                                                    elif N0>1:
                                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TS_unpert                                                                  
                                                                        Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TE_total + TS_unpert
                                                                        #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TE_total + TS_unpert - TE_total_expected - TS_unpert_expected                                                                       
                                                                        #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TS_pert_0 
                                                                        #pass                                                                  
                                                                elif N1 < N0 and K0 == K1 :
                                                                    if N0==2 and N1==1 and l0==1 and l1==0: 
                                                                        Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_unpert
                                                                        Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_source_unpert
                                                                        #pass
                                                                    
                                                                elif N1 > N0 and K0 == 2 and K1 == 2:
                                                                    Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_unpert+Lymann_source_unpert  
                                                                    #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_unpert   
                                                                    #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_source_unpert 
                                                                    #pass
                                                                    

                                                            elif Kr == 2:
                                                            
                                                                if N0 != 1 and N0!= 2 and N1 != 1 and N1 != 2 and Kr == 2:                                                                
                                                                    TA_pert_2 = T_A(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                                    TS_pert_2 = T_S(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                                                                                                 
                                                                Lymann_BB_Kr_2 = Lambda_lymann_lines_BB(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,True)  
                                                                Lymann_source_Kr_2 = Lambda_lymann_lines_source(N0,N1,l0,l1,J0[j0],J1[j1],K0,K1,Kr,F0[f0],F1[f1],F2[f2],F3[f3],T,X_e,h_scalar,h_prime,eta_scalar,eta_prime,t_b,optical_depth,True)                                                                 

                                                                #print("TA_pert_2: "+str(TA_pert_2), file=output_file)
                                                                #print("TS_pert_2: "+str(TS_pert_2), file=output_file)
                                                                
                                                                #print("Lymann_pert_2: "+str(Lymann_pert_2), file=output_file)                                                                

                                                                if N1 < N0 and K0 == 2 and K1==0:
                                                                    #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_Kr_2 
                                                                    L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_Kr_2 
                                                                    L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_source_Kr_2                                                                                                                                                                                         
                                                                    L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_pert_2
                                                                    #pass

                                                                elif N1 > N0 and K0 == 2 and K1==0:
                                                                    #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_Kr_2 + Lymann_source_Kr_2 
                                                                    L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_BB_Kr_2                                                                                 
                                                                    L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_source_Kr_2                                                                                
                                                                    L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TS_pert_2
                                                                    #pass

                                                             
                                                                                                     
                                                                                             

                                                            #print("Lambda0: "+str(Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)
                                                            #print("L0: "+str(L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)                                                                                                                        
                                                            #print("L2: "+str(L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)                                                    
                                                          
                                                                



output_file.close()

    

# We want to partition this matrix into different terms depending on whether we are focusing on the 1s state or the excited states.

Lambda0_1s_1s = Lambda0[1:2,0:1,:,:,:,:,1:2,0:1,:,:,:,:]
Lambda0_1s_exc = Lambda0[1:2,0:1,:,:,:,:,2:numN+1,:,:,:,:,:]
Lambda0_exc_1s = Lambda0[2:numN+1,:,:,:,:,:,1:2,0:1,:,:,:]
Lambda0_exc_exc = Lambda0[2:numN+1,:,:,:,:,:,2:numN+1,:,:,:,:,:]

L0_1s_1s = L0[1:2,0:1,:,:,:,:,1:2,0:1,:,:,:,:]
L0_1s_exc = L0[1:2,0:1,:,:,:,:,2:numN+1,:,:,:,:,:]
L0_exc_1s = L0[2:numN+1,:,:,:,:,:,1:2,0:1,:,:,:]
L0_exc_exc = L0[2:numN+1,:,:,:,:,:,2:numN+1,:,:,:,:,:]

L2_1s_1s = L2[1:2,0:1,:,:,:,:,1:2,0:1,:,:,:,:]
L2_1s_exc = L2[1:2,0:1,:,:,:,:,2:numN+1,:,:,:,:,:]
L2_exc_1s = L2[2:numN+1,:,:,:,:,:,1:2,0:1,:,:,:]
L2_exc_exc = L2[2:numN+1,:,:,:,:,:,2:numN+1,:,:,:,:,:]

mask_array_1s_1s = mask_array[1:2,0:1,:,:,:,:,1:2,0:1,:,:,:,:]
mask_array_1s_exc = mask_array[1:2,0:1,:,:,:,:,2:numN+1,:,:,:,:,:]
mask_array_exc_1s = mask_array[2:numN+1,:,:,:,:,:,1:2,0:1,:,:,:]
mask_array_exc_exc = mask_array[2:numN+1,:,:,:,:,:,2:numN+1,:,:,:,:,:]




# Next, we want to make each of these arrays into a 2-dim array so we can use matrix multiplication


num_1s_phy = 3

if numN == 2:
        num_exc_phy = 12
elif numN == 3:
        num_exc_phy = 36
else:
        num_exc_phy = 12*(numN-1) + 36
    


    

num_1s_total = 16
num_exc_total = 16*numN*(numN-1)
        

    

N_1s = 16 # Number of elements in the 1s aarray
N_exc = 16*numN*(numN-1) # Number of elements of the excited state array

Lambda0_1s_1s = Lambda0_1s_1s.reshape ( (N_1s,N_1s))
Lambda0_1s_exc = Lambda0_1s_exc.reshape( (N_1s, N_exc))
Lambda0_exc_1s = Lambda0_exc_1s.reshape ( (N_exc,N_1s))
Lambda0_exc_exc = Lambda0_exc_exc.reshape( (N_exc, N_exc))

L0_1s_1s = L0_1s_1s.reshape ( (N_1s,N_1s))
L0_1s_exc = L0_1s_exc.reshape( (N_1s, N_exc))
L0_exc_1s = L0_exc_1s. reshape( (N_exc, N_1s))
L0_exc_exc = L0_exc_exc.reshape( (N_exc,N_exc))

L2_1s_1s = L2_1s_1s.reshape ( (N_1s,N_1s))
L2_1s_exc = L2_1s_exc.reshape( (N_1s, N_exc))
L2_exc_1s = L2_exc_1s. reshape( (N_exc, N_1s))
L2_exc_exc = L2_exc_exc.reshape( (N_exc,N_exc))


mask_array_1s_1s = mask_array_1s_1s.reshape( (N_1s,N_1s) )
mask_array_1s_exc = mask_array_1s_exc.reshape( (N_1s,N_exc) )
mask_array_exc_1s = mask_array_exc_1s.reshape( (N_exc,N_1s) )
mask_array_exc_exc = mask_array_exc_exc.reshape( (N_exc,N_exc) )


Lambda0_exc_exc_masked = np.zeros( (num_exc_phy, num_exc_phy), dtype = complex)
Lambda0_1s_exc_masked = np.zeros( (num_1s_phy, num_exc_phy), dtype = complex)
Lambda0_exc_1s_masked = np.zeros( (num_exc_phy, num_1s_phy), dtype = complex)
Lambda0_1s_1s_masked = np.zeros( (num_1s_phy, num_1s_phy), dtype = complex)

L0_exc_exc_masked = np.zeros( (num_exc_phy, num_exc_phy), dtype = complex)
L0_1s_exc_masked = np.zeros( (num_1s_phy, num_exc_phy), dtype = complex)
L0_exc_1s_masked = np.zeros( (num_exc_phy, num_1s_phy), dtype = complex)
L0_1s_1s_masked = np.zeros( (num_1s_phy, num_1s_phy), dtype = complex)


L2_exc_exc_masked = np.zeros( (num_exc_phy, num_exc_phy), dtype = complex)
L2_1s_exc_masked = np.zeros( (num_1s_phy, num_exc_phy), dtype = complex)
L2_exc_1s_masked = np.zeros( (num_exc_phy, num_1s_phy), dtype = complex)
L2_1s_1s_masked = np.zeros( (num_1s_phy, num_1s_phy), dtype = complex)

# We want to make a matrix of what values we should expect for the exc,exc case



           
                   
        




# Masking each array

# First masking the exc_exc array
x_counter = 0
y_counter = 0

for i in range(N_exc):
    for j in range(N_exc):
        
        if mask_array_exc_exc[i,j] == True:
            Lambda0_exc_exc_masked[x_counter,y_counter] = Lambda0_exc_exc[i,j]
            L0_exc_exc_masked[x_counter,y_counter] = L0_exc_exc[i,j]
            L2_exc_exc_masked[x_counter,y_counter] = L2_exc_exc[i,j]
 
            #print(x_counter)
            #print(y_counter)
            y_counter += 1
            
            
            if y_counter == num_exc_phy:
                x_counter += 1
                y_counter = 0
            
# First masking the 1s_exc array

x_counter = 0
y_counter = 0

for i in range(N_1s):
    for j in range(N_exc):
        
        if mask_array_1s_exc[i,j] == True:
            Lambda0_1s_exc_masked[x_counter,y_counter] = Lambda0_1s_exc[i,j]
            L0_1s_exc_masked[x_counter,y_counter] = L0_1s_exc[i,j]
            L2_1s_exc_masked[x_counter,y_counter] = L2_1s_exc[i,j]



            #print(x_counter)
            #print(y_counter)
            y_counter += 1
            
            
            if y_counter == num_exc_phy:
                x_counter += 1
                y_counter = 0

# First masking the exc_1s array

x_counter = 0
y_counter = 0

for i in range(N_exc):
    for j in range(N_1s):
        
        if mask_array_exc_1s[i,j] == True:
            
            Lambda0_exc_1s_masked[x_counter,y_counter] = Lambda0_exc_1s[i,j]
            L0_exc_1s_masked[x_counter,y_counter] = L0_exc_1s[i,j]
            L2_exc_1s_masked[x_counter,y_counter] = L2_exc_1s[i,j]


            #print(x_counter)
            #print(y_counter)
            y_counter += 1
            
            
            if y_counter == num_1s_phy:
                x_counter += 1
                y_counter = 0


# First masking the 1s_1s array

x_counter = 0
y_counter = 0

for i in range(N_1s):
    for j in range(N_1s):
        
        if mask_array_1s_1s[i,j] == True:
            Lambda0_1s_1s_masked[x_counter,y_counter] = Lambda0_1s_1s[i,j]
            L0_1s_1s_masked[x_counter,y_counter] = L0_1s_1s[i,j]
            L2_1s_1s_masked[x_counter,y_counter] = L2_1s_1s[i,j]

            #print(x_counter)
            #print(y_counter)
            y_counter += 1
            
            
            if y_counter == num_1s_phy:
                x_counter += 1
                y_counter = 0
                
# Script to get rid of the diagonal elements in Lambda_exc_exc

'''
for i in range(36):
    for j in range(36):
        if i<3 and i==j:
            Lambda0_exc_exc_masked[i,j] += 2*h*freq_3s2p**3/c**2*B_2s3p*phase_space_32
            #pass
        elif i>2 and i< 6 and i==j:
            Lambda0_exc_exc_masked[i,j] += A_2p1s*(1+phase_space_21) + 2*h*freq_3s2p**3*phase_space_32*B_2p3s/c**2 + 2*h*freq_3d2p**3*phase_space_32*B_2p_1half_3d/c**2
        elif i > 5 and i <12 and i==j:
            Lambda0_exc_exc_masked[i,j] += A_2p1s*(1+phase_space_21) + 2*h*freq_3s2p**3*phase_space_32*B_2p3s/c**2 + 2*h*freq_3d2p**3*phase_space_32*B_2p_3half_3d/c**2
        elif i>11 and i<15 and i==j:
            Lambda0_exc_exc_masked[i,j] += 2*A_3s2p*(1+phase_space_32)
        elif i>14 and i< 24 and i==j:
            Lambda0_exc_exc_masked[i,j] += A_3p2s*(1+phase_space_32)
        elif i>23 and i<30 and i==j:
            Lambda0_exc_exc_masked[i,j] += 2*A_3d2p*(1+phase_space_32)        
        elif i>29 and i==j:
            Lambda0_exc_exc_masked[i,j] += A_3d2p*(1+phase_space_32)
'''
            
            



# Saving into fits files for Lambda0

hdu1 = fits.PrimaryHDU(np.real(Lambda0_1s_1s_masked))
hdu1.writeto("lambda0_1s_1s_z="+str(redshift)+"_real.fits", overwrite = True)

hdu2 = fits.PrimaryHDU(np.imag(Lambda0_1s_1s_masked))
hdu2.writeto("lambda0_1s_1s_z="+str(redshift)+"_imag.fits", overwrite = True)

hdu3 = fits.PrimaryHDU(np.real(Lambda0_1s_exc_masked))
hdu3.writeto("lambda0_1s_exc_z="+str(redshift)+"_real.fits", overwrite = True)

hdu4 = fits.PrimaryHDU(np.imag(Lambda0_1s_exc_masked))
hdu4.writeto("lambda0_1s_exc_z="+str(redshift)+"_imag.fits", overwrite = True)


hdu5 = fits.PrimaryHDU(np.real(Lambda0_exc_1s_masked))
hdu5.writeto("lambda0_exc_1s_z="+str(redshift)+"_real.fits", overwrite = True)

hdu6 = fits.PrimaryHDU(np.imag(Lambda0_exc_1s_masked))
hdu6.writeto("lambda0_exc_1s_z="+str(redshift)+"_imag.fits", overwrite = True)

hdu7 = fits.PrimaryHDU(np.real(Lambda0_exc_exc_masked))
hdu7.writeto("lambda0_exc_exc_z="+str(redshift)+"_real.fits", overwrite = True)

hdu8 = fits.PrimaryHDU(np.imag(Lambda0_exc_exc_masked))
hdu8.writeto("lambda0_exc_exc_z="+str(redshift)+"_imag.fits", overwrite = True)

# Saving into fits files for L0

hdu9 = fits.PrimaryHDU(np.real(L0_1s_1s_masked))
hdu9.writeto("L0_1s_1s_z="+str(redshift)+"_real.fits", overwrite = True)

hdu10 = fits.PrimaryHDU(np.imag(L0_1s_1s_masked))
hdu10.writeto("L0_1s_1s_z="+str(redshift)+"_imag.fits", overwrite = True)

hdu11 = fits.PrimaryHDU(np.real(L0_1s_exc_masked))
hdu11.writeto("L0_1s_exc_z="+str(redshift)+"_real.fits", overwrite = True)

hdu12 = fits.PrimaryHDU(np.imag(L0_1s_exc_masked))
hdu12.writeto("L0_1s_exc_z="+str(redshift)+"_imag.fits", overwrite = True)

hdu13 = fits.PrimaryHDU(np.real(L0_exc_1s_masked))
hdu13.writeto("L0_exc_1s_z="+str(redshift)+"_real.fits", overwrite = True)

hdu14 = fits.PrimaryHDU(np.imag(L0_exc_1s_masked))
hdu14.writeto("L0_exc_1s_z="+str(redshift)+"_imag.fits", overwrite = True)

hdu15 = fits.PrimaryHDU(np.real(L0_exc_exc_masked))
hdu15.writeto("L0_exc_exc_z="+str(redshift)+"_real.fits", overwrite = True)

hdu16 = fits.PrimaryHDU(np.imag(L0_exc_exc_masked))
hdu16.writeto("L0_exc_exc_z="+str(redshift)+"_imag.fits", overwrite = True)

# Saving into fits files for L2

hdu17 = fits.PrimaryHDU(np.real(L2_1s_1s_masked))
hdu17.writeto("L2_1s_1s_z="+str(redshift)+"_real.fits", overwrite = True)

hdu18 = fits.PrimaryHDU(np.imag(L2_1s_1s_masked))
hdu18.writeto("L2_1s_1s_z="+str(redshift)+"_imag.fits", overwrite = True)

hdu19 = fits.PrimaryHDU(np.real(L2_1s_exc_masked))
hdu19.writeto("L2_1s_exc_z="+str(redshift)+"_real.fits", overwrite = True)

hdu20 = fits.PrimaryHDU(np.imag(L2_1s_exc_masked))
hdu20.writeto("L2_1s_exc_z="+str(redshift)+"_imag.fits", overwrite = True)

hdu21 = fits.PrimaryHDU(np.real(L2_exc_1s_masked))
hdu21.writeto("L2_exc_1s_z="+str(redshift)+"_real.fits", overwrite = True)

hdu22 = fits.PrimaryHDU(np.imag(L2_exc_1s_masked))
hdu22.writeto("L2_exc_1s_z="+str(redshift)+"_imag.fits", overwrite = True)

hdu23 = fits.PrimaryHDU(np.real(L2_exc_exc_masked))
hdu23.writeto("L2_exc_exc_z="+str(redshift)+"_real.fits", overwrite = True)

hdu24 = fits.PrimaryHDU(np.imag(L2_exc_exc_masked))
hdu24.writeto("L2_exc_exc_z="+str(redshift)+"_imag.fits", overwrite = True)


# Next, take the inverse of each matrix just in case we need to use them later

#Lambda0_1s_1s_inv = np.linalg.inv(Lambda0_1s_1s_masked)
#Lambda0_1s_exc_inv = np.linalg.inv(Lambda0_1s_exc_masked)
#Lambda0_exc_1s_inv = np.linalg.inv(Lambda0_exc_1s_masked)
#Lambda0_exc_exc_inv = np.linalg.inv(Lambda0_exc_exc_masked)

#L0_1s_1s_inv = np.linalg.inv(L0_1s_1s_masked)
#L0_1s_exc_inv = np.linalg.inv(L0_1s_exc_masked)
#L0_exc_1s_inv = np.linalg.inv(L0_exc_1s_masked)
#L0_exc_exc_inv = np.linalg.inv(L0_exc_exc_masked)

#L2_1s_1s_inv = np.linalg.inv(L2_1s_1s_masked)
#L2_1s_exc_inv = np.linalg.inv(L2_1s_exc_masked)
#L2_exc_1s_inv = np.linalg.inv(L2_exc_1s_masked)
#L2_exc_exc_inv = np.linalg.inv(L2_exc_exc_masked)


# Source function

source_matrix_unpert = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = complex)
source_matrix_pert_0 = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = complex)
source_matrix_pert_2 = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = complex)

mask_matrix = np.zeros((numN+1, numL, numJ, numK, numF, numF), dtype = complex)

for N in range(1, numN+1,1):
    for L in range(numN):
        
        J = np.arange( np.abs(L-S), L+S+1, 1)
        
        for j in range(len(J)):
            
            F = np.arange( np.abs(J[j]-I), J[j]+I+1, 1)
            
            for k in range(numK):
                
                for f0 in range(len(F)):
                    for f1 in range(len(F)):
                    
                    
                        if mask_allowed(N,L,j,k, f0, f1) == True:
                            
                            mask_matrix[N,L,j,k,f0,f1] = True
                        
                        else:
                            
                            mask_matrix[N,L,j,k,f0,f1] = False
            
                        
                        if k == 0 and N != 1:
                            
                            source_matrix_unpert[N,L,j,k,f0,f1] = source_boundfree_spontaneous(N,L,j,k,f0,f1,energy_array)
                            source_matrix_unpert[N,L,j,k,f0,f1] += source_boundfree_stimulated(N,L,j,k,f0,f1,False,energy_array)

                            source_matrix_pert_0[N,L,j,k,f0,f1] = source_boundfree_stimulated(N,L,j,k,f0,f1,True,energy_array)
                            
                        elif k == 2 and N!= 1:

                            source_matrix_pert_2[N,L,j,k,f0,f1] = source_boundfree_stimulated(N,L,j,k,f0,f1,True,energy_array)


            
            
            
source_matrix_1s_unpert = source_matrix_unpert[1:2,0:1,:,:,:,:]
source_matrix_1s_pert_0 = source_matrix_pert_0[1:2,0:1,:,:,:,:]
source_matrix_1s_pert_2 = source_matrix_pert_2[1:2,0:1,:,:,:,:]

source_matrix_exc_unpert = source_matrix_unpert[2:numN+1,:,:,:,:,:]
source_matrix_exc_pert_0 = source_matrix_pert_0[2:numN+1,:,:,:,:,:]
source_matrix_exc_pert_2 = source_matrix_pert_2[2:numN+1,:,:,:,:,:]

'''
source_matrix_exc = source_matrix[2:numN+1,:,:,:,:,:]
'''

source_matrix_1s_unpert = source_matrix_1s_unpert.reshape(N_1s)
source_matrix_1s_pert_0 = source_matrix_1s_pert_0.reshape(N_1s)
source_matrix_1s_pert_2 = source_matrix_1s_pert_2.reshape(N_1s)

source_matrix_exc_unpert = source_matrix_exc_unpert.reshape(N_exc)
source_matrix_exc_pert_0 = source_matrix_exc_pert_0.reshape(N_exc)
source_matrix_exc_pert_2 = source_matrix_exc_pert_2.reshape(N_exc)

'''
source_matrix_1s = source_matrix_1s.reshape(N_1s)
source_matrix_exc = source_matrix_exc.reshape(N_exc)
'''


mask_matrix_1s = mask_matrix[1:2,0:1,:,:,:,:]
mask_matrix_exc = mask_matrix[2:numN+1,:,:,:,:,:]

mask_matrix_1s = mask_matrix_1s.reshape(N_1s)
mask_matrix_exc = mask_matrix_exc.reshape(N_exc)


source_1s_unpert_masked = np.zeros(num_1s_phy)
source_1s_pert_0_masked = np.zeros(num_1s_phy)
source_1s_pert_2_masked = np.zeros(num_1s_phy)


source_exc_unpert_masked = np.zeros(num_exc_phy)
source_exc_pert_0_masked = np.zeros(num_exc_phy)
source_exc_pert_2_masked = np.zeros(num_exc_phy)


x_counter = 0


for i in range(N_1s):
        
    if mask_matrix_1s[i] == True:
        
        source_1s_unpert_masked[x_counter] = source_matrix_1s_unpert[i]
        source_1s_pert_0_masked[x_counter] = source_matrix_1s_pert_0[i]
        source_1s_pert_2_masked[x_counter] = source_matrix_1s_pert_2[i]

        
        x_counter += 1




x_counter = 0

for i in range(N_exc):
        
    if mask_matrix_exc[i] == True:
        
        source_exc_unpert_masked[x_counter] = source_matrix_exc_unpert[i]
        source_exc_pert_0_masked[x_counter] = source_matrix_exc_pert_0[i]
        source_exc_pert_2_masked[x_counter] = source_matrix_exc_pert_2[i]

        
        x_counter += 1

        
'''
    
        
We will now calculate the density matrix in the steady-state limit. 
        
        
# First, we will calculate the unpreturbed excited  density matrix.

density_exc_unpert = - np.dot( np.dot( Lambda0_exc_exc_inv, Lambda0_exc_1s_masked), density_1s_unpert)
density_exc_unpert += - np.dot( Lambda0_exc_exc_inv, source_exc_unpert_masked)
        
# Next, we compute the Runge Kutta coefficients for the 1S density matrix such that
# drho_1s/1dt = A rho_1s + B
        
A = Lambda0_1s_1s_masked - np.dot( Lambda0_1s_exc_masked, np.dot(Lambda0_exc_exc_inv,Lambda0_exc_1s_masked))
B = source_1s_unpert_masked - np.dot(Lambda0_exc_exc_inv, source_exc_unpert_masked
        
factor_t_to_z = -(1+redshift)*H_param # factor that converts our equation to an ODE in regards to redshift
        
# We define our integration step delta_z as the difference between two redhsift entrees
        
delta_z = thermo_array[0][0]-thermo_array[0][1] # subtracting difference between two redshifts
        
# Computing the Runge-Kutta coefficients
        
K0 = np.dot(A, rho_1s_unpert) + B
K1 = 
'''
        
        
                



# Forming fits files from the source function       

hdu_source_1s_unpert = fits.PrimaryHDU(source_1s_unpert_masked)
hdu_source_1s_unpert.writeto("source_1s_unpert_z="+str(redshift)+".fits", overwrite = True)

hdu_source_1s_pert_0 = fits.PrimaryHDU(source_1s_pert_0_masked)
hdu_source_1s_pert_0.writeto("source_1s_pert_0_z="+str(redshift)+".fits", overwrite = True)

hdu_source_1s_pert_2 = fits.PrimaryHDU(source_1s_pert_2_masked)
hdu_source_1s_pert_2.writeto("source_1s_pert_2_z="+str(redshift)+".fits", overwrite = True)

hdu_source_exc_unpert = fits.PrimaryHDU(source_exc_unpert_masked)
hdu_source_exc_unpert.writeto("source_exc_unpert_z="+str(redshift)+".fits", overwrite = True)

hdu_source_exc_pert_0 = fits.PrimaryHDU(source_exc_pert_0_masked)
hdu_source_exc_pert_0.writeto("source_exc_pert_0_z="+str(redshift)+".fits", overwrite = True)

hdu_source_exc_pert_2 = fits.PrimaryHDU(source_exc_pert_2_masked)
hdu_source_exc_pert_2.writeto("source_exc_pert_2_z="+str(redshift)+".fits", overwrite = True)

# Creating a fits file for the density matrix

#hdu_density_1s_unpert = fits.PrimaryHDU(np.abs(density_1s_unpert))
#hdu_density_1s_unpert.writeto("density_1s_unpert.fits", overwrite = True)

#hdu_density_exc_unpert = fits.PrimaryHDU(np.abs(density_exc_unpert))
#hdu_density_exc_unpert.writeto("density_exc_unpert_z="+str(redshift)+".fits", overwrite = True)




