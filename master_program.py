import os
import sys
import subprocess
import numpy as np
import cProfile
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy.special import zeta


    
# Defining constants that we need to use

# Numerical quantities towards solving the Saha equation

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
He_abund = 0.25 # Primordial Helium Abundance
ion_energy = m_electron*e0**4/(2*hbar**2)

# Einstein coefficients (s^{-1})

A_2photon = 8.22 


# Defining number of elements in each array given N_max

N_max = 4
print("Maximum Excited State N="+str(N_max))

num_1s = 3

if N_max == 2:
        num_exc = 12
elif N_max == 3:
        num_exc = 36
else:
        num_exc = 12*(N_max-1) + 36

# We want to make a module that imports the T(k) values for a given k. We will then loop over it
# and then perform our RK4 program.

redshift_list = np.loadtxt(open("redshift_list.txt"))

# Defining spacing between z_i and z_i+1
z_index =  redshift_list[1]-redshift_list[0]

# Current wave number 
wave_num = 1.042057218184e-05

# Opens files from CLASS we use in our analysis
thermo_array = np.loadtxt(open("thermodynamics.dat"))
tk_array = np.loadtxt(open("tk_"+str(redshift_list[0])+".dat"))

#Boolian statement toward finding the correct file.
found_kline = False
found_zline = False


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
        
        if thermo_array[line][1] == redshift_list[0]:
            #print("The redshift in the file is "+str(thermo_array[line][1]))
            #print("The redshift is in line "+str(line))
            
            zline = line
            found_zline = True
    
    if found_zline == False:        
        redshift = input("Redshift not found. Please input a new redshift: ")


# Defining arrays to compute the evolution of the density matrix components as a function of redshift

rho_1s_unpert = np.zeros((len(redshift_list),num_1s))
rho_exc_unpert = np.zeros((len(redshift_list),num_exc))
source_1s_unpert = np.zeros((len(redshift_list),num_1s))
source_exc_unpert = np.zeros((len(redshift_list), num_exc))

rho_1s_pert0 = np.zeros((len(redshift_list),num_1s))
rho_exc_pert0 = np.zeros((len(redshift_list),num_exc))
source_1s_pert0 = np.zeros((len(redshift_list),num_1s))
source_exc_pert0 = np.zeros((len(redshift_list), num_exc))

rho_1s_pert0 = rho_1s_pert0*complex(1,0)
rho_exc_pert0 = rho_exc_pert0*complex(1,0)
source_1s_pert0 = source_1s_pert0*complex(1,0)
source_exc_pert0 = source_exc_pert0*complex(1,0)


rho_1s_pert2 = np.zeros((len(redshift_list),num_1s))
rho_exc_pert2 = np.zeros((len(redshift_list),num_exc))
source_1s_pert2 = np.zeros((len(redshift_list),num_1s))
source_exc_pert2 = np.zeros((len(redshift_list), num_exc))

rho_1s_pert2 = rho_1s_pert2*complex(1,0)
rho_exc_pert2 = rho_exc_pert2*complex(1,0)
source_1s_pert2 = source_1s_pert2*complex(1,0)
source_exc_pert2 = source_exc_pert2*complex(1,0)

X_1s_Saha = np.zeros(len(redshift_list))
X_1s_Peebles = np.zeros(len(redshift_list))

X_1s_deriv_Peebles = np.zeros(len(redshift_list)-1)


X_2s, X_2p, X_3s, X_3p, X_3d = [np.zeros(len(redshift_list)) for i in range(5)] # Arrays for the numerical results for the excited states
X_4s, X_4p, X_4d, X_4f = [np.zeros(len(redshift_list)) for i in range(4)]
Saha_2s, Saha_2p, Saha_3s, Saha_3p, Saha_3d = [np.zeros(len(redshift_list)) for i in range(5)] # Analytic Saha equation array for the excited states
Saha_4s, Saha_4p, Saha_4d, Saha_4f = [np.zeros(len(redshift_list)) for i in range(4)]
ratio_2s, ratio_2p, ratio_3s, ratio_3p, ratio_3d = [np.zeros(len(redshift_list)) for i in range(5)] # ratio of code results vs. analytic Saha equation (i.e. ratio_2s = X_2s/Saha_2s)
ratio_4s, ratio_4p, ratio_4d, ratio_4f = [np.zeros(len(redshift_list)) for i in range(4)]



'''
# Defining arrays for the source function components

source_1s_Fis0 = np.zeros(len(redshift_list)-1)
source_1s_Fis1 = np.zeros(len(redshift_list)-1)

source_exc_2s_Fis0 = np.zeros(len(redshift_list)-1)
source_exc_2s_Fis1 = np.zeros(len(redshift_list)-1)
source_exc_2p_half_Fis0 = np.zeros(len(redshift_list)-1)
source_exc_2p_half_Fis1 = np.zeros(len(redshift_list)-1)
source_exc_2p_3half_Fis0 = np.zeros(len(redshift_list)-1)
source_exc_2p_3half_Fis1 = np.zeros(len(redshift_list)-1)
'''


# Defining lists for the eigenvalues of the matrix A = Lambda0_1s,1s - Lambda0_1s_exc Lambda0_exc_exc^-1 Lambda0_exc,1s 
# where drho_1s/dz = A rho_1s + B (B is some function of the source terms).

eigenvalue_unpert = np.zeros((len(redshift_list), num_1s))

# Defining lists for the function F(rho_1s,z) such that drho_1s/dz = F(rho_1s,z)

deriv_func_unpert = np.zeros((len(redshift_list), num_1s))

# Defining arrays for A and B matrix where drho_1s/dz = A*rho_1s + B

A_unpert = np.zeros((len(redshift_list),num_1s))
B_unpert = np.zeros((len(redshift_list),num_1s))


# ######################################################################################### Now we run our code for different iterations #############################################################################################

# Running code for i=0 

# Get values from CLASS for the initial conditions i=0,1

X_e_init = thermo_array[zline][3]
X_1s_init = 1-X_e_init

# Getting an array of X_1s values and calculating derivatives

X_1s_CLASS = np.zeros(len(redshift_list))
X_1s_deriv_CLASS = np.zeros(len(redshift_list)-1)

for i in range(len(redshift_list)):

    found_zline = False

    while found_zline == False:
        for line in range(len(thermo_array)):
        
            if thermo_array[line][1] == redshift_list[i]:
                #print("The redshift in the file is "+str(thermo_array[line][1]))
                #print("The redshift is in line "+str(line))
            
                zline = line
                found_zline = True
                
                X_1s_CLASS[i] = 1-thermo_array[zline][3]
                
                if i > 0:
                    X_1s_deriv_CLASS[i-1] = (X_1s_CLASS[i] - X_1s_CLASS[i-1]) / z_index
    
        if found_zline == False:        
            redshift = input("Redshift not found. Please input a new redshift: ")


# Calculate what the Hubble parameter based on Plack 2018 best fit values. (or CLASS values)

redshift = redshift_list[0]

T0 = 2.7255
T = T0*(1+redshift_list[0])
T_star = 0.0681 # Kelvin

little_h = 0.67810 # H0 = gamma_100*h
gamma_100 = 3.24078e-18 # gamma_100 = 100 km/s/Mpc

rho_crit_today = 3*gamma_100**2*little_h**2/ (8*np.pi*G_Newton)
rho_rad_today = (np.pi**2/15) * (kB*T0)**4 / (hbar**3*c**5)

# omega_X = Omega_X * h^2 = (rho_X/rho_crit) * h^2

omega_cdm = 0.1201075
omega_b = 0.02238280
omega_m = omega_b + omega_cdm
omega_r = (rho_rad_today / rho_crit_today)*little_h**2
omega_DE = little_h**2 - omega_m - omega_r 

# Hubble parameter H(z) where H(z=0) = H0

H0 = little_h*gamma_100
H_param = gamma_100 * np.sqrt(omega_r*(1+redshift)**4 + omega_m*(1+redshift)**3  + omega_DE)

num_baryon_today = (omega_b*rho_crit_today)/(m_proton*little_h**2)
num_photon_today = (2*zeta(3)/np.pi**2)*(kB*T0)**3/(hbar*c)**3
baryon_to_photon_ratio = num_baryon_today / num_photon_today



def Saha_eqn(N,X_e,T):

    '''
    N is the excited state you want to acquire.
    
    X_e is the population of free electrons.
    
    T is the temperature
    '''

    term = X_e**2 * (2*np.pi*hbar**2/(m_electron*kB*T))**(3/2) * np.exp(ion_energy/(N**2*kB*T))

    num_den_baryon = num_baryon_today*(T/T0)**3
    
    num_den_Htot = (1-He_abund)*num_den_baryon
     
    


    return term*num_den_Htot
    
def Peebles_eqn(X_e,redshift):

    A_2p = 625564808.203944
    freq_lya = 2467449149208226 # s^-1
    
    # Hubble parameter values
    
    H_param = gamma_100 * np.sqrt(omega_r*(1+redshift)**4 + omega_m*(1+redshift)**3  + omega_DE)
    
    # Temperature
    
    T = T0*(1+redshift)
    
    num_den_baryon = num_baryon_today*(T/T0)**3
    
    num_den_Htot = (1-He_abund)*num_den_baryon
  

    # Baryon Density (cm^-3)
    '''
    num_den_baryon = (2*zeta(3)/np.pi**2)
    num_den_baryon *= (kB*T/ (hbar*c))**3
    num_den_baryon *= baryon_to_photon_ratio
    '''

    # Recombination coefficient (cm^3/s)
    
    rec_coeff = 4.309e-13*(T/1e4)**(-0.6166) / (1 + 0.6703*(T/1e4)**0.53)
    photo_coeff = (rec_coeff/4)*(m_electron*kB*T / (2*np.pi*hbar**2))**(3/2) *np.exp(-ion_energy/(4*kB*T))
    #photo_coeff = 0
    
    # Optical depth
    
    optical_depth = 3*np.pi**2 * A_2p * num_den_Htot *(1-X_e)*  c**3/ ( H_param * (2*np.pi*freq_lya)**3 )
    P_esc = 1/optical_depth
    
        
    # Rate coefficients
    
    
    
    rate_2photon = 8.22
    rate_Lya = 3*P_esc*A_2p

        
    eqn_term = (rate_2photon+rate_Lya)/(rate_2photon + rate_Lya + 4*photo_coeff)
    eqn_term *= rec_coeff*(num_den_Htot * X_e**2 - (m_electron*kB*T / (2*np.pi*hbar**2))**(3/2) * (1-X_e)*np.exp(-ion_energy/(kB*T)))
    
    eqn_term *= -1/(H_param*(1+redshift))
    
    return eqn_term


# Calculating the initial density matrix components:
# rho_1s_unpert[0] = X_1s^(F=0) , rho_1s_unpert[1] = X_1s^(F=1)/ sqrt(3)

rho_1s_unpert[0,0] = X_1s_init/(1+ 3*np.exp(-T_star/T) )
rho_1s_unpert[0,1] = np.sqrt(3)*X_1s_init*np.exp(-T_star/T) / (1 + 3*np.exp(-T_star/T))
rho_1s_unpert[0,2] = 0

# Initial condition (Polarized atom)

rho_1s_pert2[0,2] = 0

print(" ")    
print("rho_1s_unpert at z="+str(redshift_list[0]))
print(rho_1s_unpert[0])

# Reshaping our array to be viewable on ds9
#rho_1s_unpert = rho_1s_unpert.reshape(3,1) 

#hdu_rho_1s_unpert = fits.PrimaryHDU(rho_1s_unpert )
#hdu_rho_1s_unpert.writeto("rho_1s_unpert_z="+str(redshift_list[0])+".fits", overwrite = True)


#rho_1s_unpert = rho_1s_unpert.reshape(3)

# Making rho_1s_unpert complex

rho_1s_unpert = rho_1s_unpert*complex(1,0)

# Saving list of values that will be used in new_matrices_code.py where the Lambda matrices and the source terms are calculated.
# The contents are the current redshift and the three density matrix components which are saved onto a text file.

argv_list = [redshift, np.real(rho_1s_unpert[0,0]), np.real(rho_1s_unpert[0,1]), np.real(rho_1s_pert2[0,2])]

with open("argv_entrees.txt", "w") as output:
    for row in argv_list:
        output.write(str(row) + "\n")

# Executing new_matrices_code.py for i=0 iteration.
#os.system("python new_matrices_code.py")
#exec(open("new_matrices_code.py").read())

# Unpacking the i=0 case
'''
lambda0_1s_1s_real_fits = fits.open("lambda0_1s_1s_z="+str(redshift_list[0])+"_real.fits")
lambda0_1s_exc_real_fits = fits.open("lambda0_1s_exc_z="+str(redshift_list[0])+"_real.fits")
lambda0_exc_1s_real_fits = fits.open("lambda0_exc_1s_z="+str(redshift_list[0])+"_real.fits")
lambda0_exc_exc_real_fits = fits.open("lambda0_exc_exc_z="+str(redshift_list[0])+"_real.fits")
    
lambda0_1s_1s_real = lambda0_1s_1s_real_fits[0].data     
lambda0_1s_exc_real = lambda0_1s_exc_real_fits[0].data     
lambda0_exc_1s_real = lambda0_exc_1s_real_fits[0].data         
lambda0_exc_exc_real = lambda0_exc_exc_real_fits[0].data 
        
lambda0_1s_1s_imag_fits = fits.open("lambda0_1s_1s_z="+str(redshift_list[0])+"_imag.fits")
lambda0_1s_exc_imag_fits = fits.open("lambda0_1s_exc_z="+str(redshift_list[0])+"_imag.fits")
lambda0_exc_1s_imag_fits = fits.open("lambda0_exc_1s_z="+str(redshift_list[0])+"_imag.fits")
lambda0_exc_exc_imag_fits = fits.open("lambda0_exc_exc_z="+str(redshift_list[0])+"_imag.fits")
    
lambda0_1s_1s_imag = lambda0_1s_1s_imag_fits[0].data     
lambda0_1s_exc_imag = lambda0_1s_exc_imag_fits[0].data     
lambda0_exc_1s_imag = lambda0_exc_1s_imag_fits[0].data         
lambda0_exc_exc_imag = lambda0_exc_exc_imag_fits[0].data 
    
lambda0_1s_1s = lambda0_1s_1s_real*complex(1,0) + lambda0_1s_1s_imag*complex(0,1)
lambda0_1s_exc = lambda0_1s_exc_real*complex(1,0) + lambda0_1s_exc_imag*complex(0,1)    
lambda0_exc_1s = lambda0_exc_1s_real*complex(1,0) + lambda0_exc_1s_imag*complex(0,1)
lambda0_exc_exc = lambda0_exc_exc_real*complex(1,0) + lambda0_exc_exc_imag*complex(0,1)    
                
source_1s_fits = fits.open("source_1s_unpert_z="+str(redshift_list[0])+".fits")
source_exc_fits = fits.open("source_exc_unpert_z="+str(redshift_list[0])+".fits")
        
source_1s_unpert[0] = source_1s_fits[0].data*complex(1,0)
source_exc_unpert[0] = source_exc_fits[0].data*complex(1,0)



# Calculating the derivative function and the eigenvalues 
A = lambda0_1s_1s - np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), lambda0_exc_1s))
A *= - 1/((1+redshift)*H_param)

eigenvalue_unpert[0] = np.linalg.eig(A)[0]

# Calculating rho_exc_unpert
rho_exc_unpert[0] = -np.dot( np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s, rho_1s_unpert[0]) )

# Calculating deriv_func_unpert
deriv_func_unpert[0] = np.dot(lambda0_1s_1s,rho_1s_unpert[0]) + np.dot(lambda0_1s_exc, rho_exc_unpert[0]) + source_1s_unpert[0]
deriv_func_unpert[0] *= -1/((1+redshift)*H_param)
'''

# Running code for i=1
'''
# Get values from CLASS for the initial conditions i=0,1

X_e = thermo_array[zline-1][3]
X_1s = 1-X_e

redshift = redshift_list[1]
T = 2.73*(1+redshift[1])
T_star = 0.0681 # Kelvin

# Calculate what the Hubble parameter based on Plack 2018 best fit values.

omega_R = 9.2364e-5
omega_M = 0.321
omega_L = 0.679
H0 = 67.3 # km/s/Mpc

H_param = H0* np.sqrt( omega_R*(1+redshift)**4 + omega_M*(1+redshift)**3 + omega_L )
H_param = H_param*(3.24078e-20) # Conversion to 1/sec.


# Calculating the initial density matrix components:
# rho_1s_unpert[0] = X_1s^(F=0) , rho_1s_unpert[1] = X_1s^(F=1)/ sqrt(3)

rho_1s_unpert[1,0] = X_1s/(1+ 3*np.exp(-T_star/T) )
rho_1s_unpert[1,1] = np.sqrt(3)*X_1s*np.exp(-T_star/T) / (1 + 3*np.exp(-T_star/T))
rho_1s_unpert[1,2] = 0


# Reshaping our array to be viewable on ds9
#rho_1s_unpert = rho_1s_unpert.reshape(3,1) 

#hdu_rho_1s_unpert = fits.PrimaryHDU(rho_1s_unpert )
#hdu_rho_1s_unpert.writeto("rho_1s_unpert_z="+str(redshift_list[0])+".fits", overwrite = True)


#rho_1s_unpert = rho_1s_unpert.reshape(3)

# Making rho_1s_unpert complex

rho_1s_unpert = rho_1s_unpert*complex(1,0)

# Saving list of values that will be used in new_matrices_code.py where the Lambda matrices and the source terms are calculated.
# The contents are the current redshift and the three density matrix components which are saved onto a text file.

argv_list = [redshift, np.real(rho_1s_unpert[1,0]), np.real(rho_1s_unpert[1,0]), np.real(rho_1s_unpert[1,0])]

with open("argv_entrees.txt", "w") as output:
    for row in argv_list:
        output.write(str(row) + "\n")

# Executing new_matrices_code.py for i=1 iteration.
exec(open("new_matrices_code.py").read())

# Unpacking the i=1 case

lambda0_1s_1s_real_fits = fits.open("lambda0_1s_1s_z="+str(redshift_list[1])+"_real.fits")
lambda0_1s_exc_real_fits = fits.open("lambda0_1s_exc_z="+str(redshift_list[1])+"_real.fits")
lambda0_exc_1s_real_fits = fits.open("lambda0_exc_1s_z="+str(redshift_list[1])+"_real.fits")
lambda0_exc_exc_real_fits = fits.open("lambda0_exc_exc_z="+str(redshift_list[1])+"_real.fits")
    
lambda0_1s_1s_real = lambda0_1s_1s_real_fits[0].data     
lambda0_1s_exc_real = lambda0_1s_exc_real_fits[0].data     
lambda0_exc_1s_real = lambda0_exc_1s_real_fits[0].data         
lambda0_exc_exc_real = lambda0_exc_exc_real_fits[0].data 
        
lambda0_1s_1s_imag_fits = fits.open("lambda0_1s_1s_z="+str(redshift_list[1])+"_imag.fits")
lambda0_1s_exc_imag_fits = fits.open("lambda0_1s_exc_z="+str(redshift_list[1])+"_imag.fits")
lambda0_exc_1s_imag_fits = fits.open("lambda0_exc_1s_z="+str(redshift_list[1])+"_imag.fits")
lambda0_exc_exc_imag_fits = fits.open("lambda0_exc_exc_z="+str(redshift_list[1])+"_imag.fits")
    
lambda0_1s_1s_imag = lambda0_1s_1s_imag_fits[0].data     
lambda0_1s_exc_imag = lambda0_1s_exc_imag_fits[0].data     
lambda0_exc_1s_imag = lambda0_exc_1s_imag_fits[0].data         
lambda0_exc_exc_imag = lambda0_exc_exc_imag_fits[0].data 
    
lambda0_1s_1s = lambda0_1s_1s_real*complex(1,0) + lambda0_1s_1s_imag*complex(0,1)
lambda0_1s_exc = lambda0_1s_exc_real*complex(1,0) + lambda0_1s_exc_imag*complex(0,1)    
lambda0_exc_1s = lambda0_exc_1s_real*complex(1,0) + lambda0_exc_1s_imag*complex(0,1)
lambda0_exc_exc = lambda0_exc_exc_real*complex(1,0) + lambda0_exc_exc_imag*complex(0,1)    
                
source_1s_fits = fits.open("source_1s_unpert_z="+str(redshift_list[1])+".fits")
source_exc_fits = fits.open("source_exc_unpert_z="+str(redshift_list[1])+".fits")
        
source_1s_unpert[1] = source_1s_fits[0].data*complex(1,0)
source_exc_unpert[1] = source_exc_fits[0].data*complex(1,0)

# Calculating the derivative function and the eigenvalues 
A = lambda0_1s_1s - np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), lambda0_exc_1s))
A *= - 1/((1+redshift)*H_param)

eigenvalue_unpert[1] = np.linalg.eig(A)[0]

# Calculating rho_exc_unpert
rho_exc_unpert[1] = -np.dot( np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s, rho_1s_unpert[0]) )

# Calculating deriv_func_unpert
deriv_func_unpert[1] = np.dot(lambda0_1s_1s,rho_1s_unpert[0]) + np.dot(lambda0_1s_exc, rho_exc_unpert[0]) + source_1s_unpert[0]
deriv_func_unpert[1] *= -1/((1+redshift)*H_param)
'''
'''
for i in range(len(redshift_list)):

    redshift = redshift_list[i]
    T = T0*(1+redshift)

    x0 = 0 # Lower limit guess (X_e=0)
    x1 = 1 # Upper limit guess (X_e=1)

    tol = 1e-10 # Tolerance

    while (np.abs(Saha_eqn(x0,T) - Saha_eqn(x1,T))>= tol ):
        x2 = (x0+x1)/2
        
        if Saha_eqn(x2,T) == 0:
            break
        elif Saha_eqn(x2,T)*Saha_eqn(x0,T) < 0:
            x1 = x2
        else:
            x0 = x2
   
    X_e = x2
    #print(X_e)
    X_1s_Saha[i] = 1-X_e
    
'''


# Calculating the Peebles equation


    
for i in range(len(redshift_list)):
    
    redshift = redshift_list[i]
    
    if i == 0:
        X_1s_Peebles[0] = 1-X_e_init    
    else:
        X_e = 1-X_1s_Peebles[i-1]
        
        X_1s_Peebles[i] = X_1s_Peebles[i-1] + z_index * Peebles_eqn(X_e,redshift_list[i-1])
        
        
        X_1s_deriv_Peebles[i-1] = (X_1s_Peebles[i]-X_1s_Peebles[i-1])/ z_index
    
    
 # Calculating the evolution of the density matrix   
    

for i in range(len(redshift_list)-1):

   
    # Determine the redshift and Hubble parameter H(z) at the ith redshift
    
    print("X_1s_Peebles: ", X_1s_Peebles[i])
    print("X_1s_CLASS: ", X_1s_CLASS[i])
    
    print(str(i)+"th iteration")

    redshift = redshift_list[i]

    H_param = gamma_100 * np.sqrt(omega_r*(1+redshift)**4 + omega_m*(1+redshift)**3  + omega_DE)
    
    # Run new_matrices_code.py to produce the Lambda matrices we need for our code
    
    #exec(open("new_matrices_code.py").read())
    #os.system("python new_matrices_code.py")
    if i > 244:
        #os.system("python -m cProfile -o z_=_" + str(redshift_list[i])+"_profile.txt new_matrices_code.py")
        pass
    
    # We wish to calculate the i+1 element using the forward Euler method.
    
    # Getting all the unperturbed Lambda matrices
        
    lambda0_1s_1s_real_fits = fits.open("lambda0_1s_1s_z="+str(redshift_list[i])+"_real.fits")
    lambda0_1s_exc_real_fits = fits.open("lambda0_1s_exc_z="+str(redshift_list[i])+"_real.fits")
    lambda0_exc_1s_real_fits = fits.open("lambda0_exc_1s_z="+str(redshift_list[i])+"_real.fits")
    lambda0_exc_exc_real_fits = fits.open("lambda0_exc_exc_z="+str(redshift_list[i])+"_real.fits")
    
    lambda0_1s_1s_real = lambda0_1s_1s_real_fits[0].data     
    lambda0_1s_exc_real = lambda0_1s_exc_real_fits[0].data     
    lambda0_exc_1s_real = lambda0_exc_1s_real_fits[0].data         
    lambda0_exc_exc_real = lambda0_exc_exc_real_fits[0].data 
        
    lambda0_1s_1s_imag_fits = fits.open("lambda0_1s_1s_z="+str(redshift_list[i])+"_imag.fits")
    lambda0_1s_exc_imag_fits = fits.open("lambda0_1s_exc_z="+str(redshift_list[i])+"_imag.fits")
    lambda0_exc_1s_imag_fits = fits.open("lambda0_exc_1s_z="+str(redshift_list[i])+"_imag.fits")
    lambda0_exc_exc_imag_fits = fits.open("lambda0_exc_exc_z="+str(redshift_list[i])+"_imag.fits")
    
    lambda0_1s_1s_imag = lambda0_1s_1s_imag_fits[0].data     
    lambda0_1s_exc_imag = lambda0_1s_exc_imag_fits[0].data     
    lambda0_exc_1s_imag = lambda0_exc_1s_imag_fits[0].data         
    lambda0_exc_exc_imag = lambda0_exc_exc_imag_fits[0].data 
    
    lambda0_1s_1s = lambda0_1s_1s_real*complex(1,0) + lambda0_1s_1s_imag*complex(0,1)
    lambda0_1s_exc = lambda0_1s_exc_real*complex(1,0) + lambda0_1s_exc_imag*complex(0,1)    
    lambda0_exc_1s = lambda0_exc_1s_real*complex(1,0) + lambda0_exc_1s_imag*complex(0,1)
    lambda0_exc_exc = lambda0_exc_exc_real*complex(1,0) + lambda0_exc_exc_imag*complex(0,1) 
        
       
    
    # Adding new elements to my source function
                
    source_1s_fits = fits.open("source_1s_unpert_z="+str(redshift_list[i])+".fits")
    source_exc_fits = fits.open("source_exc_unpert_z="+str(redshift_list[i])+".fits")
        
    source_1s_unpert[i] = source_1s_fits[0].data*complex(1,0)
    source_exc_unpert[i] = source_exc_fits[0].data*complex(1,0)
    
    source_1s_unpert[i] = 0
    #source_exc_unpert[i] = 0       


    # Getting all the perturbed LO matrices

    L0_1s_1s_real_fits = fits.open("L0_1s_1s_z="+str(redshift_list[i])+"_real.fits")
    L0_1s_exc_real_fits = fits.open("L0_1s_exc_z="+str(redshift_list[i])+"_real.fits")
    L0_exc_1s_real_fits = fits.open("L0_exc_1s_z="+str(redshift_list[i])+"_real.fits")
    L0_exc_exc_real_fits = fits.open("L0_exc_exc_z="+str(redshift_list[i])+"_real.fits")
    
    L0_1s_1s = L0_1s_1s_real_fits[0].data*complex(1,0)
    L0_1s_exc = L0_1s_exc_real_fits[0].data*complex(1,0)     
    L0_exc_1s = L0_exc_1s_real_fits[0].data*complex(1,0)         
    L0_exc_exc = L0_exc_exc_real_fits[0].data*complex(1,0) 

    source_1s_pert0_fits = fits.open("source_1s_pert_0_z="+str(redshift_list[i])+".fits")
    source_exc_pert0_fits = fits.open("source_exc_pert_0_z="+str(redshift_list[i])+".fits")
        
    source_1s_pert0[i] = source_1s_pert0_fits[0].data*complex(1,0)
    source_exc_pert0[i] = source_exc_pert0_fits[0].data*complex(1,0) 
    

    # Getting all the perturbed L2 matrices

    L2_1s_1s_real_fits = fits.open("L2_1s_1s_z="+str(redshift_list[i])+"_real.fits")
    L2_1s_exc_real_fits = fits.open("L2_1s_exc_z="+str(redshift_list[i])+"_real.fits")
    L2_exc_1s_real_fits = fits.open("L2_exc_1s_z="+str(redshift_list[i])+"_real.fits")
    L2_exc_exc_real_fits = fits.open("L2_exc_exc_z="+str(redshift_list[i])+"_real.fits")
    
    L2_1s_1s = L2_1s_1s_real_fits[0].data*complex(1,0)
    L2_1s_exc = L2_1s_exc_real_fits[0].data*complex(1,0)     
    L2_exc_1s = L2_exc_1s_real_fits[0].data*complex(1,0)         
    L2_exc_exc = L2_exc_exc_real_fits[0].data*complex(1,0) 

    source_1s_pert2_fits = fits.open("source_1s_pert_2_z="+str(redshift_list[i])+".fits")
    source_exc_pert2_fits = fits.open("source_exc_pert_2_z="+str(redshift_list[i])+".fits")
        
    source_1s_pert2[i] = source_1s_pert2_fits[0].data*complex(1,0)
    source_exc_pert2[i] = source_exc_pert2_fits[0].data*complex(1,0) 
    
    
    # We need to define new matrices that are the coefficient matrices for our root solving method.
    # Our equation: rho_1s(n+1) - hf(rho_1s(n+1),z_n+1) - rho_1s(n) = 0
    # We now label our coefficients
        
    A = lambda0_1s_1s        
    A += -np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), lambda0_exc_1s))
    A *= -1/((1+redshift)*H_param)
    print(" ")
    print("A matrix at z="+str(redshift_list[i]))
    print(A)
        
    B = source_1s_unpert[i] - np.dot( lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert[i]))
    B *= -1/((1+redshift)*H_param)
    print(" ")
    print("B matrix at z="+str(redshift_list[i]))
    print(B)
    
    # Calculating next rho_1s_unpert = (1- deltaz*A)^-1( rho_1s_prev + deltaz*B)

    rho_1s_prev = rho_1s_unpert[i] # previous rho_1s
    
    '''    
    matrix = np.identity(3) - z_index*A
    rho_1s_unpert[i+1] = np.dot( np.linalg.inv(matrix), rho_1s_prev)
    rho_1s_unpert[i+1] += np.dot( np.linalg.inv(matrix), z_index*B)         
     '''
    
    rho_1s_unpert[i+1] = rho_1s_unpert[i] + z_index*(np.dot(A, rho_1s_unpert[i]) + B)
    
                             
    # Calculating the previous rho_exc_unpert
     
    rho_exc_unpert[i] = -np.dot( np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s, rho_1s_unpert[i]) ) - np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_unpert[i])    
    
    # Calculating the population of X_2s and X_2p
    X_2s[i] = rho_exc_unpert[i,0] + np.sqrt(3)*rho_exc_unpert[i,1]
    X_2p[i] = rho_exc_unpert[i,3] + np.sqrt(3)*rho_exc_unpert[i,4] + np.sqrt(3)*rho_exc_unpert[i,6] + np.sqrt(5)*rho_exc_unpert[i,7]

    # Calculating the population of X_3s, X_3p, and X_3d
    X_3s[i] = rho_exc_unpert[i,12] + np.sqrt(3)*rho_exc_unpert[i,13]
    X_3p[i] = rho_exc_unpert[i,15] + np.sqrt(3)*rho_exc_unpert[i,16] + np.sqrt(3)*rho_exc_unpert[i,18] + np.sqrt(5)*rho_exc_unpert[i,19]   
    X_3d[i] = np.sqrt(3)*rho_exc_unpert[i,24] + np.sqrt(5)*rho_exc_unpert[i,25] + np.sqrt(5)*rho_exc_unpert[i,30] + np.sqrt(7)*rho_exc_unpert[i,31]  
    
    # Calculating the population of X_4s, X_4p, X_4d, and X_4f
    X_4s[i] = rho_exc_unpert[i,36] + np.sqrt(3)*rho_exc_unpert[i,37]
    X_4p[i] = rho_exc_unpert[i,39] + np.sqrt(3)*rho_exc_unpert[i,40] + np.sqrt(3)*rho_exc_unpert[i,42] + np.sqrt(5)*rho_exc_unpert[i,43]
    X_4d[i] = np.sqrt(3)*rho_exc_unpert[i,48] + np.sqrt(5)*rho_exc_unpert[i,49] + np.sqrt(5)*rho_exc_unpert[i,54] + np.sqrt(7)*rho_exc_unpert[i,55]
    X_4f[i] = np.sqrt(7)*rho_exc_unpert[i,60] + np.sqrt(9)*rho_exc_unpert[i,61] + np.sqrt(9)*rho_exc_unpert[i,66] + np.sqrt(11)*rho_exc_unpert[i,67]                             
    # Calculating deriv_func_unpert         
                
    deriv_func_unpert[i] = np.dot( lambda0_1s_1s, rho_1s_unpert[i]) + np.dot(lambda0_1s_exc, rho_exc_unpert[i]) + source_1s_unpert[i]
    deriv_func_unpert[i] *= -1/((1+redshift)*H_param)
    
    
    eigenvalue_unpert[i] = np.linalg.eig(A)[0]




              
    print(" ")    
    print("rho_1s_unpert at z="+str(redshift_list[i]))
    print(rho_1s_unpert[i])  
    
               
    print(" ")    
    print("rho_exc_unpert at z="+str(redshift_list[i]))
    print(rho_exc_unpert[i])           
    
    print(" ")
    print("source_1s_unpert at z="+str(redshift_list[i]))
    print(source_1s_unpert[i])

    print(" ")
    print("source_exc_unpert at z="+str(redshift_list[i]))
    print(source_exc_unpert[i])
    
    '''
    We now compute the perturbed evolution of the density matrix.
    '''
    '''
    # Computing the perturbed K=0 rho_exc
    
    rho_exc_pert0[i] = np.dot(np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L0_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), lambda0_exc_1s))), rho_1s_unpert[i])
    rho_exc_pert0[i] += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L0_exc_1s, rho_1s_unpert[i]))
    rho_exc_pert0[i] += np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L0_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert[i])))
    rho_exc_pert0[i] += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(lambda0_exc_1s, rho_1s_pert0[i]))
    rho_exc_pert0[i] += - np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_pert0[i])
    
    rho_1s_pert0_prev = rho_1s_pert0[i]
    
    B_pert0 = L0_1s_1s - np.dot(L0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), lambda0_exc_1s))
    B_pert0 += np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L0_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), lambda0_exc_1s))))
    B_pert0 += - np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), L0_exc_1s))
    B_pert0 *= - 1/(H_param*(1+redshift))
    
    C_pert0 = source_1s_pert0[i] - np.dot(L0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert[i]))
    C_pert0 += np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L0_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert[i]))))
    C_pert0 += - np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_pert0[i]))
    C_pert0 *= - 1/(H_param*(1+redshift))   
    

    rho_1s_pert0[i+1] = rho_1s_pert0[i] + z_index*np.dot(A, rho_1s_pert0[i])
    rho_1s_pert0[i+1] += z_index*np.dot(B_pert0, rho_1s_unpert[i])
    rho_1s_pert0[i+1] += z_index*C_pert0  
    '''

    '''
    argv_pert0_list = [redshift, np.real(rho_1s_pert0[i+1,0]), np.real(rho_1s_pert0[i+1,1]), np.real(rho_1s_pert0[i+1,2])]
    
    with open("argv_pert0_entrees.txt", "w") as output:
        for row in argv_pert0_list:
            output.write(str(row) + "\n")   
    '''

              
    print(" ")    
    print("rho_1s_pert0 at z="+str(redshift_list[i]))
    print(rho_1s_pert0[i])  
    
               
    print(" ")    
    print("rho_exc_pert0 at z="+str(redshift_list[i]))
    print(rho_exc_pert0[i])           
    
    print(" ")
    print("source_1s_pert0 at z="+str(redshift_list[i]))
    print(source_1s_pert0[i])

    print(" ")
    print("source_exc_pert0 at z="+str(redshift_list[i]))
    print(source_exc_pert0[i])

    
    '''
    We now compute the perturbed evolution of the density matrix.
    '''
    '''
    # Computing the perturbed K = 2 rho_exc
    
    rho_exc_pert2[i] = np.dot(np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L2_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), lambda0_exc_1s))), rho_1s_unpert[i])
    rho_exc_pert2[i] += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L2_exc_1s, rho_1s_unpert[i]))
    rho_exc_pert2[i] += np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L2_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert[i])))
    rho_exc_pert2[i] += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(lambda0_exc_1s, rho_1s_pert2[i]))
    rho_exc_pert2[i] += - np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_pert2[i])
    
    rho_1s_pert2_prev = rho_1s_pert2[i]
    
    
    B_pert2 = L2_1s_1s - np.dot(L2_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), lambda0_exc_1s))
    B_pert2 += np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L2_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), lambda0_exc_1s))))
    B_pert2 += - np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), L2_exc_1s))
    B_pert2 *= - 1/(H_param*(1+redshift))
    
    C_pert2 = source_1s_pert2[i] - np.dot(L2_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert[i]))
    C_pert2 += np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L2_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert[i]))))
    C_pert2 += - np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_pert2[i]))
    C_pert2 *= - 1/(H_param*(1+redshift))   
    

    rho_1s_pert2[i+1] = rho_1s_pert2[i] + z_index*np.dot(A, rho_1s_pert2[i])
    rho_1s_pert2[i+1] += z_index*np.dot(B_pert2, rho_1s_unpert[i])
    rho_1s_pert2[i+1] += z_index*C_pert2  
    '''
    '''
    B_pert2 = L2_1s_1s - np.dot(L2_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), lambda0_exc_1s))
    B_pert2 += np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L2_exc_exc, lambda0_exc_1s)))
    B_pert2 += - np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), L2_exc_1s))
    B_pert2 *= - 1/(H_param*(1+redshift))
    
    C_pert2 = source_1s_pert2[i] - np.dot(L2_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert[i]))
    C_pert2 += np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L2_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert[i]))))
    C_pert2 += - np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_pert2[i]))
    C_pert2 *= - 1/(H_param*(1+redshift))    

    rho_1s_pert2[i+1] = rho_1s_pert2[i] + z_index*np.dot(A, rho_1s_pert2[i])
    rho_1s_pert2[i+1] += z_index*np.dot(B_pert2, rho_1s_unpert[i])
    rho_1s_pert2[i+1] += z_index*C_pert2
    '''  



                 
    print(" ")    
    print("rho_1s_pert2 at z="+str(redshift_list[i]))
    print(rho_1s_pert2[i])  
    
               
    print(" ")    
    print("rho_exc_pert2 at z="+str(redshift_list[i]))
    print(rho_exc_pert2[i])           
    
    print(" ")
    print("source_1s_pert2 at z="+str(redshift_list[i]))
    print(source_1s_pert2[i])

    print(" ")
    print("source_exc_pert2 at z="+str(redshift_list[i]))
    print(source_exc_pert2[i])     

    # Opening Lambda functions
    
    redshift  = redshift_list[i+1]
    argv_list = [redshift, np.real(rho_1s_unpert[i+1,0]), np.real(rho_1s_unpert[i+1,1]), np.real(rho_1s_pert2[i+1,2])]
    
    print("argv_list paramaters:", argv_list)
    #argv_list = [redshift, np.real(rho_1s_unpert[i+1,0]), np.real(rho_1s_unpert[i+1,1]), 0]  
 
    with open("argv_entrees.txt", "w") as output:
        for row in argv_list: 
            output.write(str(row) + "\n")
      
               
    '''
        
    # We compute the perturbed terms for Kr=0
    
    L0_1s_1s_fits = fits.open("L0_1s_1s_z="+str(redshift_list[i])+".fits")
    L0_1s_exc_fits = fits.open("L0_1s_exc_z="+str(redshift_list[i])+".fits")
    L0_exc_1s_fits = fits.open("L0_exc_1s_z="+str(redshift_list[i])+".fits")
    L0_exc_exc_fits = fits.open("L0_exc_exc_z="+str(redshift_list[i])+".fits")
    
    L0_1s_1s = L0_1s_1s_fits[0].data     
    L0_1s_exc = L0_1s_exc_fits[0].data     
    L0_exc_1s = L0_exc_1s_fits[0].data         
    L0_exc_exc = L0_exc_exc_fits[0].data 
        
    source_1s_pert0_fits = fits.open("source_1s_pert_0_z="+str(redshift_list[i])+".fits")
    source_exc_pert0_fits = fits.open("source_exc_pert_0_z="+str(redshift_list[i])+".fits")
        
    source_1s_pert0 = source_1s_pert0_fits[0].data
    source_exc_pert0 = source_exc_pert0_fits[0].data    
    
    rho_exc_pert0 = np.dot(np.linalg.inv(lambda0_exc_exc), np.dot( L0_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s,rho_1s_unpert))))
    rho_exc_pert0 += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L0_exc_1s,rho_1s_unpert) )
    rho_exc_pert0 += np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L0_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert)))
    rho_exc_pert0 += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(lambda0_exc_1s, rho_1s_pert0) )
    rho_exc_pert0 += - np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_pert0)
    

        
    if i == 0:

        rho_1s_pert0 = rho_1s_pert0.reshape(len(rho_1s_pert0),1)        
                    
        hdu_rho_exc_pert0 = fits.PrimaryHDU(rho_exc_pert0) 
        hdu_rho_exc_pert0.writeto("rho_exc_pert0_z="+str(redshift_list[0])+".fits", overwrite = True)    

        rho_1s_pert0 = rho_1s_pert0.reshape(len(rho_1s_pert0))    
    else:
    
        # We label the derivative function
        
        deriv_func_pert0 = np.dot(L0_1s_1s, rho_1s_unpert) + np.dot(lambda0_1s_1s, rho_1s_pert0) 
        deriv_func_pert0 += np.dot(L0_1s_exc, rho_exc_unpert) + np.dot(lambda0_1s_exc, rho_exc_pert0)
        deriv_func_pert0 += source_1s_pert0
        deriv_func_pert0 *= -1/( (1+redshift)*H_param )
        
        rho_1s_pert0 += z_index*deriv_func_pert0
        
        rho_exc_pert0 = np.dot(np.linalg.inv(lambda0_exc_exc), np.dot( L0_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s,rho_1s_unpert))))
        rho_exc_pert0 += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L0_exc_1s,rho_1s_unpert) )
        rho_exc_pert0 += np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L0_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert)))
        rho_exc_pert0 += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(lambda0_exc_1s, rho_1s_pert0) )
        rho_exc_pert0 += - np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_pert0)

        deriv_func_pert0 = deriv_func_pert0.reshape( len(deriv_func_pert0),1)
        rho_1s_pert0 = rho_1s_pert0.reshape(len(rho_1s_pert0),1)        
        rho_exc_pert0 = rho_exc_pert0.reshape(len(rho_exc_pert0),1)    
                
        hdu_deriv_func_pert0 = fits.PrimaryHDU(deriv_func_pert0)
        hdu_deriv_func_pert0.writeto("deriv_func_pert0_z="+str(redshift_list[i])+".fits", overwrite = True)            

        hdu_rho_exc_pert0 = fits.PrimaryHDU(rho_exc_pert0) 
        hdu_rho_exc_pert0.writeto("rho_exc_pert0_z="+str(redshift_list[i])+".fits", overwrite = True)            

        hdu_rho_1s_pert0 = fits.PrimaryHDU(rho_1s_pert0 )
        hdu_rho_1s_pert0.writeto("rho_1s_pert0_z="+str(redshift_list[i])+".fits", overwrite = True)    

        deriv_func_pert0 = deriv_func_pert0.reshape( len(deriv_func_pert0))
        rho_1s_pert0 = rho_1s_pert0.reshape(len(rho_1s_pert0))        
        rho_exc_pert0 = rho_exc_pert0.reshape(len(rho_exc_pert0))
        
    # We compute the perturbed terms for Kr=2
    
    L2_1s_1s_fits = fits.open("L2_1s_1s_z="+str(redshift_list[i])+".fits")
    L2_1s_exc_fits = fits.open("L2_1s_exc_z="+str(redshift_list[i])+".fits")
    L2_exc_1s_fits = fits.open("L2_exc_1s_z="+str(redshift_list[i])+".fits")
    L2_exc_exc_fits = fits.open("L2_exc_exc_z="+str(redshift_list[i])+".fits")
    
    L2_1s_1s = L2_1s_1s_fits[0].data     
    L2_1s_exc = L2_1s_exc_fits[0].data     
    L2_exc_1s = L2_exc_1s_fits[0].data         
    L2_exc_exc = L2_exc_exc_fits[0].data 
        
    source_1s_pert2_fits = fits.open("source_1s_pert_2_z="+str(redshift_list[i])+".fits")
    source_exc_pert2_fits = fits.open("source_exc_pert_2_z="+str(redshift_list[i])+".fits")
        
    source_1s_pert2 = source_1s_pert2_fits[0].data
    source_exc_pert2 = source_exc_pert2_fits[0].data    
    
    rho_exc_pert2 = np.dot(np.linalg.inv(lambda0_exc_exc), np.dot( L2_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s,rho_1s_unpert))))
    rho_exc_pert2 += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L2_exc_1s,rho_1s_unpert) )
    rho_exc_pert2 += np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L2_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert)))
    rho_exc_pert2 += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(lambda0_exc_1s, rho_1s_pert2) )
    rho_exc_pert2 += - np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_pert2)
        
    if i == 0:

        rho_1s_pert2 = rho_1s_pert2.reshape(len(rho_1s_pert2),1)        
                    
        hdu_rho_exc_pert2 = fits.PrimaryHDU(rho_exc_pert2) 
        hdu_rho_exc_pert2.writeto("rho_exc_pert2_z="+str(redshift_list[0])+".fits", overwrite = True)    

        rho_1s_pert2 = rho_1s_pert2.reshape(len(rho_1s_pert2))
            
    else:
    
        # We label the derivative function
        
        deriv_func_pert2 = np.dot(L2_1s_1s, rho_1s_unpert) + np.dot(lambda0_1s_1s, rho_1s_pert2) 
        deriv_func_pert2 += np.dot(L2_1s_exc, rho_exc_unpert) + np.dot(lambda0_1s_exc, rho_exc_pert2)
        deriv_func_pert2 += source_1s_pert2
        deriv_func_pert2 *= -1/( (1+redshift)*H_param )
        
        rho_1s_pert2 += z_index*deriv_func_pert2
        
        rho_exc_pert2 = np.dot(np.linalg.inv(lambda0_exc_exc), np.dot( L2_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s,rho_1s_unpert))))
        rho_exc_pert2 += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L2_exc_1s,rho_1s_unpert) )
        rho_exc_pert2 += np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(L2_exc_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert)))
        rho_exc_pert2 += - np.dot(np.linalg.inv(lambda0_exc_exc), np.dot(lambda0_exc_1s, rho_1s_pert2) )
        rho_exc_pert2 += - np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_pert2)

        deriv_func_pert2 = deriv_func_pert2.reshape( len(deriv_func_pert2),1)
        rho_1s_pert2 = rho_1s_pert2.reshape(len(rho_1s_pert2),1)        
        rho_exc_pert2 = rho_exc_pert2.reshape(len(rho_exc_pert2),1)                    
                        
        hdu_deriv_func_pert2 = fits.PrimaryHDU(deriv_func_pert2)
        hdu_deriv_func_pert2.writeto("deriv_func_pert2_z="+str(redshift_list[i])+".fits", overwrite = True)            

        hdu_rho_exc_pert2 = fits.PrimaryHDU(rho_exc_pert2) 
        hdu_rho_exc_pert2.writeto("rho_exc_pert2_z="+str(redshift_list[i])+".fits", overwrite = True)            

        hdu_rho_1s_pert2 = fits.PrimaryHDU(rho_1s_pert2 )
        hdu_rho_1s_pert2.writeto("rho_1s_pert2_z="+str(redshift_list[i])+".fits", overwrite = True)    

        deriv_func_pert2 = deriv_func_pert2.reshape( len(deriv_func_pert2))
        rho_1s_pert2 = rho_1s_pert2.reshape(len(rho_1s_pert2))        
        rho_exc_pert2 = rho_exc_pert2.reshape(len(rho_exc_pert2))        

    '''    

# Calculating the total X_1s and its derivative

X_1s_total = rho_1s_unpert[:,0] + np.sqrt(3)*rho_1s_unpert[:,1]
X_1s_deriv = np.zeros(len(redshift_list)-1)

#Calculating the RHS from the Saha eqn
for i in range(len(redshift_list)):

    X_e = 1-X_1s_total[i]        
    T = T0*(1+redshift_list[i])
        
    Saha_2s[i] = Saha_eqn(2,X_e, T)
    Saha_2p[i] = 3*Saha_eqn(2, X_e, T)

    Saha_3s[i] = Saha_eqn(3,X_e, T)
    Saha_3p[i] = 3*Saha_eqn(3, X_e, T)   
    Saha_3d[i] = 5*Saha_eqn(3, X_e, T)   

    Saha_4s[i] = Saha_eqn(4,X_e, T)
    Saha_4p[i] = 3*Saha_eqn(4, X_e, T)   
    Saha_4d[i] = 5*Saha_eqn(4, X_e, T)     
    Saha_4f[i] = 7*Saha_eqn(4, X_e, T)     
# Calculating the ratios in relative to the Saha equation

ratio_2s = X_2s/ Saha_2s
ratio_2p = X_2p/ Saha_2p

ratio_3s = X_3s/ Saha_3s
ratio_3p = X_3p/ Saha_3p
ratio_3d = X_3d/ Saha_3d

ratio_4s = X_4s/ Saha_4s
ratio_4p = X_4p/ Saha_4p
ratio_4d = X_4d/ Saha_4d
ratio_4f = X_4f/ Saha_4f

for i in range(len(redshift_list)-1):

    X_1s_deriv[i] = (X_1s_total[i+1] - X_1s_total[i])/ z_index
    
# Note that X_1s = sqrt(2*F+1) * rho(F,F^')          

plt.figure(1)
plt.plot(redshift_list, rho_1s_unpert[:,0], label = "F=0")    
plt.plot(redshift_list, np.sqrt(3)*rho_1s_unpert[:,1], label = "F=1")
plt.plot(redshift_list, X_1s_total, label = "total")
plt.plot(redshift_list, X_1s_Saha, label = "Saha Equation")
plt.plot(redshift_list, X_1s_Peebles, label = "Peebles Equation")
plt.plot(redshift_list, X_1s_CLASS, label = "CLASS")
plt.xlabel("Redshift")
plt.ylabel("X_1s")
plt.legend()


plt.figure(2)
plt.plot(redshift_list, X_2s, label = "X_2s Code")
plt.plot(redshift_list, X_2p, label = "X_2p Code")
plt.plot(redshift_list, X_3s, label = "X_3s Code")
plt.plot(redshift_list, X_3p, label = "X_3p Code")
plt.plot(redshift_list, X_3d, label = "X_3d Code")
plt.plot(redshift_list, X_4s, label = "X_4s Code")
plt.plot(redshift_list, X_4p, label = "X_4p Code")
plt.plot(redshift_list, X_4d, label = "X_4d Code")
plt.plot(redshift_list, X_4f, label = "X_4f Code")


plt.plot(redshift_list, Saha_2s, label = "RHS X_2s Saha Equation")
plt.plot(redshift_list, Saha_2p, label = "RHS X_2p Saha Equation")
plt.plot(redshift_list, Saha_3s, label = "RHS X_3s Saha Equation")
plt.plot(redshift_list, Saha_3p, label = "RHS X_3p Saha Equation")
plt.plot(redshift_list, Saha_3d, label = "RHS X_3d Saha Equation")
plt.plot(redshift_list, Saha_4s, label = "RHS X_4s Saha Equation")
plt.plot(redshift_list, Saha_4p, label = "RHS X_4p Saha Equation")
plt.plot(redshift_list, Saha_4d, label = "RHS X_4d Saha Equation")
plt.plot(redshift_list, Saha_4f, label = "RHS X_4f Saha Equation")

plt.yscale("log")
plt.xlabel("Redshift")
plt.ylabel("X_nl_Saha")
plt.legend()


plt.figure(3)
plt.plot(redshift_list, ratio_2s, label = "X_2s/X_2s_Saha Code")
plt.plot(redshift_list, ratio_2p, label = "X_2p/X_2p_Saha Code")
plt.plot(redshift_list, ratio_3s, label = "X_3s/X_3s_Saha Code")
plt.plot(redshift_list, ratio_3p, label = "X_3p/X_3p_Saha Code")
plt.plot(redshift_list, ratio_3d, label = "X_3d/X_3d_Saha Code")
plt.plot(redshift_list, ratio_4s, label = "X_4s/X_4s_Saha Code")
plt.plot(redshift_list, ratio_4p, label = "X_4p/X_4p_Saha Code")
plt.plot(redshift_list, ratio_4d, label = "X_4d/X_4d_Saha Code")
plt.plot(redshift_list, ratio_4f, label = "X_4f/X_4f_Saha Code")

plt.xlabel("Redshift")
plt.ylabel("X_nl / X_nl_Saha")
plt.legend()


plt.figure(4)

plt.plot(redshift_list, X_2p/(3*X_2s), label = "X_2p/3X_2s")
plt.plot(redshift_list, X_3p/(3*X_3s), label = "X_3p/3X_3s")
plt.plot(redshift_list, X_3d/(5*X_3s), label = "X_3d/5X_3s")
plt.plot(redshift_list, 3*X_3d/(5*X_3p), label = "X_3p/(5/3)*X_3p")
plt.plot(redshift_list, X_4p/(3*X_4s), label = "X_4p/3X_4s")
plt.plot(redshift_list, X_4d/(5*X_4s), label = "X_4d/5X_4s")
plt.plot(redshift_list, X_4f/(7*X_4s), label = "X_4f/7X_4s")
plt.plot(redshift_list, 3*X_4d/(5*X_4p), label = "X_4d/(5/3)*X_4p")
plt.plot(redshift_list, 3*X_4f/(7*X_4p), label = "X_4f/(7/3)*X_4p")
plt.plot(redshift_list, 5*X_4f/(7*X_4d), label = "X_4f/(7/5)*X_4p")
plt.xlabel("Redshift")
plt.ylabel("Ratios of Populations")
plt.legend()


'''
plt.title("dX_1s/dz")
plt.plot(redshift_list[:49], X_1s_deriv, label = "From Code")
plt.plot(redshift_list[:49], X_1s_deriv_Peebles, label = "Peebles Equation")
plt.plot(redshift_list[:49], X_1s_deriv_CLASS, label = "From CLASS")
plt.xlabel("Redshift")
plt.ylabel("dX_1s/dz")
plt.legend()
plt.show()
'''

plt.figure(5)
plt.title("Source 1s Unperturbed (K=0)")
plt.plot(redshift_list, source_1s_unpert[:,0], label = "1s_1/2 F=F'=0")
plt.plot(redshift_list, source_1s_unpert[:,1], label = "1s_1/2 F=F'=1")
plt.xlabel("Redshift")
plt.ylabel("Magnitude (s^{-1})")
plt.legend()

plt.figure(6)
plt.title("Source Exc Unperturbed (K=0)")
plt.plot(redshift_list, source_exc_unpert[:,0], label = "2s_1/2 F=F'=0")
plt.plot(redshift_list,  source_exc_unpert[:,1], label = "2s_1/2 F=F'=1")
plt.plot(redshift_list,  source_exc_unpert[:,3], label = "2p_1/2 F=F'=0")
plt.plot(redshift_list,  source_exc_unpert[:,4], label = "2p_1/2 F=F'=1")
plt.plot(redshift_list,  source_exc_unpert[:,6], label = "2p_3/2 F=F'=1")
plt.plot(redshift_list,  source_exc_unpert[:,7], label = "2p_3/2 F=F'=2")
plt.xlabel("Redshift")
plt.ylabel("Magnitude (s^{-1})")
plt.legend()

plt.show()

'''
plt.title("A matrix Eigenvalue")
plt.plot(redshift_list, eigenvalue_unpert[:,0], label = "F=F'=0")
plt.plot(redshift_list, eigenvalue_unpert[:,1], label = "F=F'=1")
plt.legend()
plt.show()


# d rho_1s/dz = F(rho_1s,z)
plt.title("F(rho_1s,z)")
plt.plot(redshift_list[1:len(redshift_list)-1], deriv_func_Fis0, label = "F=F'=0")
plt.plot(redshift_list[1:len(redshift_list)-1], deriv_func_Fis1, label = "F=F'=1")
plt.legend()
plt.show()
'''



    
     
