import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import fractions
import time
from astropy.io import fits
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
from scipy.special import genlaguerre, gamma, hyp2f1, factorial, gammaln, gamma, zeta

# First we wish to input the values from CLASS to compute the Lambda matrices

'''
We want to create a module that takes a text file of values given by class as a function of redshift.
For example, we first want take in a redshift from a compilation of text files. 

'''
# Constants to be used in the cgs unit system (centimeters-grams-seconds)

c = 3e10 # Speed of light ( cm/s )
h = 6.626e-27 # Planck constant ( erg*s or cm^2 g / s )
hbar = h / (2*np.pi) # Reduced Planck constant in erg*s
e0 = 4.8032e-10 # One unit of electric charge   ( cm^3/2 * g^1/2 / s )
m_electron = 9.10938e-28 # Electron rest mass ( grams )
eV_to_ergs = 1.60218e-12 # eV to ergs conversion 
kB = 1.3807e-16 # Boltzmann constant (cm^2 g / s^2 K)
baryon_to_photon_ratio = 5.9e-10 # Number of Baryons per photons
He_abund = 0.24672 # Primordial Helium Abundance

mag_field = 1e-12 # Magnetic Field Strength (Gauss)


# Quantities calculated using the above constants of nature

Bohr_radius = hbar**2 / (m_electron * e0**2 ) # Bohr radius ( cm )
larmor_freq = 1.3996e6 * mag_field # Lahmor frequency ( s^-1 )
ion_energy = m_electron*e0**4/(2*hbar**2) 

# Quantum numbers

S = 0.5 # Electron spin quantum number
I = 0.5 # Nuclear spin quantum number

# Values considered when summing over quantum numbers to determine 

                                                                                                          
numN = 4 # Largest N value considered. Will go from 1 to Nmax.
numL = numN # Largest L value considered.
numJ = 2 # Number of allowed J values other than the L=0 case which there is only one.
numK = 2 # Sum from 0 to 2
numF = 2 # Number of allowed F values given a value of J.

# We want to integrate across a ide range of energies. Here is where we set bounds on what
# we are actually integrating across.

numE = 1000 # Pick a number of elements to integrate across
Emax = 100*eV_to_ergs #Pick a maximum energy for free energy of electron


# We want to develop a program where we can index our matrices correctly.

# Creating a text file to put printed output

redshift = np.float(sys.argv[1])
wave_num = np.float(sys.argv[2])

print(redshift)

thermo_array = np.loadtxt(open("thermo.dat"))
tk_array = np.loadtxt(open("tk_z="+str(redshift)+".dat"))

found_kline = False # Boolian for finding line in CLASS file
found_zline = False # Boolian for finding line in redshift file

'''
We wish to calculate find the wave number line that is closest to the entered value. This is 
set by the value of k_error which we set to 1e-3.
'''


print(redshift)
print(wave_num)



while found_kline == False:
	for line in range(len(tk_array)):
	
		#print(line)
	
		if tk_array[line][0] == wave_num :
			print("The wave number in the file is "+str(tk_array[line][0]))
			print("The wave number is in line "+str(line))
			
			kline = line
			found_kline = True
	
	if found_kline == False:		
		wave_num = input("Wave number not found. Please input a new wave number: ")
	

z_error = 1e-3

print(redshift)
while found_zline == False:
	for line in range(len(thermo_array)):
		
		if thermo_array[line][1] == redshift:
			print("The redshift in the file is "+str(thermo_array[line][1]))
			print("The redshift is in line "+str(line))
			
			zline = line
			found_zline = True
	
	if found_zline == False:		
		redshift = input("Redshift not found. Please input a new redshift: ")


# If synchronous gauge is chosen
	
wave_num = tk_array[kline][0]
delta_g = tk_array[kline][1]
delta_b = tk_array[kline][2]
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



# If newtonian gauge is chosen

'''
wave_num = tk_array[kline][0] 
delta_g = tk_array[kline][1]
phi = tk_array[kline][7]
psi = tk_array[kline][8]
phi_dot = tk_array[kline][9]
t_g = tk_array[kline][12]
t_b = tk_array[kline][13]
shear_g = tk_array[kline][16]
pol_0 = tk_array[kline][17]
pol_2 = tk_array[kline][18]
shear_33 = 0

print(wave_num)
print(delta_g)
print(phi)
print(psi)
print(phi_dot)
print(t_g)
print(shear_g)
print(pol_0)
print(pol_2)

'''

Theta_0 = delta_g
Theta_2 = 2*shear_g

# Values from thermo_array
	
scale_factor = thermo_array[zline][0]
redshift = thermo_array[zline][1]
T = 2.73*(1+redshift)
conform_time = thermo_array[zline][2]
x_e = thermo_array[zline][3]




conform_time = conform_time * (3.0824e24) / c # Conversion to seconds	
				
# Calculate what the Hubble parameter based on Plack 2018 best fit values.

omega_R = 9.2364e-5
omega_M = 0.321
omega_L = 0.679
H0 = 67.3 # km/s/Mpc

H_param = H0* np.sqrt( omega_R*(1+redshift)**4 + omega_M*(1+redshift)**3 + omega_L )
print( H_param)

H_param = H_param*(3.24078e-20) # Conversion to 1/sec.


# Converting wave number to units of 1/cm

wave_num = wave_num*(H0/100)/(3.086e24)

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

def energy_noF(N, L, J):

    # Unperturbed energy of the Hydrogen atom
    energy_0th = -13.6*eV_to_ergs / N**2

    # NOTE: Need to add perturbed terms to this quantity later.

    return energy_0th


# Defining variables to calculate Sobolev optical depth

A_2p = 6.2e8 # s^{-1}


num_baryon = baryon_to_photon_ratio * (2*zeta(3)/np.pi**2) * (kB*T)**3/(hbar*c)**3 # baryon number density
num_H_tot = (1-He_abund)*num_baryon # hydrogen number density

freq_lya = np.abs(energy_noF(2,1,0.5) - energy_noF(1,0,0.5)) / h # Lya frequency

optical_depth = 3*np.pi**2 * A_2p * num_H_tot * (1-x_e) * c**3/ ( H_param * (2*np.pi*freq_lya)**3 ) # Lya optical depth



        

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
    

    # Prefactors
    
    prefactor = np.sqrt(2*F[f0]+1)
    prefactor *= 1/(2*J[j]+1)
    prefactor *= np.sqrt(m_electron/(2*hbar**2))
    prefactor *= x_e  / 2
    prefactor *= Bohr_radius
    #print("Prefactor before loop:", prefactor)
    
    # Number density of baryons (cm^{-3})
    
    
    num_den_proton = (2*zeta(3)/np.pi**2)
    num_den_proton *= (kB*T/ (hbar*c))**3
    num_den_proton *= baryon_to_photon_ratio * x_e
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
            integral = 0
            
            for i in range(numE):
                
                integral += hstep*A_Einstein_array1[i,N,L,j,ju]*np.exp(-( energy_array[i] - chemical_potential) /(kB*T)) / np.sqrt(energy_array[i])
               
            
            temp *= integral
            
            term += temp
            
            
            '''
            
            
            integral = 0.5*hstep*A_Einstein_array1[0,N,L]*np.exp(-energy_array[0]/(kB*T))/np.sqrt(energy_array[0])
            
            integral += 0.5*hstep*A_Einstein_array1[numE-1,N,L]*np.exp(-energy_array[numE-1]/(kB*T))/np.sqrt(energy_array[numE-1]) 
                
            #print(0.5*hstep*A_Einstein_array1[numE-1,N,L]*np.exp(-energy_array[numE-1]/(kB*T))/np.sqrt(energy_array[numE-1]-ion_energy))
            
            for i in range(1,numE-1):
                        
                integral += hstep*A_Einstein_array1[i,N,L]*np.exp(-energy_array[i]/(kB*T))/np.sqrt(energy_array[i])
                
                #print(hstep*A_Einstein_array1[i,N,L]*np.exp(-energy_array[i]/(kB*T))/np.sqrt(energy_array[i]-ion_energy)) 
                
            temp *= integral 
            
            term += temp
            '''
            
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
                    
                    #print("Lu", Lu)
                    #print("Ju:", Ju[ju])
            
                    #integral = 0.5*hstep*A_Einstein_array1[0,N,L]*np.exp(-energy_array[0]/(kB*T))/np.sqrt(energy_array[0])
            
                    #integral += 0.5*hstep*A_Einstein_array1[numE-1,N,L]*np.exp(-energy_array[numE-1]/(kB*T))/np.sqrt(energy_array[numE-1]) 
                  
                    #print(integral)
                    
                    integral = 0

                    for i in range(numE):
                
                        integral += hstep*A_Einstein_array1[i,N,L,j,ju]*np.exp(- ( energy_array[i] - chemical_potential) / (kB*T)) / np.sqrt(energy_array[i])
             
                    #print("Integral:",integral)
                    temp *= integral
                    #print("Temp:",temp)
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
                    
                    integral = 0

                    for i in range(numE):
                
                        integral += hstep*A_Einstein_array2[i,N,L,j,ju]*np.exp(- ( energy_array[i] - chemical_potential)/(kB*T)) / np.sqrt(energy_array[i])
                     
                    temp *= integral
                    #print(temp)
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
    

def boundfree_photoionization(N, L, J, I, K, K_prime, Kr, F0, F1, F2, F3, pert_index):
    
   
    term = 0
    
    if L == 0:
        
        Lu = L + 1
        
        Ju = np.arange( np.abs(Lu-S), Lu+S+1, 1)
        
        if J == 0.5:
            j = 0
        
        
        for ju in range(len(Ju)):
            
            term1 = 0
            term2 = 0
            
            prefactor = (2*Ju[ju]+1)/(2*J+1)     # Conversiton factor from B(unbound to bound) and B(bound to unbound)                   
            prefactor *= Bohr_radius*np.sqrt(m_electron/(2*hbar**2))
            prefactor *= (2*J+1)*np.sqrt(3*(2*K+1)*(2*K_prime+1)*(2*Kr+1))
            prefactor *= (-1)**(1+Ju[ju]-I+ F0)
            prefactor *= wigner_6j(J,J,Kr,1,1,Ju[ju])*wigner_3j(K,K_prime,Kr,0,0,0)
            prefactor = np.float(prefactor)
            #print("Prefactor:", prefactor)
            
            if F0 == F2:
                
                term1 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
                term1 *= wigner_6j(J,J,Kr,F3,F1,I)*wigner_6j(K,K_prime,Kr,F3,F1,F0)
                term1 = np.float(term1)
                #print("Term1:", term1)
                
            if F1 == F3:
                
                term2 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
                term2 *= (-1)**(F2-F1+K+K_prime+Kr)
                term2 *= wigner_6j(J,J,Kr,F2,F0,I)*wigner_6j(K,K_prime,Kr,F2,F0,F1)
                term2 = np.float(term2)
                #print("Term2:", term2)
                
            integral = 0
                        
            for i in range(numE):
                
                freq_bf = ( energy_array[i] - energy(N,L,J,I,F0) )/h
                
                
               
                integral += hstep*B_Einstein_array1[i,N,L,j,ju]*rad_field_tensor(Kr, freq_bf, T, Theta_0, Theta_2, pert_index) / np.sqrt(energy_array[i])

                '''
                print("i:", i)                
                print("N:", N)
                print("L:", L)
                print("j:", j)
                print("ju:", ju)
                print("freq_bf:", freq_bf)
                print("hstep:", hstep)
                print("B_Einstein:", B_Einstein_array1[i,N,L,j,ju])
                print("rad_field_tensor:",rad_field_tensor(Kr, freq_bf, T, pert_index))
                
                '''
                #print("integral sum:", integral)
                                     
            #print ("Integral:", integral)
                
            temp = prefactor*(term1 + term2)*integral
            #print("Temp:", temp)
            
            term += temp
            #print("Term:", term)
            
    elif L > 0:
        
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
                   
                prefactor = (2*Ju[ju]+1)/(2*J+1) #factor to go from B(unbound->bound) to B(bound->unbound)
                prefactor *= Bohr_radius*np.sqrt(m_electron/(2*hbar**2))
                prefactor *= (2*J+1)*np.sqrt(3*(2*K+1)*(2*K_prime+1)*(2*Kr+1))
                prefactor *= (-1)**(1+Ju[ju]-I+ F0)
                prefactor *= wigner_6j(J,J,Kr,1,1,Ju[ju]) * wigner_3j(K,K_prime,Kr,0,0,0)
                prefactor = np.float(prefactor)
            
                if F0 == F2:
                
                    term1 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
                    term1 *= wigner_6j(J,J,Kr,F3,F1,I)*wigner_6j(K,K_prime,Kr,F3,F1,F0)
                    term1 = np.float(term1)
                
                if F1 == F3:
                
                    term2 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
                    term2 *= (-1)**(F2-F1+K+K_prime+Kr)
                    term2 *= wigner_6j(J,J,Kr,F2,F0,I)*wigner_6j(K,K_prime,Kr,F2,F0,F1)
                    term2 = np.float(term2)  
                    
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
                
    return term
                    
                    
                    
                
                
            
            
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
        prefactor *= x_e  / 2
        prefactor *= Bohr_radius
        
    elif k==1:
        
        prefactor = np.sqrt( 3*(2*F[f0]+1)*(2*F[f1]+1) )
        prefactor *= np.sqrt(m_electron/(2*hbar**2))
        prefactor *= x_e / 2
        prefactor *= Bohr_radius
        
        
    #print("Prefactor before loop:", prefactor)
    
    # Number density of baryons (cm^{-3})
    
    
    num_den_proton = (2*zeta(3)/np.pi**2)
    num_den_proton *= (kB*T/ (hbar*c))**3
    num_den_proton *= baryon_to_photon_ratio * x_e
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
            
            if k==0:
                temp = (2*Ju[ju] + 1)*prefactor
            
            elif k==1:
                
                temp = (-1)**(Ju[ju]+F[f1]+0.5)
                temp *= (2*Ju[ju] + 1)*prefactor
                temp *= np.float(wigner_6j(J[j], J[j], 2, F[f1], F[f0], 0.5))
                temp *= np.float(wigner_6j(J[j], J[j], 2, 1, 1, Ju[ju]))
                
            
            #hstep = np.abs(energy_array[1]-energy_array[0])
            #print(hstep)
            integral = 0
            
            for i in range(numE):
                
                
                freq_bf = ( energy_array[i] - energy(N,L,J[j],I,F[f0]) )/h
                
                integral += hstep*B_Einstein_array1[i,N,L,j,ju]*rad_field_tensor(K, freq_bf, T, Theta_0, Theta_2, pert_index)*np.exp(-( energy_array[i] - chemical_potential) /(kB*T)) / np.sqrt(energy_array[i])
               
            
            temp *= integral
            
            term += temp
            
            
            '''
            
            
            integral = 0.5*hstep*A_Einstein_array1[0,N,L]*np.exp(-energy_array[0]/(kB*T))/np.sqrt(energy_array[0])
            
            integral += 0.5*hstep*A_Einstein_array1[numE-1,N,L]*np.exp(-energy_array[numE-1]/(kB*T))/np.sqrt(energy_array[numE-1]) 
                
            #print(0.5*hstep*A_Einstein_array1[numE-1,N,L]*np.exp(-energy_array[numE-1]/(kB*T))/np.sqrt(energy_array[numE-1]-ion_energy))
            
            for i in range(1,numE-1):
                        
                integral += hstep*A_Einstein_array1[i,N,L]*np.exp(-energy_array[i]/(kB*T))/np.sqrt(energy_array[i])
                
                #print(hstep*A_Einstein_array1[i,N,L]*np.exp(-energy_array[i]/(kB*T))/np.sqrt(energy_array[i]-ion_energy)) 
                
            temp *= integral 
            
            term += temp
            '''
            
            if F[f0] < np.abs(J[j]-I) or F[f0] > J[j]+I:
                term = 0
    
    # Computes the bound state if L is not zero
        
    elif L > 0:
        
        for lu in range(2):
            
            Lu = L - 1 + 2*lu
            Ju = np.arange(np.abs(Lu-S), Lu+S+1, 1)  

            for ju in range(len(Ju)): 
                

                if k==0:
                    temp = (2*Ju[ju] + 1)*prefactor
            
                elif k==1:
                
                    temp = (-1)**(Ju[ju]+F[f1]+0.5)
                    temp *= (2*Ju[ju] + 1)*prefactor
                    temp *= np.float(wigner_6j(J[j], J[j], 2, F[f1], F[f0], 0.5))
                    temp *= np.float(wigner_6j(J[j], J[j], 2, 1, 1, Ju[ju]))
                    #print("temp:",temp)
                
                
                #print("Temp", temp)
                    
                #hstep = np.abs(energy_array[1]-energy_array[0])
                #print(hstep)
                
                if L < Lu:
                    
                    #print("Lu", Lu)
                    #print("Ju:", Ju[ju])
            
                    #integral = 0.5*hstep*A_Einstein_array1[0,N,L]*np.exp(-energy_array[0]/(kB*T))/np.sqrt(energy_array[0])
            
                    #integral += 0.5*hstep*A_Einstein_array1[numE-1,N,L]*np.exp(-energy_array[numE-1]/(kB*T))/np.sqrt(energy_array[numE-1]) 
                  
                    #print(integral)
                    
                    integral = 0

                    for i in range(numE):
                    
                        freq_bf = ( energy_array[i] - energy(N,L,J[j],I,F[f0]) )/h
                
                        integral += hstep*B_Einstein_array1[i,N,L,j,ju]*rad_field_tensor(K, freq_bf, T, Theta_0, Theta_2, pert_index)*np.exp(- ( energy_array[i] - chemical_potential) / (kB*T)) / np.sqrt(energy_array[i])
             
                    #print("Integral:",integral)
                    temp *= integral
                    #print("Temp:",temp)
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
                    
                    integral = 0

                    for i in range(numE):
                        
                        freq_bf = ( energy_array[i] - energy(N,L,J[j],I,F[f0]) )/h
                
                        integral += hstep*B_Einstein_array2[i,N,L,j,ju]*rad_field_tensor(K, freq_bf, T, Theta_0, Theta_2, pert_index)*np.exp(- ( energy_array[i] - chemical_potential)/(kB*T)) / np.sqrt(energy_array[i])
                     
                    temp *= integral
                    #print(temp)
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

    alpha_e = 1/137
    g = 2
    m = 9.109e-31 # electron mass in kg
    M_p = 1.673e-27 # proton mass in kg


    # Unperturbed energy of the Hydrogen atom
    energy_0th = -13.6*eV_to_ergs / N**2 
    #energy_1st = 0


    energy_1st = alpha_e**2*g*m/M_p
    energy_1st *= F*(F+1) - I*(I+1) - J*(J+1)
    energy_1st *= 1 / ( J*(J+1)*(2*L+1) )
    energy_1st *= 1/N**3
    energy_1st *= 13.6*eV_to_ergs

    #print(energy_1st)
    #print(energy_0th)

    # NOTE: Need to add perturbed terms to this quantity later.

    return energy_0th + energy_1st


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

    if L0 == L1-1 and n0 != n1:

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

    elif L0 == L1+1 and n0 != n1:

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

def Lambda_Lymann_lines(N0,N1,L0,L1,J0,J1,K0,K1,K_r,F0,F1,F2,F3,T,h_scalar,h_prime,eta_scalar,t_b, optical_depth,pert_index):
    
    term = 0 # initial value
    
    P_esc = 1/optical_depth # Approximately true during recombination.
    
    x_1s = 1-x_e # Just to see if we are getting appropriate values, I am setting this to 0th order.
    
    if N0==1 and L0==0 and N1==2 and L1==1
    :
        
        N_u = N1
        L_u = L1
        J_u = J1
        F_u = F2
        F_u_prime = F3
        K_u = K1
        
        N = N0
        L = L0
        J = J0
        F = F0
        F_prime = F1
        K = K0
        
        freq = energy_noF(N_u,L_u,J_u) - energy_noF(N, L, J)
        freq = np.abs(freq)/h
        
        weight = 2*h*freq**3/c**2
        x = h*freq/(kB*T)
        
        if x > 0:
        	phase_space = np.exp(-x)/(1-np.exp(-x))
        	phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2


        energy_exponential = np.exp(-x) # Boltzman exponential from n=2 to n=1
        
        B_Einstein_stim = 32*np.pi**4 / (3*h**2*c)
        B_Einstein_stim *= np.abs( dipole_element(N, N_u, L, L_u, J, J_u) )**2
        A_Einstein = weight*B_Einstein_stim

        if K==K_u and F==F_prime and freq > 0 and pert_index == False:
            
            term = (2*J_u+1)*(2*F+1)*np.sqrt((2*F_u+1)*(2*F_u_prime+1))
            term *= np.float( wigner_6j(F,F,K,F_u,F_u_prime,1)*wigner_6j(J_u,0.5,1,F,F_u,0.5)*wigner_6j(J_u,0.5,1,F,F_u_prime,0.5))
            term *= A_Einstein
            
        else:
            term = 0
            
    if N0 == 1 and L0 == 0 and N1==1 and L1==0 and J0==J1:
        
        N = N0
        L = L0
        J = J0
        F = F0
        F_prime = F1
        K = K0
        
        
        freq = energy_noF(2,1,0.5) - energy_noF(1, 0, 0.5)
        freq = np.abs(freq)/h
        
        weight = 2*h*freq**3/c**2
        x = h*freq/(kB*T)
        
        if x > 0:
        	phase_space = np.exp(-x)/(1-np.exp(-x))
        	phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2


        energy_exponential = np.exp(-x) # Boltzman exponential from n=2 to n=1
        
        B_stim_sum = 0
        
        for counter in range(2):
            
            J_u = 0.5 + counter # J_u = 1/2, 3/2
            B_Einstein_stim = 32*np.pi**4 / (3*h**2*c)
            B_Einstein_stim *= np.abs( dipole_element(1, 2, 0, 1, 0.5, J_u) )**2
            B_stim_sum += B_Einstein_stim            

        if K==K1 and F==F2 and F_prime==F3 and K_r == 0 and pert_index == False:
            
            term1 = P_esc*weight*phase_space # Unperturbed blackbody term in radiation field.
            term2 = (1-P_esc)*weight*energy_exponential # Unperturbed source term in radiation field
            
            term = -B_stim_sum*(term1+term2)
            
        elif K==K1 and F==F2 and F_prime==F3 and K_r == 0 and pert_index == True:
            
            term1 = -2*P_esc*(kB*T)**3*x**4*phase_deriv*Theta_0/(h*c)**2
            term2 = P_esc*weight*energy_exponential*delta_b/x_1s
            term3 = P_esc*(weight*phase_space -weight*energy_exponential)*Hubble_pert(0, h_scalar, h_prime, eta_scalar, t_b)
            
            term = -B_stim_sum*(term1+term2+term3)
            
        else:
            term = 0
 
    if N0 == 2 and L0 == 1 and N1==2 and L1==1 and J0==J1:
        
        N = N0
        L = L0
        J = J0
        F = F0
        F_prime = F1
        K = K0
        
        
        freq = energy_noF(2,1,J1) - energy_noF(1, 0, 0.5)
        freq = np.abs(freq)/h
        
        weight = 2*h*freq**3/c**2
        x = h*freq/(kB*T)
        
        if x > 0:
        	phase_space = np.exp(-x)/(1-np.exp(-x))
        	phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2


        energy_exponential = np.exp(-x) # Boltzman exponential from n=2 to n=1
        
        B_Einstein_stim = 32*np.pi**4 / (3*h**2*c)
        B_Einstein_stim *= np.abs( dipole_element(1, 2, 0, 1, 0.5, J) )**2
        A_Einstein = weight*B_Einstein_stim
              

        if K==K1 and F==F2 and F_prime==F3 and K_r == 0 and pert_index == False:
            
            term = A_Einstein
            
        else:
            term = 0

    if N0==2 and L0==1 and N1==1 and L1==0:
        
        N = N0
        L = L0
        J = J0
        F = F0
        F_prime = F1
        K = K0
        
        N_l = N1
        L_l = L1
        J_l = J1
        F_l = F2
        F_l_prime = F3
        K_l = K1
        
        freq = energy_noF(N,L,J) - energy_noF(N_l, L_l, J_l)
        freq = np.abs(freq)/h
        
        weight = 2*h*freq**3/c**2
        x = h*freq/(kB*T)
        
        if x > 0:
        	phase_space = np.exp(-x)/(1-np.exp(-x))
        	phase_deriv = - np.exp(-x)/(1-np.exp(-x))**2


        energy_exponential = np.exp(-x) # Boltzman exponential from n=2 to n=1
        
        B_Einstein_abs = 32*np.pi**4 / (3*h**2*c)
        B_Einstein_abs *= np.abs( dipole_element(N, N_l, L, L_l, J, J_l) )**2

        if K==K_l and F_l==F_l_prime and K_r==0 and pert_index == False:
            
            term = B_Einstein_abs
            term *= 2*np.sqrt(3)*(2*F_l+1)*np.sqrt( (2*F+1)*(2*F_prime+1)*(2*K+1) )
            term *= np.float(wigner_9j(F,F_l,1,F_prime,F_l,1,K,K,0))
            term *= np.float( wigner_6j(J,0.5,1,F_l,F,0.5)*wigner_6j(J,0.5,1,F_l,F_prime,0.5) )

            term1 = P_esc*weight*phase_space # Unperturbed blackbody term in radiation field.
            term2 = (1-P_esc)*weight*energy_exponential # Unperturbed source term in radiation field

            term = term*(term1+term2)            
            
        elif K==K_l and F_l==F_l_prime and K_r==0 and pert_index == True:
            
            term = B_Einstein_abs
            term *= 2*np.sqrt(3)*(2*F_l+1)*np.sqrt( (2*F+1)*(2*F_prime+1)*(2*K+1) )
            term *= np.float(wigner_9j(F,F_l,1,F_prime,F_l,1,K,K,0))
            term *= np.float( wigner_6j(J,0.5,1,F_l,F,0.5)*wigner_6j(J,0.5,1,F_l,F_prime,0.5) )
            
            
            term1 = -2*P_esc*(kB*T)**3*x**4*phase_deriv*Theta_0/(h*c)**2
            term2 = P_esc*weight*energy_exponential*delta_b/x_1s
            term3 = P_esc*(weight*phase_space -weight*energy_exponential)*Hubble_pert(0, h_scalar, h_prime, eta_scalar, t_b)
            
            term = term*(term1+term2+term3)

        if K==2 and K_l== 0 and F_l==F_l_prime and J == 1.5 and K_r==2 and pert_index == True:
            
            term = B_Einstein_abs
            term *= 2*np.sqrt(15)*(2*F_l+1)*np.sqrt( (2*F+1)*(2*F_prime+1))
            term *= np.float(wigner_9j(F,F_l,1,F_prime,F_l,1,2,0,0))
            term *= np.float( wigner_6j(1.5,0.5,1,F_l,F,0.5)*wigner_6j(1.5,0.5,1,F_l,F_prime,0.5) )

            term1 = np.sqrt(2)*P_esc*(kB*T)**3*x**4*phase_deriv*Theta_2/(h*c)**2
            term2 = 0
            term3 = (P_esc*weight*phase_space - weight*energy_exponential)*Hubble_pert(2, h_scalar, h_prime, eta_scalar, t_b)
            
            term = term*(term1+term2+term3)

        else:
            term = 0
            

    return term
             

    
    
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

def Nhat(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3):

    # Calculating the Lande g-factor

    g = J0*(J0+1) + 2.25 - l0*(l0+1)
    g = g/( 2*J0*(J0+1))
    g += 1 

    # Defining each term to be used later on

    term1 = 0
    term2 = 0
    term3 = 0
    term4 = 0

    # Calculating the Bohr frequency nu_alphaJIF, alphaJIF'
    if n0 == n1 and l0 == l1 and J0 == J1 and K0==K1 and F0==F2 and F1==F3:
        term1 = energy(n0, l0, J0, I, F0) - energy(n1, l1, J1, I, F1)
        term1 = np.abs(term1)/h

    
    # Calculating more terms
    if n0 == n1 and l0==l1 and J0 == J1:

        J = J0
    
        term2 = larmor_freq
        term2 *= g*(-1)**(F1+F2)
        term2 *= np.sqrt( (2*K0+1)*(2*K1+1) )
        term2 *= float(wigner_3j(K0, K1, 1, 0, 0, 0))


        if F1==F3:
            term3 = (-1)**(1+J+I+F0)
            term3 *= np.sqrt(J*(J+1)*(2*J+1)*(2*F0+1)*(2*F2+1))
            term3 *= float(wigner_6j(F0, F2, 1, J, J, I))
            term3 *= float(wigner_6j(K0, K1, 1, F2, F0, F1))

        if F0==F2:
            term4 = (-1)**(K0-K1)
            term4 *= (-1)**(1+J+I+F0)
            term4 *= np.sqrt(J*(J+1)*(2*J+1)*(2*F1+1)*(2*F3+1))
            term4 *= float(wigner_6j(F1, F3, 1, J, J, I))
            term4 *= float(wigner_6j(K0, K1, 1, F3, F1, F0))


    return term1+term2*(term3+term4)

'''

Note that all the T functions more-or-less behave the same way as far as input and output are concerned.

Input:

    1) Two sets of (n, l, J, K, F, F'). 

    2) Must obey the laws of physics such as J = L-1/2, L + 1/2.

    3) Frequency since we need to define a radiation field tensor.

Output:

    1) Tracks evolution and coherence between different states.

'''

    
    

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

        B_Einstein = 32*np.pi**4 / (3*h**2*c)
        B_Einstein *= np.abs( dipole_element(n_l, n, l_l, l, J_l, J) )**2

        #print("This is the B Einstein coefficient:", B_Einstein)

        term1 = (2*J_l + 1)*B_Einstein # Prefactor to the sum
        
        # Frequency of radiation (s^{-1})
        freq = (energy_noF(n, l, J) - energy_noF(n_l, l_l, J_l))/h
        freq = np.abs(freq)

        term2 = 0 # Value of the sum across K_r 
    
        # Computing the sum across K_r from 0 to 2 where Q_r=0 is fixed.

        temp = np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_l+1)*(2*F_lprime+1)*(2*K+1)*(2*K_l+1)*(2*Kr+1))
        temp *= (-1)**(K_l + F_lprime - F_l)
        temp *= float(wigner_9j(F, F_l, 1, F_prime, F_lprime, 1, K, K_l, Kr) ) * float(wigner_6j(J, J_l, 1, F_l, F, I))
        temp *= float(wigner_6j(J, J_l, 1, F_lprime, F_prime, I) ) * float(wigner_3j(K, K_l, Kr, 0, 0, 0) )
        temp *= rad_field_tensor(Kr, freq, T, Theta_0, Theta_2,pert_index)
        #temp *= rad_field_tensor(Kr, n0, n1, l0, l1, J0, J1, T, pert_index)


        term2 += temp



    return term1*term2




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
    
        B_Einstein = 32*np.pi**4 / (3*h**2*c)
        B_Einstein *= np.abs( dipole_element(n_u, n, l_u, l, J_u, J) )**2
        term1 = (2*J_u + 1)*B_Einstein
        
        # Frequency of radiation (s^{-1})
        freq = (energy_noF(n_u, l_u, J_u) - energy_noF(n, l, J))/h
        freq = np.abs(freq)


        #print("This is the B Einstein coefficient:", B_Einstein)

        # Computing the sum across K_r from 0 to 2 where Q_r=0 is fixed.


        temp = np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_u+1)*(2*F_uprime+1)*(2*K+1)*(2*K_u+1)*(2*Kr+1))
        temp *= (-1)**(Kr+K_u+F_uprime-F_u)
        temp *= float( wigner_9j(F, F_u, 1, F_prime, F_uprime, 1, K, K_u, Kr))
        temp *= float( wigner_6j(J_u, J, 1, F, F_u, I) )
        temp *= float( wigner_6j(J_u, J, 1, F_prime, F_uprime, I) )
        temp *= float( wigner_3j(K, K_u, Kr, 0, 0, 0) )
        temp *= rad_field_tensor(Kr, freq, T, Theta_0, Theta_2, pert_index)       
        #temp *= rad_field_tensor(Kr, n0, n1, l0, l1, J0, J1, T, pert_index)

        term2 += temp

    

    return term1*term2

def T_E(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3):

    # Compute the frequency of the transition.

    freq = energy_noF(n0, l0, J0) - energy_noF(n1, l1, J1)
    freq = np.abs(freq)/h

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

        A_Einstein = 64*np.pi**4 * freq**3 / (3*h*c**3)
        A_Einstein *= np.abs( dipole_element(n_u, n, l_u, l, J_u, J) )**2
        #print("This is the A Einstein coefficient:", A_Einstein)

        # Computing the term itself
        term = 0

        if K == K1:
        
            term = (2*J_u + 1)*A_Einstein
            term *= np.sqrt( (2*F+1)*(2*F_prime+1)*(2*F_u+1)*(2*F_uprime+1) )
            term *= (-1)**(1 + K + F_prime + F_uprime)
            term *= float( wigner_6j(F , F_prime, K, F_uprime, F_u, 1) )
            term *= float( wigner_6j(J_u , J, 1, F, F_u, I) )
            term *= float( wigner_6j(J_u, J, 1, F_prime, F_uprime, I) )
    


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

                if n_level > n:
                    if l_level == l-1 or l_level == l or l_level == l+1:

                        #print("Atomic levels (n,l,J_u)")
                        #print(n_level,l_level,J_u)
                        
                        # Frequency of radiation (s^{-1})
                        freq = (energy_noF(n_level, l_level, J_u) - energy_noF(n, l, J))/h
                        freq = np.abs(freq)


                        B_Einstein = 32*np.pi**4 / (3*h**2*c)
                        B_Einstein *= np.abs( dipole_element(n, n_level, l, l_level, J, J_u) )**2

                        
                        term1 = B_Einstein
                        term1 *= np.sqrt(3*(2*K+1)*(2*K_prime+1)*(2*Kr+1))
                        term1 *= (-1)**(1+J_u-I+F0)
                        term1 *= float( wigner_6j(J, J, Kr, 1, 1, J_u) )
                        term1 *= float( wigner_3j(K, K_prime, Kr, 0,0, 0) )
                        term1 *=(2*J+1)
                        #print("This is term1")
                        #print(term1)
    
                        if F0 == F2:
                            term2 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
                            term2 *= float( wigner_6j(J, J, Kr, F3, F1, I) )
                            term2 *= float( wigner_6j(K, K_prime, Kr, F3, F1, F0) )
                            #term2 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_u, T, pert_index)
                            term2 *= rad_field_tensor(K, freq, T, Theta_0, Theta_2, pert_index)
                            #print("This is term2")
                            #print(term2)
                        if F1 == F3:
                            term3 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
                            term3 *= (-1)**(F2 - F1 + K + K_prime + Kr)
                            term3 *= float( wigner_6j(J, J, Kr, F2, F0, I) )
                            term3 *= float( wigner_6j(K, K_prime, Kr, F2, F0, F1) )
                            #term3 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_u, T, pert_index)
                            term3 *= rad_field_tensor(K, freq, T, Theta_0, Theta_2, pert_index)
                            #print("This is term3")
                            #print(term3)

                        total_term += term1*(term2+term3)

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

                            freq = energy_noF(n, l, J) - energy_noF(n_level, l_level, J_l) / h
                            freq = np.abs(freq) 

                            # Compute Einstein A coefficient

                            #print("Atomic levels (n,l,J_l)")
                            #print(n_level,l_level,J_l)


                            A_Einstein = 64*np.pi**4 * freq**3 / (3*h*c**3)
                            A_Einstein *= np.abs( dipole_element(n, n_level, l, l_level, J, J_l) )**2

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

                if n_level < n:
                    if l_level == l-1 or l_level == l or l_level == l+1:

                        #print("Atomic levels (n,l,J_l)")
                        #print(n_level,l_level,J_l)


                        # Calculate frequency

                        freq = energy_noF(n, l, J) - energy_noF(n_level, l_level, J_l)
                        freq = np.abs(freq) / h

                        # Calculating allowed Einstein-B coefficients

                        B_Einstein = 32*np.pi**4 / (3*h**2*c)
                        B_Einstein *= np.abs( dipole_element(n, n_level, l, l_level, J, J_l) )**2
                        
                        term1 = B_Einstein
                        term1 *= np.sqrt( 3*(2*K+1)*(2*K_prime+1)*(2*Kr+1) )
                        term1 *= (-1)**(1+J_l - I + F0 +Kr)
                        term1 *= float( wigner_6j(J, J, Kr, 1, 1, J_l) )
                        term1 *= float( wigner_3j(K, K_prime, Kr, 0,0,0) )
                        term1 *= (2*J+1) #Prefactor
                        #print("This is term1")
                        #print(term1)
    
                        if F0 == F2:
                            term2 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
                            term2 *= float( wigner_6j(J, J, Kr, F3, F1, I) )
                            term2 *= float( wigner_6j(K, K_prime, Kr, F3, F1, F0) )
                            #term2 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_l, T, pert_index)
                            term2 *= rad_field_tensor(K, freq, T, Theta_0, Theta_2,pert_index)
                            #print("This is term2")
                            #print(term2)
    
                        if F1 == F3:
                            term3 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
                            term3 *= (-1)**(F2 - F1 + K + K_prime + Kr)
                            term3 *= float( wigner_6j(J, J, Kr, F2, F0, I) )
                            term3 *= float( wigner_6j(K, K_prime, Kr, F2, F0, F1) )
                            #term3 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_l, T, pert_index)
                            term3 *= rad_field_tensor(K, freq, T, Theta_0, Theta_2, pert_index)
                            #print("This is term3")
                            #print(term3)
                        
                        total_term += term1*(term2+term3) # Summing each term.


    
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

def source_function_2photon(N, l, j, k, F0, F1, density_matrix):
    
    twophoton_rate = 8.22 # s^{-1}
    energy_dif = 10.2*eV_to_ergs # Energy from n=2 to n=1
    
    
    if N == 1 and L == 0 and J == 0.5:
        source_term =twophoton_rate*density_matrix[N,l,j, k, f0, f1]
    elif N == 2 and L == 0 and J == 0.5:
        source_term = twophoton_rate* np.exp(energy_dif/(kB*T)) *density_matrix[N, l, j, k, f0, f1]
    else:
        source_term = 0
        
    return source_term

	
'''
Arrays for the unperturbed evolution matrix (Lambda0), the perturbed evolution matrix corresponding to K=0 (L0),
and the perturbed evolution matrix corresponding to K=2 (L2).

The structure of each matrix is the following:

Matrix(N0, L0, J0, K0, F0, F1, N1, L1, J1, K1, F2, F3)

'''


Lambda0 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)

L0 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)

L2 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)

mask_array = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)






'''
We want to define the matrix structure for the density matrix and the source function so that we can compute matrix
multiplication.
'''

density_matrix = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)
source_matrix = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)





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

for i in range(numE):
    
    E_free = hstep/2 + i*hstep
    
    
    #E_free = i*(Emax/(numE-1)) + 1e-8*eV_to_ergs
    energy_array[i] = E_free
       
    bf_dipole_array1 = dipole_bf(numN, E_free, True) # Bound state l < Free state l
    bf_dipole_array2 = dipole_bf(numN, E_free, False) # Bound state l > Free state l

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
                        ang_prefactor *= np.float(wigner_6j(L_u,L,1,J[j],J_u[j_u],S))

                        ang_prefactor *= np.sqrt(L_u / (2*L_u+1))
                                                
    
                        freq = (E_free + ion_energy/N**2) /h #Hydrogen energy is negative, so total is positive.

        
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
                        ang_prefactor *= np.float(wigner_6j(L_u,L,1,J[j],J_u[j_u],S))

                        ang_prefactor *= np.sqrt(L / (2*L_u+1))
                                                
    
                        freq = (E_free + ion_energy/N**2) /h #Hydrogen energy is negative, so total is positive.

        
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
                                                        

                                                    print("####### New State #####", file = output_file)
                                                    print(" ", file = output_file)
                                                    print("N0:"+str(N0), file = output_file)
                                                    print("N1:"+str(N1), file = output_file)
                                                    print("L0:"+str(l0), file = output_file)
                                                    print("L1:"+str(l1), file = output_file)
                                                    print("J0:"+str(J0[j0]), file = output_file)
                                                    print("J1:"+str(J1[j1]), file = output_file)
                                                    print("K0:"+str(K0), file = output_file)
                                                    print("K1:"+str(K1), file = output_file)
                                                    print("Kr:"+str(Kr), file = output_file)
                                                    print("F0:"+str(F0[f0]), file = output_file)
                                                    print("F1:"+str(F1[f1]), file = output_file)
                                                    print("F2:"+str(F2[f2]), file = output_file)
                                                    print("F3:"+str(F3[f3]), file = output_file)
                                                    print(" ", file = output_file)
                                                    
                                                    print("##### Code Index #####", file = output_file)
                                                    print(" ", file = output_file)
                                                    print("N0:"+str(N0), file = output_file)
                                                    print("N1:"+str(N1), file = output_file)
                                                    print("L0:"+str(l0), file = output_file)
                                                    print("L1:"+str(l1), file = output_file)
                                                    print("j0:"+str(j0), file = output_file)
                                                    print("j1:"+str(j1), file = output_file)
                                                    print("k0:"+str(k0), file = output_file)
                                                    print("k1:"+str(k1), file = output_file)
                                                    print("kr:"+str(kr), file = output_file)
                                                    print("f0:"+str(f0), file = output_file)
                                                    print("f1:"+str(f1), file = output_file)
                                                    print("f2:"+str(f2), file = output_file)
                                                    print("f3:"+str(f3), file = output_file)
                                                    print(" ", file = output_file)
                                                    
                                                    print("Mapping to 2D fits file", file = output_file)
                                                    
                                                    # Mapping from 12D array to 2D array. Where are the elements?
                                                    
                                                    if N1>1:
                                                        element_x = 16*numN*(N1-2) + 16*l1 + 8*j1 + 4*k1 + 2*f2 + f3 + 1
                                                        print("Nx (excited): "+str(element_x), file = output_file)                                                    
                                                    else:
                                                        element_x = 8*j1 + 4*k1 + 2*f2 + f3 + 1
                                                        print("Nx (1s): "+str(element_x), file = output_file)
                                                    
                                                    if N0>1:
                                                        element_y = 16*numN*(N0-2) + 16*l0 + 8*j0 + 4*k0 + 2*f0 + f1 + 1
                                                        print("Ny (excited): "+str(element_y), file = output_file)                                                    
                                                    else:
                                                        element_y = 8*j0 + 4*k0 + 2*f0 + f1 + 1
                                                        print("Ny (1s): "+str(element_y), file = output_file)                                                        
                                                    
                                                    


                                                    
                                        
                                                
                                                    

                                                    if N0==N1 and l0==l1 and J0[j0]==J1[j1]:
                                                        
                                                        print(" ", file=output_file)
                                                        print("R terms (and N hat)", file=output_file)
                                                        
                                                        Nhat_total = Nhat(N0,N1,l0, l1, J0[j0], J1[j1], K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])
                                                        Nhat_total *= -2*np.pi*complex(0,1)
                                                        
                                                        print("Nhat: "+str(Nhat_total), file = output_file)
                                                        
                                                        if Kr==0:
                                                            
                                                            RA_unpert = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)
                                                            RS_unpert = R_S(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)
                                                            RE_total = R_E(N0, l0, J0[j0], I, K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])
                                                
                                                            RA_pert_0 = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                            RS_pert_0 = R_S(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                            
                                                            Lymann_unpert = Lambda_Lymann_lines(N0, N1, l0, l1, J0, J1, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], T, h_scalar, h_prime, eta_scalar, t_b, optical_depth, False)
                                                            Lymann_pert_0 = Lambda_Lymann_lines(N0, N1, l0, l1, J0, J1, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], T, h_scalar, h_prime, eta_scalar, t_b, optical_depth, True)
                                                            
                                                            # Photoionization terms
                                                            
                                                            photo_unpert = boundfree_photoionization(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)
                                                            photo_pert_0 = boundfree_photoionization(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                               
                                                            print("RA_unpert: "+str(RA_unpert), file=output_file)
                                                            print("RS_unpert: "+str(RS_unpert), file=output_file)
                                                            print("RE_total: "+str(RE_total), file=output_file)
                                                            
                                                            print("RA_pert_0: "+str(RA_pert_0), file=output_file)
                                                            print("RS_pert_0: "+str(RS_pert_0), file=output_file)
                                                            
                                                            print("photoionization_unpert: " + str(photo_unpert), file=output_file)
                                                            print("photoionization_pert_0: " + str(photo_pert_0), file=output_file)
                                                            
                                                            print("Lymann_unpert: " + str(Lymann_unpert), file=output_file)
                                                            print("Lymann_pert_0: " + str(Lymann_pert_0), file=output_file)                                                            
                                                         
                                                            #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Nhat_total - RA_unpert - RS_unpert - RE_total
                                                            #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += -RA_pert_0 - RS_pert_0

                                                            #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Nhat_total - RA_unpert - RS_unpert - RE_total - photo_unpert - Lymann_unpert
                                                            #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += -RA_pert_0 - RS_pert_0 - photo_pert_0 - Lymann_pert_0

                                                            Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - Lymann_unpert
                                                            L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - Lymann_pert_0
                                                            
                                                            '''
                                                            Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - photo_unpert
                                                            L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - photo_pert_0
                                                            '''
                                                            
                                                        if Kr==2:
                                                            
                                                            RA_pert_2 = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                            RS_pert_2 = R_S(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                            
                                                            photo_pert_2 = boundfree_photoionization(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)

                                                            Lymann_pert_2 = Lambda_Lymann_lines(N0, N1, l0, l1, J0, J1, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], T, h_scalar, h_prime, eta_scalar, t_b, optical_depth, True)
                                                                                                                        
                                                            print("RA_pert_2: " +str(RA_pert_2), file=output_file)
                                                            print("RS_pert_2: " +str(RS_pert_2), file=output_file)
                                                            
                                                            print("photoionization_pert_2: " + str(photo_pert_2), file=output_file)
                                                            print("Lymann_pert_2: " + str(Lymann_pert_2), file=output_file)
                                                            

                                                            #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - RA_pert_2 - RS_pert_2
                                                            #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - RA_pert_2 - RS_pert_2 - photo_pert_2 - Lymann_pert_2
                                                            L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - Lymann_pert_2                                                            
                                                            #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += - photo_pert_2
                                                            
                                                        print(" ", file = output_file)
                                                        print("Lambda0: "+str(Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)
                                                        print("L0: "+ str(L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)                                                                                                                        
                                                        print("L2: "+ str(L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)                                                    

                                                    elif N0 != N1:
                                                        
                                                            print(" ", file=output_file)
                                                            print("T terms", file=output_file)
                                                            
                                                            if Kr == 0:
                                                                
                                                                TA_unpert = T_A(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)
                                                                TS_unpert = T_S(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)
                                                                TE_total = T_E(N0, N1, l0, l1 , J0[j0], J1[j1], K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])
                                                                
                                                                TA_pert_0 = T_A(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                                TS_pert_0 = T_S(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)                                                                                                      
									
                                                                Lymann_unpert = Lambda_Lymann_lines(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], T, h_scalar, h_prime, eta_scalar, t_b, optical_depth, False)

                                                                Lymann_pert_0 = Lambda_Lymann_lines(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], T, h_scalar, h_prime, eta_scalar, t_b, optical_depth, True)  

                                                                print("TA_unpert: "+str(TA_unpert), file=output_file)
                                                                print("TS_unpert: "+str(TS_unpert), file=output_file)
                                                                print("TE_total: "+str(TE_total), file=output_file)
                                                            
                                                                print("TA_pert_0: "+str(TA_pert_0), file=output_file)
                                                                print("TS_pert_0: "+str(TS_pert_0), file=output_file)
                                                                
                                                                print("Lymann_unpert: "+str(Lymann_unpert), file=output_file)
                                                                print("Lymann_pert_0: "+str(Lymann_pert_0), file=output_file)
                                                            
                                                            
                                                                    
                                                                # Version that just gives you Lymann line contribution
                                                             
                                                                 
                                                                if N1 < N0 and K0 == 0 and K0 == K1:
                                                                    Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_unpert
                                                                    L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_pert_0
                                                                    #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_unpert + Lymann_unpert
                                                                    #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_pert_0 + Lymann_pert_0
                                                                    #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_unpert 
                                                                    #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_pert_0 
                                                                elif N1 > N0 and K0 == 0 and K0 == K1:
                                                                    Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_unpert
                                                                    L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_pert_0
                                                                    #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TE_total + TS_unpert + Lymann_unpert
                                                                    #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TS_pert_0 + Lymann_pert_0                                                                
                                                                    #Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TE_total + TS_unpert 
                                                                    #L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TS_pert_0                                                                     
                                                             	  
                                                                                                                          
                                                                
                                                                                                                                   
                                                            elif Kr == 2:
                                                                
                                                                TA_pert_2 = T_A(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                                TS_pert_2 = T_S(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)                                                                                                      
 
                                                                Lymann_pert_2 = Lambda_Lymann_lines(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], T, h_scalar, h_prime, eta_scalar, t_b, optical_depth, True)
                                                                
                                                                print("TA_pert_2: "+str(TA_pert_2), file=output_file)
                                                                print("TS_pert_2: "+str(TS_pert_2), file=output_file)
                                                                
                                                                print("Lymann_pert_2: "+str(Lymann_pert_2), file=output_file)
                                                                                                                               
                                                                                                                           
                                                                if N1 < N0 and K0 == 2 and K0 == K1:
                                                                    L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_pert_2                                                             
                                                                    #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_pert_2
                                                                    #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_pert_2 + Lymann_pert_2                                                                    
                                                                elif N1 > N0 and K0 == 2 and K0 == K1:
                                                                    L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Lymann_pert_2                                                                                   
                                                                              
                                                                    #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TS_pert_2
                                                                    #L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TS_pert_2  + Lymann_pert_2                                       
                                                                                             

                                                            print("Lambda0: "+str(Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)
                                                            print("L0: "+str(L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)                                                                                                                        
                                                            print("L2: "+str(L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)                                                    
                                                          
                                                                



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


Lambda0_exc_exc_masked = np.zeros( (num_exc_phy, num_exc_phy), dtype = np.complex)
Lambda0_1s_exc_masked = np.zeros( (num_1s_phy, num_exc_phy), dtype = np.complex)
Lambda0_exc_1s_masked = np.zeros( (num_exc_phy, num_1s_phy), dtype = np.complex)
Lambda0_1s_1s_masked = np.zeros( (num_1s_phy, num_1s_phy), dtype = np.complex)

L0_exc_exc_masked = np.zeros( (num_exc_phy, num_exc_phy), dtype = np.complex)
L0_1s_exc_masked = np.zeros( (num_1s_phy, num_exc_phy), dtype = np.complex)
L0_exc_1s_masked = np.zeros( (num_exc_phy, num_1s_phy), dtype = np.complex)
L0_1s_1s_masked = np.zeros( (num_1s_phy, num_1s_phy), dtype = np.complex)


L2_exc_exc_masked = np.zeros( (num_exc_phy, num_exc_phy), dtype = np.complex)
L2_1s_exc_masked = np.zeros( (num_1s_phy, num_exc_phy), dtype = np.complex)
L2_exc_1s_masked = np.zeros( (num_exc_phy, num_1s_phy), dtype = np.complex)
L2_1s_1s_masked = np.zeros( (num_1s_phy, num_1s_phy), dtype = np.complex)


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


# Saving into fits files for Lambda0

hdu1 = fits.PrimaryHDU(np.abs(Lambda0_1s_1s_masked))
hdu1.writeto("lambda0_1s_1s_z="+str(redshift)+".fits", overwrite = True)

hdu2 = fits.PrimaryHDU(np.abs(Lambda0_1s_exc_masked))
hdu2.writeto("lambda0_1s_exc_z="+str(redshift)+".fits", overwrite = True)

hdu3 = fits.PrimaryHDU(np.abs(Lambda0_exc_1s_masked))
hdu3.writeto("lambda0_exc_1s_z="+str(redshift)+".fits", overwrite = True)

hdu4 = fits.PrimaryHDU(np.abs(Lambda0_exc_exc_masked))
hdu4.writeto("lambda0_exc_exc_z="+str(redshift)+".fits", overwrite = True)

# Saving into fits files for L0

hdu5 = fits.PrimaryHDU(np.abs(L0_1s_1s_masked))
hdu5.writeto("L0_1s_1s_z="+str(redshift)+".fits", overwrite = True)

hdu6 = fits.PrimaryHDU(np.abs(L0_1s_exc_masked))
hdu6.writeto("L0_1s_exc_z="+str(redshift)+".fits", overwrite = True)

hdu7 = fits.PrimaryHDU(np.abs(L0_exc_1s_masked))
hdu7.writeto("L0_exc_1s_z="+str(redshift)+".fits", overwrite = True)

hdu8 = fits.PrimaryHDU(np.abs(L0_exc_exc_masked))
hdu8.writeto("L0_exc_exc_z="+str(redshift)+".fits", overwrite = True)

# Saving into fits files for L2

hdu9 = fits.PrimaryHDU(np.abs(L2_1s_1s_masked))
hdu9.writeto("L2_1s_1s_z="+str(redshift)+".fits", overwrite = True)

hdu10 = fits.PrimaryHDU(np.abs(L2_1s_exc_masked))
hdu10.writeto("L2_1s_exc_z="+str(redshift)+".fits", overwrite = True)

hdu11 = fits.PrimaryHDU(np.abs(L2_exc_1s_masked))
hdu11.writeto("L2_exc_1s_z="+str(redshift)+".fits", overwrite = True)

hdu12 = fits.PrimaryHDU(np.abs(L2_exc_exc_masked))
hdu12.writeto("L2_exc_exc_z="+str(redshift)+".fits", overwrite = True)



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

source_matrix_unpert = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)
source_matrix_pert_0 = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)
source_matrix_pert_2 = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)

mask_matrix = np.zeros((numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)

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
            
                        
                        if k == 0:
                            
                            source_matrix_unpert[N,L,j,k,f0,f1] = source_boundfree_spontaneous(N,L,j,k,f0,f1,energy_array)
                            source_matrix_unpert[N,L,j,k,f0,f1] += source_boundfree_stimulated(N,L,j,k,f0,f1,False,energy_array)

                            source_matrix_pert_0[N,L,j,k,f0,f1] = source_boundfree_stimulated(N,L,j,k,f0,f1,True,energy_array)
                            
                        elif k == 2:

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

hdu_source_1s_unpert = fits.PrimaryHDU(np.abs(source_1s_unpert_masked))
hdu_source_1s_unpert.writeto("source_1s_unpert_z="+str(redshift)+".fits", overwrite = True)

hdu_source_1s_pert_0 = fits.PrimaryHDU(np.abs(source_1s_pert_0_masked))
hdu_source_1s_pert_0.writeto("source_1s_pert_0_z="+str(redshift)+".fits", overwrite = True)

hdu_source_1s_pert_2 = fits.PrimaryHDU(np.abs(source_1s_pert_2_masked))
hdu_source_1s_pert_2.writeto("source_1s_pert_2_z="+str(redshift)+".fits", overwrite = True)

hdu_source_exc_unpert = fits.PrimaryHDU(np.abs(source_exc_unpert_masked))
hdu_source_exc_unpert.writeto("source_exc_unpert_z="+str(redshift)+".fits", overwrite = True)

hdu_source_exc_pert_0 = fits.PrimaryHDU(np.abs(source_exc_pert_0_masked))
hdu_source_exc_pert_0.writeto("source_exc_pert_0_z="+str(redshift)+".fits", overwrite = True)

hdu_source_exc_pert_2 = fits.PrimaryHDU(np.abs(source_exc_pert_2_masked))
hdu_source_exc_pert_2.writeto("source_exc_pert_2_z="+str(redshift)+".fits", overwrite = True)

# Creating a fits file for the density matrix

#hdu_density_1s_unpert = fits.PrimaryHDU(np.abs(density_1s_unpert))
#hdu_density_1s_unpert.writeto("density_1s_unpert.fits", overwrite = True)

#hdu_density_exc_unpert = fits.PrimaryHDU(np.abs(density_exc_unpert))
#hdu_density_exc_unpert.writeto("density_exc_unpert_z="+str(redshift)+".fits", overwrite = True)




