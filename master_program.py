import sys
import subprocess
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt


    
# Constants

c = 3e10 # Speed of light ( cm/s )


# We want to make a module that imports the T(k) values for a given k. We will then loop over it
# and then perform our RK4 program.

redshift_list = np.loadtxt(open("redshift_list.txt"))

z_index =  redshift_list[1]-redshift_list[0]

# Current wave number 

wave_num = 1.042057218184e-05


thermo_array = np.loadtxt(open("thermodynamics.dat"))
tk_array = np.loadtxt(open("tk_"+str(redshift_list[0])+".dat"))

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
        
        if round(thermo_array[line][1],2) == redshift_list[0]:
            #print("The redshift in the file is "+str(thermo_array[line][1]))
            #print("The redshift is in line "+str(line))
            
            zline = line
            found_zline = True
    
    if found_zline == False:        
        redshift = input("Redshift not found. Please input a new redshift: ")


# Important variables acquired from the Thermodynamics datafile
    
scale_factor = thermo_array[zline][0]
redshift = thermo_array[zline][1]
T = 2.73*(1+redshift)
conform_time = thermo_array[zline][2]
X_e = thermo_array[zline][3]

conform_time = conform_time * (3.0824e24) / c # Conversion to seconds    
                
# Calculate what the Hubble parameter based on Plack 2018 best fit values.

omega_R = 9.2364e-5
omega_M = 0.321
omega_L = 0.679
H0 = 67.3 # km/s/Mpc

H_param = H0* np.sqrt( omega_R*(1+redshift)**4 + omega_M*(1+redshift)**3 + omega_L )
H_param = H_param*(3.24078e-20) # Conversion to 1/sec.



# Create an array for the initial density matrix

rho_1s_unpert = np.zeros(3)
rho_1s_pert0 = np.zeros(3)
rho_1s_pert2 = np.zeros(3)


X_1s = 1-X_e
rho_1s_unpert[0] = X_1s/4
rho_1s_unpert[1] = np.sqrt(3)*X_1s/4
rho_1s_unpert[2] = 0

# Reshaping our array to be viewable on ds9
rho_1s_unpert = rho_1s_unpert.reshape(3,1) 

hdu_rho_1s_unpert = fits.PrimaryHDU(rho_1s_unpert )
hdu_rho_1s_unpert.writeto("rho_1s_unpert_z="+str(redshift_list[0])+".fits", overwrite = True)


rho_1s_unpert = rho_1s_unpert.reshape(3)

# Making rho_1s_unpert complex

rho_1s_unpert = rho_1s_unpert*complex(1,0)

# Defining arrays to compute plots


rho_1s_Fis0 = np.zeros(len(redshift_list)-1)
rho_1s_Fis1 = np.zeros(len(redshift_list)-1)
rho_1s_Kis2 = np.zeros(len(redshift_list)-1)


source_1s_Fis0 = np.zeros(len(redshift_list)-1)
source_1s_Fis1 = np.zeros(len(redshift_list)-1)

source_exc_2s_Fis0 = np.zeros(len(redshift_list)-1)
source_exc_2s_Fis1 = np.zeros(len(redshift_list)-1)
source_exc_2p_half_Fis0 = np.zeros(len(redshift_list)-1)
source_exc_2p_half_Fis1 = np.zeros(len(redshift_list)-1)
source_exc_2p_3half_Fis0 = np.zeros(len(redshift_list)-1)
source_exc_2p_3half_Fis1 = np.zeros(len(redshift_list)-1)

Eigenvalue_Fis0 = []
Eigenvalue_Fis1 = []

deriv_func_Fis0 = []
deriv_func_Fis1 = []

# Running code for i=0 

rho_1s_Fis0_val = rho_1s_unpert[0]
rho_1s_Fis1_val = rho_1s_unpert[1]
rho_1s_Kis2_val = rho_1s_unpert[2]

redshift = redshift_list[0]

exec(open("new_matrices_code.py").read())

# RUnning code for i=1

redshift = thermo_array[zline-1,1]
X_e = thermo_array[zline-1,3]

X_1s = 1-X_e
rho_1s_unpert[0] = X_1s/4
rho_1s_unpert[1] = np.sqrt(3)*X_1s/4
rho_1s_unpert[2] = 0


rho_1s_Fis0_val = rho_1s_unpert[0]
rho_1s_Fis1_val = rho_1s_unpert[1]
rho_1s_Kis2_val = rho_1s_unpert[2]

exec(open("new_matrices_code.py").read())


for i in range(len(redshift_list)-1):

    # Opening Lambda functions
    
    redshift  = redshift_list[i+1]
    exec(open("new_matrices_code.py").read())

    # We compute the unperturbed piece in our equations
        
    lambda0_1s_1s_real_fits = fits.open("lambda0_1s_1s_z="+str(redshift_list[i+1])+"_real.fits")
    lambda0_1s_exc_real_fits = fits.open("lambda0_1s_exc_z="+str(redshift_list[i+1])+"_real.fits")
    lambda0_exc_1s_real_fits = fits.open("lambda0_exc_1s_z="+str(redshift_list[i+1])+"_real.fits")
    lambda0_exc_exc_real_fits = fits.open("lambda0_exc_exc_z="+str(redshift_list[i+1])+"_real.fits")
    
    lambda0_1s_1s_real = lambda0_1s_1s_real_fits[0].data     
    lambda0_1s_exc_real = lambda0_1s_exc_real_fits[0].data     
    lambda0_exc_1s_real = lambda0_exc_1s_real_fits[0].data         
    lambda0_exc_exc_real = lambda0_exc_exc_real_fits[0].data 
        
    lambda0_1s_1s_imag_fits = fits.open("lambda0_1s_1s_z="+str(redshift_list[i+1])+"_imag.fits")
    lambda0_1s_exc_imag_fits = fits.open("lambda0_1s_exc_z="+str(redshift_list[i+1])+"_imag.fits")
    lambda0_exc_1s_imag_fits = fits.open("lambda0_exc_1s_z="+str(redshift_list[i+1])+"_imag.fits")
    lambda0_exc_exc_imag_fits = fits.open("lambda0_exc_exc_z="+str(redshift_list[i+1])+"_imag.fits")
    
    lambda0_1s_1s_imag = lambda0_1s_1s_imag_fits[0].data     
    lambda0_1s_exc_imag = lambda0_1s_exc_imag_fits[0].data     
    lambda0_exc_1s_imag = lambda0_exc_1s_imag_fits[0].data         
    lambda0_exc_exc_imag = lambda0_exc_exc_imag_fits[0].data 
    
    lambda0_1s_1s = lambda0_1s_1s_real*complex(1,0) + lambda0_1s_1s_imag*complex(0,1)
    lambda0_1s_exc = lambda0_1s_exc_real*complex(1,0) + lambda0_1s_exc_imag*complex(0,1)    
    lambda0_exc_1s = lambda0_exc_1s_real*complex(1,0) + lambda0_exc_1s_imag*complex(0,1)
    lambda0_exc_exc = lambda0_exc_exc_real*complex(1,0) + lambda0_exc_exc_imag*complex(0,1)    
                
    source_1s_fits = fits.open("source_1s_unpert_z="+str(redshift_list[i+1])+".fits")
    source_exc_fits = fits.open("source_exc_unpert_z="+str(redshift_list[i+1])+".fits")
        
    source_1s_unpert = source_1s_fits[0].data*complex(1,0)
    source_exc_unpert = source_exc_fits[0].data*complex(1,0)
    
    #source_1s_unpert = np.zeros(len(source_1s_unpert))
    #source_exc_unpert = np.zeros(len(source_exc_unpert))

    rho_exc_unpert = -np.dot( np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s, rho_1s_unpert) )
    rho_exc_unpert += - np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_unpert)
    
    print("rho_exc_unpert at z="+str(redshift_list[i]))
    print(rho_exc_unpert)
    
    
    

        
    if i == 0:

        #rho_1s_unpert = rho_1s_unpert.reshape(len(rho_1s_unpert),1)        
                    
        #hdu_rho_exc_unpert = fits.PrimaryHDU(np.abs(rho_exc_unpert) )
        #hdu_rho_exc_unpert.writeto("rho_exc_unpert_z="+str(redshift_list[0])+".fits", overwrite = True)    
        
        #rho_1s_unpert = rho_1s_unpert.reshape(len(rho_1s_unpert))
        print("nothing")
    
    else:
    

        # We need to define new matrices that are the coefficient matrices for our root solving method.
        # Our equation: rho_1s(n+1) - hf(rho_1s(n+1),z_n+1) - rho_1s(n) = 0
        # We now label our coefficients
        
        A = lambda0_1s_1s        
        A += -np.dot(lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), lambda0_exc_1s))
        A *= -1/((1+redshift)*H_param)
        print(" ")
        print("A matrix at z="+str(redshift_list[i]))
        print(A)
        
        B = source_1s_unpert - np.dot( lambda0_1s_exc, np.dot(np.linalg.inv(lambda0_exc_exc), source_exc_unpert))
        B *= -1/((1+redshift)*H_param)
        print(" ")
        print("B matrix at z="+str(redshift_list[i]))
        print(B)
                
                        
        rho_1s_prev = rho_1s_unpert # previous rho_1s

        # Calculating next rho_1s_unpert = (1- deltaz*A)^-1( rho_1s_prev + deltaz*B)
        
        matrix = np.identity(3) - z_index*A
        rho_1s_unpert = np.dot( np.linalg.inv(matrix), rho_1s_prev)
        rho_1s_unpert += np.dot( np.linalg.inv(matrix), z_index*B)            
                
                
        # Calculating the next rho_exc_unpert
                
        rho_exc_unpert = -np.dot( np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s, rho_1s_unpert) )
        rho_exc_unpert += - np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_unpert)        
        
        # Calculating deriv_func_unpert         
                
        deriv_func_unpert = np.dot( lambda0_1s_1s, rho_1s_unpert) + np.dot(lambda0_1s_exc, rho_exc_unpert) + source_1s_unpert
        deriv_func_unpert *= -1/((1+redshift)*H_param)

        
        #deriv_func_unpert = deriv_func_unpert.reshape( len(deriv_func_unpert),1)
        #rho_1s_unpert = rho_1s_unpert.reshape(len(rho_1s_unpert),1)        
        #rho_exc_unpert = rho_exc_unpert.reshape(len(rho_exc_unpert),1)        
                
        #hdu_deriv_func_unpert = fits.PrimaryHDU(deriv_func_unpert)
        #hdu_deriv_func_unpert.writeto("deriv_func_unpert_z="+str(redshift_list[i])+".fits", overwrite = True)            

        #hdu_rho_exc_unpert = fits.PrimaryHDU(rho_exc_unpert )
        #hdu_rho_exc_unpert.writeto("rho_exc_unpert_z="+str(redshift_list[i])+".fits", overwrite = True)            

        #hdu_rho_1s_unpert = fits.PrimaryHDU(rho_1s_unpert )
        #hdu_rho_1s_unpert.writeto("rho_1s_unpert_z="+str(redshift_list[i])+".fits", overwrite = True)    

        #deriv_func_unpert = deriv_func_unpert.reshape( len(deriv_func_unpert))
        #rho_1s_unpert = rho_1s_unpert.reshape(len(rho_1s_unpert))        
        #rho_exc_unpert = rho_exc_unpert.reshape(len(rho_exc_unpert))
        
        print(" ")
        print("deriv_func_unpert at z="+str(redshift_list[i]))
        print(deriv_func_unpert)
        
        Eigenvalue_Fis0.append(np.linalg.eig(A)[0][0])
        Eigenvalue_Fis1.append(np.linalg.eig(A)[0][1])
        
        deriv_func_Fis0.append(deriv_func_unpert[0])
        deriv_func_Fis1.append(deriv_func_unpert[1])
    	
    	      
    print(" ")    
    print("rho_1s_unpert at z="+str(redshift_list[i]))
    print(rho_1s_unpert)        
    
    print(" ")
    print("source_1s_unpert at z="+str(redshift_list[i]))
    print(source_1s_unpert)

    # Gets values to an array
        
    rho_1s_Fis0[i] = rho_1s_unpert[0]
    rho_1s_Fis1[i] = rho_1s_unpert[1]
    rho_1s_Kis2[i] = rho_1s_unpert[2]
    
    source_1s_Fis0[i] = source_1s_unpert[0]
    source_1s_Fis1[i] = source_1s_unpert[1]

    source_exc_2s_Fis0[i] = source_exc_unpert[0]
    source_exc_2s_Fis1[i] = source_exc_unpert[1]
    source_exc_2p_half_Fis0[i] = source_exc_unpert[3]
    source_exc_2p_half_Fis1[i] = source_exc_unpert[4]
    source_exc_2p_3half_Fis0[i] = source_exc_unpert[6]
    source_exc_2p_3half_Fis1[i] = source_exc_unpert[7] 
       
    rho_1s_Fis0_val = rho_1s_unpert[0]
    rho_1s_Fis1_val = rho_1s_unpert[1]
    rho_1s_Kis2_val = rho_1s_unpert[2]               
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
# Note that X_1s = sqrt(2*F+1) * rho(F,F^')          
plt.title("X_1s1/2")      
plt.plot(redshift_list[0:len(redshift_list)-1], rho_1s_Fis0, label = "F=F'=0")    
plt.plot(redshift_list[0:len(redshift_list)-1], np.sqrt(3)*rho_1s_Fis1, label = "F=F'=1")
plt.legend()
plt.show()

plt.title("Source 1s Unperturbed (K=0)")
plt.plot(redshift_list[0:len(redshift_list)-1], source_1s_Fis0, label = "1s_1/2 F=F'=0")
plt.plot(redshift_list[0:len(redshift_list)-1], source_1s_Fis1, label = "1s_1/2 F=F'=1")
plt.legend()
plt.show()

plt.title("Source Exc Unperturbed (K=0)")
plt.plot(redshift_list[0:len(redshift_list)-1], source_exc_2s_Fis0, label = "2s_1/2 F=F'=0")
plt.plot(redshift_list[0:len(redshift_list)-1], source_exc_2s_Fis1, label = "2s_1/2 F=F'=1")
plt.plot(redshift_list[0:len(redshift_list)-1], source_exc_2p_half_Fis0, label = "2p_1/2 F=F'=0")
plt.plot(redshift_list[0:len(redshift_list)-1], source_exc_2p_half_Fis1, label = "2p_1/2 F=F'=1")
plt.plot(redshift_list[0:len(redshift_list)-1], source_exc_2p_3half_Fis0, label = "2p_3/2 F=F'=1")
plt.plot(redshift_list[0:len(redshift_list)-1], source_exc_2p_3half_Fis1, label = "2p_3/2 F=F'=2")
plt.legend()
plt.show()

plt.title("A matrix Eigenvalue")
plt.plot(redshift_list[1:len(redshift_list)-1], Eigenvalue_Fis0, label = "F=F'=0")
plt.plot(redshift_list[1:len(redshift_list)-1], Eigenvalue_Fis1, label = "F=F'=1")
plt.legend()
plt.show()

# d rho_1s/dz = F(rho_1s,z)
plt.title("F(rho_1s,z)")
plt.plot(redshift_list[1:len(redshift_list)-1], deriv_func_Fis0, label = "F=F'=0")
plt.plot(redshift_list[1:len(redshift_list)-1], deriv_func_Fis1, label = "F=F'=1")
plt.legend()
plt.show()




    
     
