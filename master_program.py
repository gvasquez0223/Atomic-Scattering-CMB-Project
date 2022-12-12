import sys
import subprocess
import numpy as np
from astropy.io import fits

# Constants

c = 3e10 # Speed of light ( cm/s )


# We want to make a module that imports the T(k) values for a given k. We will then loop over it
# and then perform our RK4 program.

redshift_list = np.loadtxt(open("redshift_list.txt"))

z_index =  2*(redshift_list[0]-redshift_list[1])

# Current wave number 

wave_num = 1.042057218184e-05


thermo_array = np.loadtxt(open("thermodynamics.dat"))
tk_array = np.loadtxt(open("tk_z="+str(redshift_list[0])+".dat"))

found_kline = False
found_zline = False



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
	
while found_zline == False:
	for line in range(len(thermo_array)):
		
		if thermo_array[line][1] == redshift_list[0]:
			print("The redshift in the file is "+str(thermo_array[line][1]))
			print("The redshift is in line "+str(line))
			
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

for i in range(len(redshift_list)-2):

	# We compute the unperturbed piece in our equations
		
	lambda0_1s_1s_fits = fits.open("lambda0_1s_1s_z="+str(redshift_list[i+1])+".fits")
	lambda0_1s_exc_fits = fits.open("lambda0_1s_exc_z="+str(redshift_list[i+1])+".fits")
	lambda0_exc_1s_fits = fits.open("lambda0_exc_1s_z="+str(redshift_list[i+1])+".fits")
	lambda0_exc_exc_fits = fits.open("lambda0_exc_exc_z="+str(redshift_list[i+1])+".fits")
	
	lambda0_1s_1s = lambda0_1s_1s_fits[0].data 	
	lambda0_1s_exc = lambda0_1s_exc_fits[0].data 	
	lambda0_exc_1s = lambda0_exc_1s_fits[0].data 		
	lambda0_exc_exc = lambda0_exc_exc_fits[0].data 
		
	source_1s_fits = fits.open("source_1s_unpert_z="+str(redshift_list[i+1])+".fits")
	source_exc_fits = fits.open("source_exc_unpert_z="+str(redshift_list[i+1])+".fits")
		
	source_1s_unpert = source_1s_fits[0].data
	source_exc_unpert = source_exc_fits[0].data


	rho_exc_unpert = -np.dot( np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s, rho_1s_unpert) )
	rho_exc_unpert += - np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_unpert)
	
	print(rho_exc_unpert)
	
	

		
	if i == 0:

		rho_1s_unpert = rho_1s_unpert.reshape(len(rho_1s_unpert),1)		
					
		hdu_rho_exc_unpert = fits.PrimaryHDU(np.abs(rho_exc_unpert) )
		hdu_rho_exc_unpert.writeto("rho_exc_unpert_z="+str(redshift_list[0])+".fits", overwrite = True)	
		
		rho_1s_unpert = rho_1s_unpert.reshape(len(rho_1s_unpert))
	
	else:
	
		# We label the derivative function
		
		deriv_func_unpert = np.dot( lambda0_1s_1s, rho_1s_unpert) + np.dot(lambda0_1s_exc, rho_exc_unpert) + source_1s_unpert
		deriv_func_unpert *= -1/((1+redshift)*H_param)
		
		rho_1s_unpert += z_index*deriv_func_unpert
		
		rho_exc_unpert = -np.dot( np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s, rho_1s_unpert) )
		rho_exc_unpert += - np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_unpert)
		
		deriv_func_unpert = deriv_func_unpert.reshape( len(deriv_func_unpert),1)
		rho_1s_unpert = rho_1s_unpert.reshape(len(rho_1s_unpert),1)		
		rho_exc_unpert = rho_exc_unpert.reshape(len(rho_exc_unpert),1)		
				
		hdu_deriv_func_unpert = fits.PrimaryHDU(deriv_func_unpert)
		hdu_deriv_func_unpert.writeto("deriv_func_unpert_z="+str(redshift_list[i])+".fits", overwrite = True)			

		hdu_rho_exc_unpert = fits.PrimaryHDU(rho_exc_unpert )
		hdu_rho_exc_unpert.writeto("rho_exc_unpert_z="+str(redshift_list[i])+".fits", overwrite = True)			

		hdu_rho_1s_unpert = fits.PrimaryHDU(rho_1s_unpert )
		hdu_rho_1s_unpert.writeto("rho_1s_unpert_z="+str(redshift_list[i])+".fits", overwrite = True)	

		deriv_func_unpert = deriv_func_unpert.reshape( len(deriv_func_unpert))
		rho_1s_unpert = rho_1s_unpert.reshape(len(rho_1s_unpert))		
		rho_exc_unpert = rho_exc_unpert.reshape(len(rho_exc_unpert))		
	
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
for i in range(len(redshift_list)-2):
	for j in range(3):

		
		
		lambda0_1s_1s_fits = fits.open("lambda0_1s_1s_z="+str(redshift_list[i+j])+".fits")
		lambda0_1s_exc_fits = fits.open("lambda0_1s_exc_z="+str(redshift_list[i+j])+".fits")
		lambda0_exc_1s_fits = fits.open("lambda0_exc_1s_z="+str(redshift_list[i+j])+".fits")
		lambda0_exc_exc_fits = fits.open("lambda0_exc_exc_z="+str(redshift_list[i+j])+".fits")
	
		lambda0_1s_1s = lambda0_1s_1s_fits[0].data 	
		lambda0_1s_exc = lambda0_1s_exc_fits[0].data 	
		lambda0_exc_1s = lambda0_exc_1s_fits[0].data 		
		lambda0_exc_exc = lambda0_exc_exc_fits[0].data 
		
		source_1s_fits = fits.open("source_1s_unpert_z="+str(redshift_list[i+j])+".fits")
		source_exc_fits = fits.open("source_exc_unpert_z="+str(redshift_list[i+j])+".fits")
		
		source_1s_unpert = source_1s_fits[0].data
		source_exc_unpert = source_exc_fits[0].data
		
		# Computes and saves the first rho_exc to a Fits file
		
		if i == 0:
		
			rho_exc_unpert = -np.dot( np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s, rho_1s_unpert) )
			rho_exc_unpert += - np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_unpert)
			
			hdu_rho_exc_unpert = fits.PrimaryHDU(np.abs(rho_exc_unpert) )
			hdu_rho_exc_unpert.writeto("rho_exc_unpert_z="+str(redshift_list[0])+".fits", overwrite = True)				
		
		# Computing the Runge-Kutta coefficients depending on which file we are in by index
		
		if j == 0:
			k1 = np.dot(lambda0_1s_1s, rho_1s_unpert)
			k1 += - np.dot( np.dot(  np.dot(lambda0_1s_exc, np.linalg.inv(lambda0_exc_exc)), lambda0_exc_1s), rho_1s_unpert) 
			k1 += - np.dot( lambda0_1s_exc, np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_unpert) ) 
			k1 += source_1s_unpert 
			
		elif j == 1:
			k2 = np.dot(lambda0_1s_1s, rho_1s_unpert)
			k2 += - np.dot(np.dot( np.dot(lambda0_1s_exc, np.linalg.inv(lambda0_exc_exc)), lambda0_exc_1s), rho_1s_unpert + z_index*k1/2)
			k2 += - np.dot( lambda0_1s_exc, np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_unpert) ) 
			k2 += source_1s_unpert 

			k3 = np.dot(lambda0_1s_1s, rho_1s_unpert)
			k3 += - np.dot(np.dot( np.dot(lambda0_1s_exc, np.linalg.inv(lambda0_exc_exc)), lambda0_exc_1s), rho_1s_unpert + z_index*k2/2) 
			k3 += - np.dot( lambda0_1s_exc, np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_unpert) ) 
			k3 += source_1s_unpert
					
		elif j == 2:

			k4 = np.dot(lambda0_1s_1s, rho_1s_unpert)
			k4 += - np.dot(np.dot( np.dot(lambda0_1s_exc, np.linalg.inv(lambda0_exc_exc)), lambda0_exc_1s), rho_1s_unpert + z_index*k3) 
			k4 += - np.dot( lambda0_1s_exc, np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_unpert) ) 
			k4 += source_1s_unpert 
			
	rho_1s_unpert += (k1+2*k2+2*k3+k4)/6

	rho_exc_unpert = -np.dot( np.linalg.inv(lambda0_exc_exc), np.dot( lambda0_exc_1s, rho_1s_unpert) )
	rho_exc_unpert += - np.dot( np.linalg.inv(lambda0_exc_exc), source_exc_unpert)		

	# Saves the next iteration to a Fits file for both the density matrix components
		
	hdu_rho_1s_unpert = fits.PrimaryHDU(np.abs(rho_1s_unpert) )
	hdu_rho_1s_unpert.writeto("rho_1s_unpert_z="+str(redshift_list[i+1])+".fits", overwrite = True)
	
	hdu_rho_exc_unpert = fits.PrimaryHDU(np.abs(rho_exc_unpert) )
	hdu_rho_exc_unpert.writeto("rho_exc_unpert_z="+str(redshift_list[i+1])+".fits", overwrite = True)	
		
	#rho_exc_unpert = - np.dot(np.linalg.inv(lambda0_exc_exc), ( np.dot(lambda0_exc_1s,rho_1s_unpert + source_exc_unpert) ) )
					
'''	
			
		
	
	




	
	 
