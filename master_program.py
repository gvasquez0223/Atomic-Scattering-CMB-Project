import sys
import subprocess
import numpy as np

# We want to make a module that imports the T(k) values for a given k. We will then loop over it
# and then perform our RK4 program.

redshift_list = np.loadtxt(open("redshift_list.dat"))

# Asks the user to input their wave number 

wave_num = np.float(sys.argv1[1])

# Given the value of wave_num we find the line number on both the thermo array and tk array given wave_num.

for i in range(len(redshift_list)):

	thermo_array = np.loadtxt(open("thermo.dat"))
	tk_array = np.loadtxt(open("tk_z="+str(redshift_list[i])+".dat"))
	
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
	

		z_error = 1e-3

	while found_zline == False:
		for line in range(len(thermo_array)):
		
			if thermo_array[line][1] == redshift:
				print("The redshift in the file is "+str(thermo_array[line][1]))
				print("The redshift is in line "+str(line))
			
				zline = line
				found_zline = True


	
	
	 
