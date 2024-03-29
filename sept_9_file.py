import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import fractions
import time
from astropy.io import fits
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
from scipy.special import genlaguerre, gamma, hyp2f1, factorial, gammaln
from sympy import N



'''
Want to build a primer that dictates in our matrix which value of the quantum numbers we are trying to associate
'''

# Constants to be used in the cgs unit system (centimeters-grams-seconds)

c = 3e10 # Speed of light ( cm/s )
h = 6.626e-27 # Planck constant ( erg*s or cm^2 g / s )
hbar = h / (2*np.pi) # Reduced Planck constant in erg*s
e0 = 4.8032e-10 # One unit of electric charge   ( cm^3/2 * g^1/2 / s )
m_electron = 9.10938e-28 # Electron rest mass ( grams )
eV_to_ergs = 1.60218e-12 # eV to ergs conversion 
kB = 1.3807e-16 # Boltzmann constant (cm^2 g / s^2 K)

# Variables to be determined before running the program

T = 3000 # Temperature ( Kelvin )
mag_field = 1e-12 # Magnetic field strength ( Gauss )

# Quantities calculated using the above constants of nature

Bohr_radius = hbar**2 / (m_electron * e0**2 ) # Bohr radius ( cm )
larmor_freq = 1.3996e6 * mag_field # Lahmor frequency ( s^-1 )


# Quantum numbers

S = 0.5 # Electron spin quantum number
I = 0.5 # Nuclear spin quantum number


# Values considered when summing over quantum numbers to determine Lmabdafortenight

                                                                                                          
numN = 3 # Largest N value considered. Will go from 1 to Nmax.
numL = numN # Largest L value considered.
numJ = 2 # Number of allowed J values other than the L=0 case which there is only one.
numK = 2 # Sum from 0 to 2
numF = 2 # Number of allowed F values given a value of J.

# We want to seperate matrices for the 1s term and the excited states.


# We want to develop a program where we can index our matrices correctly.

# Creating a text file to put printed output



'''
Arrays for the unperturbed evolution matrix (Lambda0), the perturbed evolution matrix corresponding to K=0 (L0),
and the perturbed evolution matrix corresponding to K=2 (L2).

The structure of each matrix is the following:

Matrix(N0, L0, J0, K0, F0, F1, N1, L1, J1, K1, F2, F3)

'''


Lambda0 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)

L0 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)

L2 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)



'''
We want to define the matrix structure for the density matrix and the source function so that we can compute matrix
multiplication.
'''

density_matrix = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)
source_matrix = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)


'''

We want to theoretically define the energy of a Hydrogen atom that we will use as input
for other sections of this code. We want to incorporate corrections due to the Hyperfine 
structure of the atom along with other effects. 

Inputs: n, l, J, I, F
Output: Energy of a given level

'''

def energy_noF(N, L, J):

	# Unperturbed energy of the Hydrogen atom
	energy_0th = -13.6*eV_to_ergs / N**2

	# NOTE: Need to add perturbed terms to this quantity later.

	return energy_0th




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
		rad_field = (1/np.sqrt(2)) * weight * x * phase_deriv*Theta_2

	else:
		rad_field = 0

	return rad_field
	
		

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
		F = F1
		F_prime = F2

		n_l = n1
		l_l = l1
		J_l = J1
		K_l = K1
		F_l = F1
		F_lprime = F2

		# Calculating the appropriate Einstein coefficients

		B_Einstein = 32*np.pi**4 / (3*h**2*c)
		B_Einstein *= np.abs( dipole_element(n_l, n, l_l, l, J_l, J) )**2
		#print("This is the B Einstein coefficient:", B_Einstein)

		term1 = (2*J_l + 1)*B_Einstein # Prefactor to the sum


		term2 = 0 # Value of the sum across K_r 
	
		# Computing the sum across K_r from 0 to 2 where Q_r=0 is fixed.

		temp = np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_l+1)*(2*F_lprime+1)*(2*K+1)*(2*K_l+1)*(2*Kr+1))
		temp *= (-1)**(K_l + F_lprime - F_l)
		temp *= float(wigner_9j(F, F_l, 1, F_prime, F_lprime, 1, K, K_l, Kr) ) * float(wigner_6j(J, J_l, 1, F_l, F, I))
		temp *= float(wigner_6j(J, J_l, 1, F_lprime, F_prime, I) ) * float(wigner_3j(K, K_l, Kr, 0, 0, 0) )
		temp *= rad_field_tensor(Kr, n0, n1, l0, l1, J0, J1, T, pert_index)

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

		#print("This is the B Einstein coefficient:", B_Einstein)

		# Computing the sum across K_r from 0 to 2 where Q_r=0 is fixed.


		temp = np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_u+1)*(2*F_uprime+1)*(2*K+1)*(2*K_u+1)*(2*Kr+1))
		temp *= (-1)**(Kr+K_u+F_uprime-F_u)
		temp *= float( wigner_9j(F, F_u, 1, F_prime, F_uprime, 1, K, K_u, Kr))
		temp *= float( wigner_6j(J_u, J, 1, F, F_u, I) )
		temp *= float( wigner_6j(J_u, J, 1, F_prime, F_uprime, I) )
		temp *= float( wigner_3j(K, K_u, Kr, 0, 0, 0) )
		temp *= rad_field_tensor(Kr, n0, n1, l0, l1, J0, J1, T, pert_index)

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
							term2 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_u, T, pert_index)
							#print("This is term2")
							#print(term2)
						if F1 == F3:
							term3 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
							term3 *= (-1)**(F2 - F1 + K + K_prime + Kr)
							term3 *= float( wigner_6j(J, J, Kr, F2, F0, I) )
							term3 *= float( wigner_6j(K, K_prime, Kr, F2, F0, F1) )
							term3 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_u, T, pert_index)
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

							freq = energy_noF(n, l, J) - energy_noF(n_level, l_level, J_l)
							freq = np.abs(freq) / h

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
							term2 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_l, T, pert_index)
							#print("This is term2")
							#print(term2)
	
						if F1 == F3:
							term3 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
							term3 *= (-1)**(F2 - F1 + K + K_prime + Kr)
							term3 *= float( wigner_6j(J, J, Kr, F2, F0, I) )
							term3 *= float( wigner_6j(K, K_prime, Kr, F2, F0, F1) )
							term3 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_l, T, pert_index)
							#print("This is term3")
							#print(term3)
						
						total_term += term1*(term2+term3) # Summing each term.


	
	return total_term

'''
We now want to input terms that contribute to the source function that we need to calculate.
'''





'''
Portion of the code that organizes values from each of the seven functions: Nhat, T's, and R's into their respective matrices. Note
that 

1) Lambda0 corresponds wih the unpreturbed matrix.

2) L0 corresponds with the preturbed matrix, but only the K_r = 0 portion.

2) L2 corresponds with the preturbed matrix, but only the K_r = 2 portion.


One caviot: The 0th terms correspond with with the time derivative of the density matrix on the LHS of equation 7.69 while the 1th terms
correspond with the RHS of equation 7.69.

d/dt rho(N0, L0, J0, I, K0, Q0, F0, F1)  =  (Sum of quantum numbers) Lambda(N0,L0, J0, I, K0, Q0, F0, F1, N1, L1, J1, I, K1, Q1, F2, F3) * rho(N1, L1, J1, I, K1, Q1, F2, F3) 
'''

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
                                                            
                                                            print("RA_unpert: "+str(RA_unpert), file=output_file)
                                                            print("RS_unpert: "+str(RS_unpert), file=output_file)
                                                            print("RE_total: "+str(RE_total), file=output_file)
                                                            
                                                            print("RA_pert_0: "+str(RA_pert_0), file=output_file)
                                                            print("RS_pert_0: "+str(RS_pert_0), file=output_file)
                                                            
                                                            Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += Nhat_total + RA_unpert + RS_unpert + RE_total
                                                            L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += RA_pert_0 + RS_pert_0
                                                            
                                                        if Kr==2:
                                                            
                                                            RA_pert_2 = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                            RS_pert_2 = R_S(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                            
                                                            print("RA_pert_2: " +str(RA_pert_2), file=output_file)
                                                            print("RS_pert_2: " +str(RS_pert_2), file=output_file)
                                                            
                                                            L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += RA_pert_2 + RS_pert_2
                                                            
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
                               
                                                                print("TA_unpert: "+str(TA_unpert), file=output_file)
                                                                print("TS_unpert: "+str(TS_unpert), file=output_file)
                                                                print("TE_total: "+str(TE_total), file=output_file)
                                                            
                                                                print("TA_pert_0: "+str(TA_pert_0), file=output_file)
                                                                print("TS_pert_0: "+str(TS_pert_0), file=output_file)
                                                            
                                                                if N1 < N0:
                                                                    Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_unpert
                                                                    L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_pert_0
                                                                elif N1 > N0:
                                                                    Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TE_total + TS_unpert
                                                                    L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TS_pert_0
                                                                    
                                                            elif Kr == 2:
                                                            
                                                                TA_pert_2 = T_A(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)
                                                                TS_pert_2 = T_S(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], True)                                                                                                      
 
                                                                
                                                                print("TA_pert_2: "+str(TA_pert_2), file=output_file)
                                                                print("TS_pert_2: "+str(TS_pert_2), file=output_file)    
                                                                
                                                                if N1 < N0:
                                                                    L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TA_pert_2
                                                                elif N1 > N0:
                                                                    L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] += TS_pert_2
                                                                    
                                                            print("", file = output_file)
                                                            print("Lambda0: "+str(Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)
                                                            print("L0: "+str(L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)                                                                                                                        
                                                            print("L2: "+str(L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3]), file=output_file)                                                    
                                                          
                                                                
output_file.close()
                                                                                                                       
                                                                
# We want to create a mask that dictates which values we can use or not use for our calculation


for N0 in range(1,numN+1):
    for N1 in range(1,numN+1):
        for l0 in range(numN):
            for l1 in range(numN):
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
                                                    
                                                    if l0 > N0-1 or l1 >N1-1:
                                                        
                                                        Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] = np.nan
                                                        L0[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] = np.nan
                                                        L2[N0, l0, j0, k0, f0, f1, N1, l1, j1, k1, f2, f3] = np.nan
                                                        
                                                    if  N0 > 1 and l0 == 0 and N1>1 and l1>0:
                                                        Lambda0[N0, l0, 1, k0, f0, f1, N1, l1, j1, k1, f2, f3] = np.nan
                                                        L0[N0, l0, 1, k0, f0, f1, N1, l1, j1, k1, f2, f3] = np.nan
                                                        L2[N0, l0, 1, k0, f0, f1, N1, l1, j1, k1, f2, f3] = np.nan
                                                        
                                                    if  N0 > 1 and l0 > 0 and N1>1 and l1 == 0:
                                                        Lambda0[N0, l0, j0, k0, f0, f1, N1, l1, 1, k1, f2, f3] = np.nan
                                                        L0[N0, l0, j0, k0, f0, f1, N1, l1, 1, k1, f2, f3] = np.nan
                                                        L2[N0, l0, j0, k0, f0, f1, N1, l1, 1, k1, f2, f3] = np.nan
                                                        
                                                    if  N0 > 1 and l0 == 0 and N1>1 and l1 == 0:
                                                        Lambda0[N0, l0, 1, k0, f0, f1, N1, l1, 1, k1, f2, f3] = np.nan
                                                        L0[N0, l0, 1, k0, f0, f1, N1, l1, 1, k1, f2, f3] = np.nan
                                                        L2[N0, l0, 1, k0, f0, f1, N1, l1, 1, k1, f2, f3] = np.nan                                                        
             

			

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

# Next, we want to make each of these arrays into a 2-dim array so we can use matrix multiplication

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

# We to mask all the invalid values in our array

Lambda0_1s_1s_masked = np.ma.masked_invalid(Lambda0_1s_1s)
Lambda0_1s_exc_masked = np.ma.masked_invalid(Lambda0_1s_exc)
Lambda0_exc_1s_masked = np.ma.masked_invalid(Lambda0_exc_1s)
Lambda0_exc_exc_masked = np.ma.masked_invalid(Lambda0_exc_exc)

L0_1s_1s_masked = np.ma.masked_invalid(Lambda0_1s_1s)
L0_1s_exc_masked = np.ma.masked_invalid(Lambda0_1s_exc)
L0_exc_1s_masked = np.ma.masked_invalid(Lambda0_exc_1s)
L0_exc_exc_masked = np.ma.masked_invalid(Lambda0_exc_exc)

L2_1s_1s_masked = np.ma.masked_invalid(Lambda0_1s_1s)
L2_1s_exc_masked = np.ma.masked_invalid(Lambda0_1s_exc)
L2_exc_1s_masked = np.ma.masked_invalid(Lambda0_exc_1s)
L2_exc_exc_masked = np.ma.masked_invalid(Lambda0_exc_exc)

# Saving into fits files for Lambda0

hdu1 = fits.PrimaryHDU(np.abs(Lambda0_1s_1s))
hdu1.writeto("lambda0_1s_1s.fits", overwrite = True)

hdu2 = fits.PrimaryHDU(np.abs(Lambda0_1s_exc))
hdu2.writeto("lambda0_1s_exc.fits", overwrite = True)

hdu3 = fits.PrimaryHDU(np.abs(Lambda0_exc_1s))
hdu3.writeto("lambda0_exc_1s.fits", overwrite = True)

hdu4 = fits.PrimaryHDU(np.abs(Lambda0_exc_exc))
hdu4.writeto("lambda0_exc_exc.fits", overwrite = True)

# Saving into fits files for L0

hdu5 = fits.PrimaryHDU(np.abs(L0_1s_1s))
hdu5.writeto("L0_1s_1s.fits", overwrite = True)

hdu6 = fits.PrimaryHDU(np.abs(L0_1s_exc))
hdu6.writeto("L0_1s_exc.fits", overwrite = True)

hdu7 = fits.PrimaryHDU(np.abs(L0_exc_1s))
hdu7.writeto("L0_exc_1s.fits", overwrite = True)

hdu8 = fits.PrimaryHDU(np.abs(L0_exc_exc))
hdu8.writeto("L0_exc_exc.fits", overwrite = True)

# Saving into fits files for L2

hdu9 = fits.PrimaryHDU(np.abs(L2_1s_1s))
hdu9.writeto("L2_1s_1s.fits", overwrite = True)

hdu10 = fits.PrimaryHDU(np.abs(L2_1s_exc))
hdu10.writeto("L2_1s_exc.fits", overwrite = True)

hdu11 = fits.PrimaryHDU(np.abs(L2_exc_1s))
hdu11.writeto("L2_exc_1s.fits", overwrite = True)

hdu12 = fits.PrimaryHDU(np.abs(L2_exc_exc))
hdu12.writeto("L2_exc_exc.fits", overwrite = True)



# Next, take the inverse of each matrix just in case we need to use them later

Lambda0_1s_1s_inv = np.linalg.pinv(Lambda0_1s_1s)
Lambda0_1s_exc_inv = np.linalg.pinv(Lambda0_1s_exc)
Lambda0_exc_1s_inv = np.linalg.pinv(Lambda0_exc_1s)
Lambda0_exc_exc_inv = np.linalg.pinv(Lambda0_exc_exc)

L0_1s_1s_inv = np.linalg.pinv(L0_1s_1s)
L0_1s_exc_inv = np.linalg.pinv(L0_1s_exc)
L0_exc_1s_inv = np.linalg.pinv(L0_exc_1s)
L0_exc_exc_inv = np.linalg.pinv(L0_exc_exc)

L2_1s_1s_inv = np.linalg.pinv(L2_1s_1s)
L2_1s_exc_inv = np.linalg.pinv(L2_1s_exc)
L2_exc_1s_inv = np.linalg.pinv(L2_exc_1s)
L2_exc_exc_inv = np.linalg.pinv(L2_exc_exc)


# First, create a density and source matrix that represents the system

density_matrix = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)
source_matrix = np.zeros( (numN+1, numL, numJ, numK, numF, numF), dtype = np.complex)

density_matrix_1s = density_matrix[1:2,0:1,:,:,:,:]
density_matrix_exc = density_matrix[2:numN+1,:,:,:,:,:]

source_matrix_1s = source_matrix[1:2,0:1,:,:,:,:]
source_matrix_exc = source_matrix[2:numN+1,:,:,:,:,:]

# Now, we can produce the psuedo inverse of these matricies.

density_matrix_1s = density_matrix_1s.reshape(N_1s)
density_matrix_exc = density_matrix_exc.reshape(N_exc)

source_matrix_1s = source_matrix_1s.reshape(N_1s)
source_matrix_exc = source_matrix_exc.reshape(N_exc)

# Let's run a test run for some given source function

for i in range(N_exc):
    source_matrix_exc[i] = i
    
# Let's calculate the density matrix for one run

density_matrix_exc = - Lambda0_exc_exc_inv * source_matrix_exc
