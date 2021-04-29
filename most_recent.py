import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import fractions
import time
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
eV_to_J = 1.60218e-19
kB = 1.3806e-23

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
numK = 3 # Sum from 0 to 2
numF = 2 # Number of allowed F values given a value of J.


# We want to develop a program where we can index our matrices correctly.


'''
Arrays for the unperturbed evolution matrix (Lambda0), the perturbed evolution matrix corresponding to K=0 (L0),
and the perturbed evolution matrix corresponding to K=2 (L2).

The structure of each matrix is the following:

Matrix(N0, L0, J0, K0, F0, F1, N1, L1, J1, K1, F2, F3)

'''


Lambda0 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN, numL, numJ, numK, numF, numF), dtype = np.complex)

L0 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN, numL, numJ, numK, numF, numF), dtype = np.complex)

L2 = np.zeros( (numN+1, numL, numJ, numK, numF, numF, numN, numL, numJ, numK, numF, numF), dtype = np.complex)


'''

We want to theoretically define the energy of a Hydrogen atom that we will use as input
for other sections of this code. We want to incorporate corrections due to the Hyperfine 
structure of the atom along with other effects. 

Inputs: n, l, J, I, F
Output: Energy of a given level

'''

def energy_noF(N, L, J):

	# Unperturbed energy of the Hydrogen atom
	energy_0th = -13.6*eV_to_J / N**2

	# NOTE: Need to add perturbed terms to this quantity later.

	return energy_0th




def energy(N, L, J, I, F):


	# Unperturbed energy of the Hydrogen atom
	energy_0th = -13.6*eV_to_J / N**2

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

def dipole_element(n0, n1, l0, l1, J0, J1):

	print("Dipole_term")
	print(n0, n1, l0, l1, J0, J1)

	# We need to determine which set of quantum numbers is the upper or lower state.
	# Also computes the prefactor.

	if n0 < n1:
		term1 = (-1)**(l0+S+J1+1)
		term1 *= np.sqrt((2*J1+1)*(2*l0+1))
		term1 *= float(wigner_6j(l0, l1, 1, J1, J0, S) )

	elif n0 > n1:
		term1 = (-1)**(l1+S+J0+1)
		term1 *= np.sqrt((2*J0+1)*(2*l1+1))
		term1 *= float(wigner_6j(l1, l0, 1, J0, J1, S) )

	elif n0 == n1:
		term1 = 0

	# Calculation to determine <n',l-1|| vec{d} ||n,l>. First, we need to find
	# which value of l is larger.

	if l0 == l1-1 and n0 != n1:

		l = l1
		n = n1
		n_prime = n0

		term2 = np.sqrt(l)
		term2 *= (-1)**(n_prime-1) / (4*factorial(2*l-1))
		term2 *= np.exp( 0.5*gammaln(n+l+1) + 0.5*gammaln(n_prime+l) - 0.5*gammaln(n-l) - 0.5*gammaln(n_prime-l+1) )
		term2 *= (4*n*n_prime)**(l+1) * (n-n_prime)**(n+n_prime-2*l-2) / (n+n_prime)**(n+n_prime)
		term2 *= ( hyp2f1(-n+l+1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2 ) - (n-n_prime)**2/(n+n_prime)**2 * hyp2f1(-n+l-1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2) )


	elif l0 == l1+1 and n0 != n1:

		l = l0
		n = n0
		n_prime = n1

		term2 = np.sqrt(l)
		term2 *= (-1)**(n_prime-1) / (4*factorial(2*l-1))
		term2 *= np.exp( 0.5*gammaln(n+l+1) + 0.5*gammaln(n_prime+l) - 0.5*gammaln(n-l) - 0.5*gammaln(n_prime-l+1) )
		term2 *= (4*n*n_prime)**(l+1) * (n-n_prime)**(n+n_prime-2*l-2) / (n+n_prime)**(n+n_prime)
		term2 *= ( hyp2f1(-n+l+1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2 ) - (n-n_prime)**2/(n+n_prime)**2 * hyp2f1(-n+l-1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2) )

	else:

		term2 = 0

	return Bohr_radius*e0*term1*term2

'''
def dipole_element(n0, n1, l, lprime, J0, J1):

	e0 = 1.602e-19 # Electron charge

	# Computing the second term before determining <alpha J ||vec{d}||alpha' J'>
	term1 = e0*(-1)**(J1+1)*np.sqrt(2*J1+1)
	term1 *= float(wigner_3j(J0, J1, 1, 0, 0, 0))

	# Determines which l value is the lower one or the higher

	if l == (lprime - 1):
		l0 = l
		l1 = lprime
	elif l == (lprime + 1):
		l0 = lprime
		l1 = l
	# Computes the overlap integral
	if l0 == l1 + 1 or l0 == l1 - 1:

		term2 = (-1)**(n1-1)/(4* factorial(2*l0-1))
		term2 *= np.exp( 0.5*gammaln(n0+l0) + 0.5*gammaln(n1+l0-1) - 0.5*gammaln(n0-l0-1) - 0.5*gammaln(n1-1) )
		term2 *= (4*n0*n1)**(l0+1) * (n0-n1)**(n0+n1-2*l0-2) / (n0+n1)**(n0+n1)
		term2 *= ( hyp2f1(-n0+l0+1, -n1+l0, 2*l0, -4*n0*n1/(n0-n1)**2 ) - (n0-n1)**2/(n0+n1)**2 * hyp2f1(-n0+l0-1, -n1+l0, 2*l0, -4*n0*n1/(n0-n1)**2 ) )

	elif:
		term2 = 0

	return term1*term2
		

'''

	
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

	# Define a blackbody

	weight = 2*h*freq**3/c**2
	x = h*freq/(kB*T) # Ratio of photon energy with thermal energy

	phase_space = np.exp(-x)/(1-np.exp(-x))
	phase_deriv = -np.exp(-x)/(1-np.exp(-x))**2

	# The variable "pert_index" let's us know if we want the perturbed or unperturbed value.
	# If pert_index = False, then rad_field_tensor gives unperturbed value while pert_index = True
	# gives the 1st order term. 

	# Also, the value of K gives us which rad_field_Tensor we are interested in. Only the K=0,2 cases
	# are nonzero.

	if K==0 and pert_index == False:
		rad_field = weight*phase_space

	elif K==0 and pert_index == True:
		rad_field = weight*x*phase_deriv

	elif K==2 and pert_index == True:
		rad_field = (1/np.sqrt(2)) * weight * x * phase_deriv 

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
		term1 = term1/h

	
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
	
	# Calculating the appropriate Einstein coefficients

	B_Einstein = 32*np.pi**4 / (3*h**2*c)
	B_Einstein *= np.abs( dipole_element(n0, n1, l0, l1, J0, J1) )**2

	# We need to isolate which state is the lower state.

	if energy(n0, l0, J0, I, F0) == energy(n1, l1, J1, I, F1):

		term1 = 0
		term2 = 0 

	elif energy(n0, l0, J0, I, F0) > energy(n1, l1, J1, I, F1):
		
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

		term1 = (2*J_l + 1)*B_Einstein # Prefactor to the sum
		term2 = 0 # Value of the sum across K_r 
	
		# Computing the sum across K_r from 0 to 2 where Q_r=0 is fixed.

		temp = np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_l+1)*(2*F_lprime+1)*(2*K+1)*(2*K_l+1)*(2*Kr+1))
		temp *= (-1)**(K_l + F_lprime - F_l)
		temp *= float(wigner_9j(F, F_l, 1, F_prime, F_lprime, 1, K, K_l, Kr) ) * float(wigner_6j(J, J_l, 1, F_l, F, I))
		temp *= float(wigner_6j(J, J_l, 1, F_lprime, F_prime, I) ) * float(wigner_3j(K, K_l, Kr, 0, 0, 0) )
		temp *= rad_field_tensor(Kr, n0, n1, l0, l1, J0, J1, T, pert_index)

		term2 += temp

	elif energy(n0, l0, J0, I, F0) < energy(n1, l1, J1, I, F1):

		n = n1
		l = l1
		J = J1
		K = K1
		F = F2
		F_prime = F3

		n_l = n0
		l_l = l0
		J_l = J0
		K_l = K0
		F_l = F0
		F_lprime = F1

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




def T_S(n0, n1, l0, l1, J0, J1, K0, K1,Kr, F0, F1, F2, F3, pert_index):

	# Calculating the appropriate Einstein coefficients
	
	B_Einstein = 32*np.pi**4 / (3*h**2*c)
	B_Einstein *= np.abs( dipole_element(n0, n1, l0, l1, J0, J1) )**2

	# We need to isolate which state is the upper state.

	if energy(n0, l0, J0, I, F0) == energy(n1, l1, J1, I, F1):
	
		term1 = 0
		term2 = 0

	elif energy(n0, l0, J0, I, F0) > energy(n1, l1, J1, I, F1):
		
		n = n1
		l = l1
		J = J1
		K = K1
		F = F2
		F_prime = F3

		n_u = n0
		l_u = l0
		J_u = J0
		K_u = K0
		F_u = F0
		F_uprime = F1

		term1 = (2*J_u + 1)*B_Einstein # Prefactor to the sum
		term2 = 0 # Value of the sum across K_r 
	
		# Computing the sum across K_r from 0 to 2 where Q_r=0 is fixed.


		temp = np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_u+1)*(2*F_uprime+1)*(2*K+1)*(2*K_u+1)*(2*Kr+1))
		temp *= (-1)**(Kr+K_u+F_uprime-F_u)
		temp *= float( wigner_9j(F, F_u, 1, F_prime, F_uprime, 1, K, K_u, Kr))
		temp *= float( wigner_6j(J_u, J, 1, F, F_u, I) )
		temp *= float( wigner_6j(J_u, J, 1, F_prime, F_uprime, I) )
		temp *= float( wigner_3j(K, K_u, Kr, 0, 0, 0) )
		temp *= rad_field_tensor(Kr, n0, n1, l0, l1, J0, J1, T, pert_index)

		term2 += temp

	elif energy(n0, l0, J0, I, F0) < energy(n1, l1, J1, I, F1):

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

		term1 = (2*J_u + 1)*B_Einstein # Prefactor to the sum
		term2 = 0 # Value of the sum across K_r 
	
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

	# Calculating the appropriate Einstein Coefficients

	A_Einstein = 64*np.pi**4 * freq**3 / (3*h*c**3)
	A_Einstein *= np.abs( dipole_element(n0, n1, l0, l1, J0, J1) )**2

	# We need to determine which state is the upper state.

	if energy(n0, l0, J0, I, F0) == energy(n1, l1, J1, I, F1):
		
		term = 0

	if energy(n0, l0, J0, I, F0) > energy(n1, l1, J1, I, F1):
		
		n = n1
		l = l1
		J = J1
		K = K1
		F = F2
		F_prime = F3

		n_u = n0
		l_u = l0
		J_u = J0
		K_u = K0
		F_u = F0
		F_uprime = F1

		# Computing the term itself
		term = 0

		if K == K1:
		
			term = (2*J_u + 1)*A_Einstein
			term *= np.sqrt( (2*F+1)*(2*F_prime+1)*(2*F_u+1)*(2*F_uprime+1) )
			term *= (-1)**(1 + K + F_prime + F_uprime)
			term *= float( wigner_6j(F , F_prime, K, F_uprime, F_u, 1) )
			term *= float( wigner_6j(J_u , J, 1, F, F_u, I) )
			term *= float( wigner_6j(J_u, J, 1, F_prime, F_uprime, I) )
	

	elif energy(n0, l0, J0, I, F0) < energy(n1, l1, J1, I, F1):

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


	Nmax = 100 # Total number of different values of n we are considering.

	# Define 3 terms to make the calculation easier

	term1 = 0
	term2 = 0
	term3 = 0

	for n_level in range(Nmax+1):
		for l_level in range(n_level):

			# Composes allowed values of J given a value of L and S.
			# Allowed values: J = L- 1/2, L+ 1/2

			J_level = np.arange( np.abs(l_level - S), l_level + S+1, 1)

			for j_index in range(len(J_level)):

				J_u = J_level[j_index] # J_u value

				# Need to determine if a state is more energetic.

				if n_level > n:
					if l_level == l-1 or l_level == l or l_level == l+1:

						B_Einstein = 32*np.pi**4 / (3*h**2*c)
						B_Einstein *= np.abs( dipole_element(n_level, n, l_level, l, J_u, J) )**2
	

						
						term1 = B_Einstein
						term1 *= np.sqrt(3*(2*K+1)*(2*K_prime+1)*(2*Kr+1))
						term1 *= (-1)**(1+J_u-I+F0)
						term1 *= float( wigner_6j(J, J, Kr, 1, 1, J_u) )
						term1 *= float( wigner_3j(K, K_prime, Kr, 0,0, 0) )
	
						if F0 == F2:
							term2 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
							term2 *= float( wigner_6j(J, J, Kr, F3, F1, I) )
							term2 *= float( wigner_6j(K, K_prime, Kr, F3, F1, F0) )
							term2 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_u, T, pert_index)
						if F1 == F3:
							term3 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
							term3 *= (-1)**(F2 - F1 + K + K_prime + Kr)
							term3 *= float( wigner_6j(J, J, Kr, F2, F0, I) )
							term3 *= float( wigner_6j(K, K_prime, Kr, F2, F0, F1) )
							term3 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_u, T, pert_index)

		
	term1 *= (2*J + 1) # Multiplying by a prefactor
	
	return term1*(term2 + term3)



def R_E(n, l, J, I, K, K_prime, F0, F1, F2, F3):

	Nmax = 100 # Total number of principal quantum states being considered.

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

							A_Einstein = 64*np.pi**4 * freq**3 / (3*h*c**3)
							A_Einstein *= np.abs( dipole_element(n_level, n, l_level, l, J_l, J) )**2

							A_sum += A_Einstein # Sum each allowed Einstein-A coefficient.

	return A_sum



def R_S(n, l, J, I, K, K_prime, Kr, F0, F1, F2, F3, pert_index):

	Nmax = 100 # Total number of principal quantum states being considered.

	# Define 3 terms to make the calculations easier.

	term1 = 0
	term2 = 0
	term3 = 0

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

						# Calculate frequency

						freq = energy_noF(n, l, J) - energy_noF(n_level, l_level, J_l)
						freq = np.abs(freq) / h

						# Calculating allowed Einstein-B coefficients

						B_Einstein = 32*np.pi**4 / (3*h**2*c)
						B_Einstein *= np.abs( dipole_element(n_level, n, l_level, l, J_l, J) )**2
						
						term1 = B_Einstein
						term1 *= np.sqrt( 3*(2*K+1)*(2*K_prime+1)*(2*Kr+1) )
						term1 *= (-1)**(1+J_l - I + F0 +Kr)
						term1 *= float( wigner_6j(J, J, Kr, 1, 1, J_l) )
						term1 *= float( wigner_3j(K, K_prime, Kr, 0,0,0) )
	
						if F0 == F2:
							term2 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
							term2 *= float( wigner_6j(J, J, Kr, F3, F1, I) )
							term2 *= float( wigner_6j(K, K_prime, Kr, F3, F1, F0) )
							term2 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_l, T, pert_index)
	
						if F1 == F3:
							term3 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
							term3 *= (-1)**(F2 - F1 + K + K_prime + Kr)
							term3 *= float( wigner_6j(J, J, Kr, F2, F0, I) )
							term3 *= float( wigner_6j(K, K_prime, Kr, F2, F0, F1) )
							term3 *= rad_field_tensor(Kr, n, n_level, l, l_level, J, J_l, T, pert_index)

		
	term1 *= (2*J+1) # Prefactor we need to include
	
	return term1*(term2 + term3)



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

for N0 in range(1,numN+1):
	for N1 in range(1,numN+1):
		for l0 in range(N0):
			for l1 in range(N1):
			
				# Computing allowed values of J: J = L-0.5, L+0.5
				J0 = np.arange(np.abs(l0-S), l0+S+1, 1)
				J1 = np.arange(np.abs(l1-S), l1+S+1, 1)

				for j0 in range(len(J0)):
					for j1 in range(len(J1)):


						'''

						print(N0)
						print(N1)
						print(l0)
						print(l1)
						print(J0[j0])
						print(J1[j1])

						'''

						# Calculating allowed values of F: F = J-0.5, J+0.5
	
						F0 = np.arange(np.abs(J0[j0]-I), J0[j0]+I+1,1)
						F1 = F0
		
						F2 = np.arange(np.abs(J1[j1]-I), J1[j1]+I+1, 1)
						F3 = F2

						for K0 in range(numK):
							for K1 in range(numK):
								for Kr in range(numK):

									for f0 in range(len(F0)):
										for f1 in range(len(F1)):
											for f2 in range(len(F2)):
												for f3 in range(len(F3)):

													

													#print("Outside if statement")

													#print(N0,N1,l0,l1, J0[j0],J1[j1])
																														
													#print(K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3])
	
													#print( time.asctime(time.localtime(time.time())))

													if N0 == N1 and l0==l1 and J0[j0]==J1[j1]:

														#print("Inside if statement")

														#print( time.asctime(time.localtime(time.time())))

														#print(N0,N1,l0,l1, J0[j0],J1[j1])
														#print(K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3])

													

														Nhat_total = Nhat(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])
														Nhat_total *= -2*np.pi*complex(0,1)

														RA_unpert = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)

														RS_unpert = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)

														RE_total = R_E(N0, l0, J0[j0], I, K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])


														RA_pert_0 = R_A(N0, l0, J0[j0], I, K0, K1, 0, F0[f0], F1[f1], F2[f2], F3[f3], True)

														RS_pert_0 = R_S(N0, l0, J0[j0], I, K0, K1, 0, F0[f0], F1[f1], F2[f2], F3[f3], True)

														RA_pert_2 = R_A(N0, l0, J0[j0], I, K0, K1, 2, F0[f0], F1[f1], F2[f2], F3[f3], True)

														RS_pert_2 = R_S(N0, l0, J0[j0], I, K0, K1, 2, F0[f0], F1[f1], F2[f2], F3[f3], True)
	
	
														Lambda0[N0, l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += Nhat_total + RA_unpert + RS_unpert + RE_total


														L0[N0, l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += RA_pert_0 + RS_pert_0


														L2[N0, l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += RA_pert_2 + RS_pert_2


				


														

													# Computing the unpreturbed portion.

													# Nhat term
												



													# Unperturbed or total T terms. Only T_E has no perturbed term.

													TA_unpert = T_A(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)

													TS_unpert = T_S(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)

													TE_total = T_E(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])

													# Unperturbed or total R terms. Only R_E has no perturbed term.

													'''

													RA_unpert = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)

													RS_unpert = R_A(N0, l0, J0[j0], I, K0, K1, Kr, F0[f0], F1[f1], F2[f2], F3[f3], False)

													RE_total = R_E(N0, l0, J0[j0], I, K0, K1, F0[f0], F1[f1], F2[f2], F3[f3])
													'''

													# Compute the perturbed (K=0) terms

													TA_pert_0 = T_A(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, 0, F0[f0], F1[f1], F2[f2], F3[f3], True)

													TS_pert_0 = T_S(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, 0, F0[f0], F1[f1], F2[f2], F3[f3], True)
													'''

													RA_pert_0 = R_A(N0, l0, J0[j0], I, K0, K1, 0, F0[f0], F1[f1], F2[f2], F3[f3], True)

													RS_pert_0 = R_S(N0, l0, J0[j0], I, K0, K1, 0, F0[f0], F1[f1], F2[f2], F3[f3], True)
													'''

													# Compute the perturbed (K=2) terms

													TA_pert_2 = T_A(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, 2, F0[f0], F1[f1], F2[f2], F3[f3], True)

													TS_pert_2 = T_S(N0, N1, l0, l1, J0[j0], J1[j1], K0, K1, 2, F0[f0], F1[f1], F2[f2], F3[f3], True)

													'''

													RA_pert_2 = R_A(N0, l0, J0[j0], I, K0, K1, 2, F0[f0], F1[f1], F2[f2], F3[f3], True)

													RS_pert_2 = R_S(N0, l0, J0[j0], I, K0, K1, 2, F0[f0], F1[f1], F2[f2], F3[f3], True)
													'''

													'''
													We want to now sort out which terms correspond to which elements 
													of each matrix.
													'''

													# Lambda0 matrix elements

													#Lambda0[N0, l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += Nhat_total + RA_unpert + RS_unpert + RE_total

													if N1 < N0:
														Lambda0[N0,l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += TA_unpert
													elif N1 < N0:
														Lambda0[N0, l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += TE_total + TS_unpert


													# L0 matrix elements

													#L0[N0, l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += RA_pert_0 + RS_pert_0

													if N1 < N0:
														L0[N0,l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += TA_pert_0
													elif N1 < N0:
														L0[N0, l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += TS_pert_0

 							
													# L2 matrix elements

													#L2[N0, l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += RA_pert_2 + RS_pert_2

													if N1 < N0:
														L2[N0,l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += TA_pert_2
													elif N1 < N0:
														L2[N0, l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] += TS_pert_2


# Setting each nonphysical value to np.nan

'''
for N0 in range(1,numN+1):
	for N1 in range(1, num+1):
		for l0 in range(numN):
			for l1 in range(numN):

				# Computing allowed values of J: J = L-0.5, L+0.5
				J0 = np.arange(np.abs(l0-S), l0+S+1, 1)
				J1 = np.arange(np.abs(l1-S), l1+S+1, 1)

				for j0 in range(len(J0)):
					for j1 in range(len(J1)):

						# Calculating allowed values of F: F = J-0.5, J+0.5

	
						F0 = np.arange(np.abs(J0[j0]-I), J0[j0]+I+1,1)
						F1 = F0
		
						F2 = np.arange(np.abs(J1[j1]-I), J1[j1]+I+1, 1)
						F3 = F2

						for K0 in range(numK):
							for K1 in range(numK):
								for Kr in range(numK):

									for f0 in range(len(F0)):
										for f1 in range(len(F1)):
											for f2 in range(len(F2)):
												for f3 in range(len(F3)):
												
													if l0 > N0-1 or l1 > N1-1 or K0 == 1 or K1 == 1:

														Lambda0[N0,l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] = np.nan

														L0[N0,l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] = np.nan

														L2[N0,l0, j0, K0, f0, f1, N1, l1, j1, K1, f2, f3] = np.nan				
						


'''					


		

						


					



