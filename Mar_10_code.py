import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import fractions
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
from scipy.special import genlaguerre, gamma, hyp2f1, factorial, gammaln
from sympy import N


'''
Want to build a primer that dictates in our matrix which value of the quantum numbers we are trying to associate
'''

# Constants to be used

h = 6.626e-34 # Planck constant m^2 kg/s
c = 3e8 # Speed of light in units of m/s
hbar = h/(2*np.pi)
eV_to_J = 1.602e-19 # Converting from eV to Joules
e0 = 1.602e-19 # Electron charge
a0 = 5.29e-11 # Bohr radius 
epsilon0 = 8.854e-12 # Epsilon 0 in units of F m^-1
mew0 = 1/(epsilon0*c**2)
kB = 1.381e-23

# Quantum numbers to consider

S = 0.5 # Electron spin quantum number
I = 0.5 # Nuclear spin quantum number

# Setting strength of the magnetic field

B = 1e-12
larmor_freq = mew0*B/h # Larmor frequency 

numK = 3
numN = 10
numL = numN 
numJ = 2
Nvar = 2


T = 2.73

# We want to develop a program where we can index our matrices correctly.

rho_QK = np.zeros((numN, numL, numJ, numK, Nvar, Nvar), dtype = np.complex)
Lambda = np.zeros((numN, numN, numL, numL, numJ, numJ,numK, numK, Nvar, Nvar, Nvar, Nvar), dtype = np.complex)


'''

We want to theoretically define the energy of a Hydrogen atom that we will use as input
for other sections of this code. We want to incorporate corrections due to the Hyperfine 
structure of the atom along with other effects. 

Inputs: n, l, J, I, F
Output: Energy of a given level

'''

def energy(n, l, J, I, F):

	# Unperturbed energy of the Hydrogen atom
	energy_0th = -13.6*eV_to_J / n**2

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

	# We need to determine which set of quantum numbers is the upper or lower state.
	# Also computes the prefactor.

	if n0 < n1:
		term1 = (-1)**(l0+S+J1)
		term1 *= np.sqrt((2*J1+1)*(2*l0+1))
		term1 *= float(wigner_6j(l0, l1, 1, J1, J0, S) )

	elif n0 > n1:
		term1 = (-1)**(l1+S+J0)
		term1 *= np.sqrt((2*J0+1)*(2*l1+1))
		term1 *= float(wigner_6j(l1, l0, 1, J0, J1, S) )

	else:
		term1 = 0

	# Calculation to determine <n',l-1|| vec{d} ||n,l>. First, we need to find
	# which value of l is larger.

	if l0 == l1-1:

		l = l1
		n = n1
		n_prime = n0

		term2 = np.sqrt(l)
		term2 *= (-1)**(n_prime-1) / (4*factorial(2*l-1))
		term2 *= np.exp( 0.5*gammaln(n+l+1) + 0.5*gammaln(n_prime+l) - 0.5*gammaln(n-l) - 0.5*gammaln(n_prime-l+1) )
		term2 *= (4*n*n_prime)**(l+1) * (n-n_prime)**(n+n_prime-2*l-2) / (n+n_prime)**(n+n_prime)
		term2 *= ( hyp2f1(-n+l+1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2 ) - (n-n_prime)**2/(n+n_prime)**2 * hyp2f1(-n+l-1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2) )

	elif l0 == l1+1:

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

	return e0*a0*term1*term2

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
		print(term1)

	
	# Calculating more terms
	if n0 == n1 and l0==l1 and J0 == J1:

		J = J0
	
		term2 = larmor_freq
		term2 *= g*(-1)**(F1+F2)
		term2 *= np.sqrt( (2*K0+1)*(2*K1+1) )
		term2 *= float(wigner_3j(K0, K1, 1, 0, 0, 0))
		print(term2)


		if F1==F3:
			term3 = (-1)**(1+J+I+F0)
			term3 *= np.sqrt(J*(J+1)*(2*J+1)*(2*F0+1)*(2*F2+1))
			term3 *= float(wigner_6j(F0, F2, 1, J, J, I))
			term3 *= float(wigner_6j(K0, K1, 1, F2, F0, F1))
			print(term3)

		if F0==F2:
			term4 = (-1)**(K0-K1)
			term4 *= (-1)**(1+J+I+F0)
			term4 *= np.sqrt(J*(J+1)*(2*J+1)*(2*F1+1)*(2*F3+1))
			term4 *= float(wigner_6j(F1, F3, 1, J, J, I))
			term4 *= float(wigner_6j(K0, K1, 1, F3, F1, F0))
			print(term4)


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

def T_A(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3, freq):
	
	# Calculating the appropriate Einstein coefficients

	B_Einstein = 32*np.pi**4 / (3*h**2*c)
	B_Einstein *= np.abs( dipole_element(n0, n1, l0, l1, J0, J1) )**2

	# We need to isolate which state is the lower state.

	if energy(n0, l0, J0, I, F0) > energy(n1, l1, J1, I, F1):
		
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

	for Kr in range(3):

		temp = np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_l+1)*(2*F_lprime+1)*(2*K+1)*(2*K_l+1)*(2*Kr+1))
		temp *= (-1)**(K_l + F_lprime - F_l)
		temp *= float(wigner_9j(F, F_l, 1, F_prime, F_lprime, 1, K, K_l, Kr) ) * float(wigner_6j(J, J_l, 1, F_l, F, I))
		temp *= float(wigner_6j(J, J_l, 1, F_lprime, F_prime, I) ) * float(wigner_3j(K, K_l, Kr, 0, 0, 0) )
		temp *= rad_field_tensor(Kr, n0, n1, l0, l1, freq, T)

		term2 += temp

	return term1*term2




def T_S(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3, freq):

	# Calculating the appropriate Einstein coefficients
	
	B_Einstein = 32*np.pi**4 / (3*h**2*c)
	B_Einstein *= np.abs( dipole_element(n0, n1, l0, l1, J0, J1) )**2

	# We need to isolate which state is the upper state.

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

	for Kr in range(3):

		temp = np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_u+1)*(2*F_uprime+1)*(2*K+1)*(2*K_u+1)*(2*Kr+1))
		temp *= (-1)**(Kr+K_u+F_uprime-F_u)
		temp *= float( wigner_9j(F, F_u, 1, F_prime, F_uprime, 1, K, K_u, Kr))
		temp *= float( wigner_6j(J_u, J, 1, F, F_u, I) )
		temp *= float( wigner_6j(J_u, J, 1, F_prime, F_uprime, I) )
		temp *= float( wigner_3j(K, K_u, Kr, 0, 0, 0) )
		temp *= rad_field_tensor(Kr, n0, n1, l0, l1, freq, T)

		term2 += temp

	return term1*term2

def T_E(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3, freq):

	# Calculating the appropriate Einstein Coefficients

	A_Einstein = 64*np.pi**4 * freq**3 / (3*h*c**3)
	A_Einstein *= np.abs( dipole_element(n0, n1, l0, l1, J0, J1) )**2

	# We need to determine which state is the upper state.

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

def R_A(n, l, J, I, K, K_prime, F0, F1, F2, F3, freq):


	Nmax = 100 # Total number of different values of n we are considering.

	# Define 3 terms to make the calculation easier

	term1 = 0
	term2 = 0
	term3 = 0

	for n_level in range(Nmax+1):
		for l_level in range(Nmax):

			# Composes allowed values of J given a value of L and S.
			# Allowed values: J = L- 1/2, L+ 1/2

			J_level = np.arange( np.abs(l_level - S), l_level + S+1, 1)

			for j_index in range(len(J_level)):

				J_u = J_level[j_index] # J_u value

				# Need to determine if a state is more energetic.

				if n_level > n:

					B_Einstein = 32*np.pi**4 / (3*h**2*c)
					B_Einstein *= np.abs( dipole_element(n_level, n, l_level, l, J_u, J) )**2
	
					for Kr in range(3):
						
						term1 = B_Einstein
						term1 *= np.sqrt(3*(2*K+1)*(2*K_prime+1)*(2*Kr+1))
						term1 *= (-1)**(1+J_u-I+F0)
						term1 *= float( wigner_6j(J, J, Kr, 1, 1, J_u) )
						term1 *= float( wigner_3j(K, K_prime, Kr, 0,0, 0) )
	
						if F0 == F2:
							term2 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
							term2 *= float( wigner_6j(J, J, Kr, F3, F1, I) )
							term2 *= float( wigner_6j(K, K_prime, Kr, F3, F1, F0) )
							term2 *= rad_field_tensor(Kr, n, n_level, l, l_level, freq, T)
						if F1 == F3:
							term3 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
							term3 *= (-1)**(F2 - F1 + K + K_prime + Kr)
							term3 *= float( wigner_6j(J, J, Kr, F2, F0, I) )
							term3 *= float( wigner_6j(K, K_prime, Kr, F2, F0, F1) )
							term3 *= rad_field_tensor(Kr, n, n_level, l, l_level, freq, T)

		
	term1 *= (2*J + 1) # Multiplying by a prefactor
	
	return term1*(term2 + term3)



def R_E(n, l, J, I, K, K_prime, F0, F1, F2, F3, freq):

	Nmax = 100 # Total number of principal quantum states being considered.

	A_sum = 0 # Sum of Einstein A coefficients.

	if K == K_prime and F0 == F2 and F1 == F3:

		for n_level in range(Nmax+1):
			for l_level in range(Nmax):


				# Composes allowed values of J given a value of L and S.
				# Allowed values: J = L- 1/2, L+ 1/2

				J_level = np.arange(np.abs(l_level - S), l_level + S + 1, 1)

				for j_index in range(len(J_level)):

					J_l = J_level[j_index] # J for the lower state.

					# Need to determine if a state is less energetic.

					if n_level < n:

						A_Einstein = 64*np.pi**4 * freq**3 / (3*h*c**3)
						A_Einstein *= np.abs( dipole_element(n_level, n, l_level, l, J_l, J) )**2

						A_sum += A_Einstein # Sum each allowed Einstein-A coefficient.

	return A_sum



def R_S(n, l, J, I, K, K_prime, F0, F1, F2, F3, freq):

	Nmax = 100 # Total number of principal quantum states being considered.

	# Define 3 terms to make the calculations easier.

	term1 = 0
	term2 = 0
	term3 = 0

	for n_level in range(Nmax+1):
		for l_level in range(Nmax):

			# Composes allowed values of J given a value of L and S.
			# Allowed values: J = L- 1/2, L+ 1/2

			J_level = np.arange(np.abs(l_level - S), l_level + S + 1, 1)

			for j_index in range(len(J_level)):

				J_l = J_level[j_index] # J value for the lower level.


				# Need to determine if a state is less energetic.

				if n_level < n:

					# Calculating allowed Einstein-B coefficients

					B_Einstein = 32*np.pi**4 / (3*h**2*c)
					B_Einstein *= np.abs( dipole_element(n_level, n, l_level, l, J_l, J) )**2
	
					for Kr in range(3):
						
						term1 = B_Einstein
						term1 *= np.sqrt( 3*(2*K+1)*(2*K_prime+1)*(2*Kr+1) )
						term1 *= (-1)**(1+J_l - I + F0 +Kr)
						term1 *= float( wigner_6j(J, J, Kr, 1, 1, J_l) )
						term1 *= float( wigner_3j(K, K_prime, Kr, 0,0,0) )
	
						if F0 == F2:
							term2 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
							term2 *= float( wigner_6j(J, J, Kr, F3, F1, I) )
							term2 *= float( wigner_6j(K, K_prime, Kr, F3, F1, F0) )
							term2 *= rad_field_tensor(Kr, n, n_level, l, l_level, freq, T)
	
						if F1 == F3:
							term3 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
							term3 *= (-1)**(F2 - F1 + K + K_prime + Kr)
							term3 *= float( wigner_6j(J, J, Kr, F2, F0, I) )
							term3 *= float( wigner_6j(K, K_prime, Kr, F2, F0, F1) )
							term3 *= rad_field_tensor(Kr, n, n_level, l, l_level, freq, T)

		
	term1 *= (2*J+1) # Prefactor we need to include
	
	return term1*(term2 + term3)

'''		


for i1 in range(numN):
	for i2 in range(numN):
		for l1 in range(numL-1):
			for l2 in range(numL-1):
				
				
				J1 = np.arange(np.abs(l1-S), l1+S+1, 1)
				J2 = np.arange(np.abs(l2-S), l2+S+1, 1)

				for j1 in range(len(J1)):
					for j2 in range(len(J2)):

						F1 = np.arange(np.abs(J1[j1]-I), J1[j1]+I+1,1)
						F2 = np.arange(np.abs(J1[j1]-I), J1[j1]+I+1,1)
						F3 = np.arange(np.abs(J2[j2]-I), J2[j2]+I+1,1)
						F4 = np.arange(np.abs(J2[j2]-I), J2[j2]+I+1,1)

						for k1 in range(numK):
							for k2 in range(numK):
								for f1 in range(len(F1)):
									for f2 in range(len(F2)):
										for f3 in range(len(F3)):
											for f4 in range(len(F4)):


												term = - 2*np.pi*complex(0,Nhat(i1, i2, l1, l2, J1[j1], J2[j2], k1, k2, F1[f1], F2[f2], F3[f3], F4[f4]))
												#term += T_A(i1, i2, l1, l2, J1[j1], J2[j2], k1, k2, F1[f1], F2[f2], F3[f3], F4[f4])
												#term += T_E(i1, i2, l1, l2, J1[j1], J2[j2], k1, k2, F1[f1], F2[f2], F3[f3], F4[f4])
												#term += T_S(i1, i2, l1, l2, J1[j1], J2[j2], k1, k2, F1[f1], F2[f2], F3[f3], F4[f4])
												
						

				
												Lambda[i1, i2, l1, l2, j1, j2, k1, k2, f1, f2, f3, f4] = term


'''

	
					



