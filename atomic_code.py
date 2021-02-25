import numpy as np
import matplotlib.pyplot as plt
import fractions
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j
from scipy.special import genlaguerre, gamma, hyp2f1, factorial, gammaln
from sympy import N

# Constants to be used

h = 6.626e-34 # Planck constant m^2 kg/s
c = 3e8 # Speed of light in units of m/s
hbar = h/(2*np.pi)
eV_to_J = 1.602e-19 # Converting from eV to Joules
e0 = 1.602e-19 # Electron charge
a0 = 5.29e-11 # Bohr radius 
epsilon0 = 8.854e-12 # Epsilon 0 in units of F m^-1
mew0 = 1/(epsilon0*c**2)

# Quantum numbers to consider

S = 0.5 # Electron spin quantum number
I = 0.5 # Nuclear spin quantum number

# Setting strength of the magnetic field

B = 1e-12
larmor_freq = mew0*B/h # Larmor frequency 



'''

Indexing Portion of the code.

'''

# Number of elements for each element of the array

numK = 3
numN = 10
numL = numN
numJ = 2
numF = 2

# We want to create a matrix to index our stuff correctly.

rho_QK = np.zeros((numN, numL, numJ, numK, numF , numF), dtype = np.complex)
Lambda = np.zeros((numN, numN, numL, numL, numJ, numJ,numK, numK, numF, numF, numF, numF), dtype = np.complex)
	
def dipole_element(n0, n1, l0, l1, J0, J1):
	
	
	e0 = 1.602e-19 # electric charge

	# Computing the second term before determining <alpha J ||vec{d}||alpha' J'>
	term1 = e0*(-1)**(J1+1)*np.sqrt(2*J1+1)
	term1 *= float(wigner_3j(J0, J1, 1, 0, 0, 0))

	# We acquire two values of l and l'. We want to check that the selection rules aren't violated.
	# Since there is symmetry, we want to now which value of l is larger or not.

	# We need to indicate which of n0 or n1 is larger.
	
	if n0 > n1:
		n = n0
		n_prime = n1

	elif n0 < n1:
		n = n1
		n_prime = n0


	if l0 == (l1+1):

		l = l0 

		term2 = complex(0,1) * np.sqrt(l)
		term2 *= (-1)**(n_prime-1)/(4* factorial(2*l-1))
		term2 *= np.exp( 0.5*gammaln(n+l+1) + 0.5*gammaln(n_prime+l) - 0.5*gammaln(n-l) - 0.5*gammaln(n_prime-l+1) )
		term2 *= (4*n*n_prime)**(l+1) * (n-n_prime)**(n+n_prime-2*l-2) / (n+n_prime)**(n+n_prime)
		term2 *= ( hyp2f1(-n+l+1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2 ) - (n-n_prime)**2/(n+n_prime)**2 * hyp2f1(-n+l-1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2 ) )

	elif l0 == (l1-1):

		l = l1

		term2 = complex(0,1) * np.sqrt(l)
		term2 *= (-1)**(n_prime-1)/(4* factorial(2*l-1))
		term2 *= np.exp( 0.5*gammaln(n+l+1) + 0.5*gammaln(n_prime+l) - 0.5*gammaln(n-l) - 0.5*gammaln(n_prime-l+1) )
		term2 *= (4*n*n_prime)**(l+1) * (n-n_prime)**(n+n_prime-2*l-2) / (n+n_prime)**(n+n_prime)
		term2 *= ( hyp2f1(-n+l+1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2 ) - (n-n_prime)**2/(n+n_prime)**2 * hyp2f1(-n+l-1, -n_prime+l, 2*l, -4*n*n_prime/(n-n_prime)**2 ) )
		

	else:
		term2 = 0

	return term1*term2

# We need to define the radiation field tensor J_Q^K(nu)

def rad_field_tensor(K, n0, n1, l0, l1, freq):

	# Defines a blackbody

	T = 2.73 # Temperature (K)

	weight = 2*h*freq**3/c**2
	x = h*freq / (kB*T)
	
	BB = 1/ (np.exp(x)-1)

	BB_deriv = np.exp(x) / ( np.exp(x) - 1)**2

	# Determine which radiation field tensor to choose

	if K == 0:
		pert_K0 = 0.001
		rad_field = weight*BB - weight*x*BB_deriv*pert_K0
	elif K == 2:
		pert_K2 = 0.001
		rad_field = (1/np.sqrt(2)) * weight * BB_deriv * pert_K2 #I am aware the polarization piece is missing. Will impliment soon.
	else:
		rad_field = 0

	return rad_field

'''

Functions that represent each term in our equation.

'''

def Nhat(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3):

	Q = 0 # Q=0 is only for scalar perturbations

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
	if K0==K1 and F0==F2 and F1==F3:
		term1 = energy(n0, l0, J0, I, F0) - energy(n1, l1, J1, I, F1)
		term1 = term1/h

	
	# Calculating more terms
	if n0 == n1 and l0==l1:
		J = J0
	
		
		term2 = larmor_freq
		term2 *= g*(-1)**(F1+F3+Q)
		term2 *= np.sqrt( (2*K0+1)*(2*K1+1) )
		term2 *= float(wigner_3j(K0, K1, 1, -Q, Q, 0))



		if F1==F3:
			term3 = (-1)**(1+J+I+F0)
			term3 *= np.sqrt(J*(J+1)*(2*J+1)*(2*F0+1)*(2*F2+1))
			term3 *= float(wigner_6j(F0, F2, 1, J, J, I))
			term3 *= float(wigner_6j(K0, K1, 1, F2, F0, F1))
			term3 = N(term3)

		if F0==F2:
			term4 = (-1)**(K0-K1)
			term4 *= (-1)**(1+J+I+F0)
			term4 *= np.sqrt(J*(J+1)*(2*J+1)*(2*F1+1)*(2*F3+1))
			term4 *= float(wigner_6j(F1, F3, 1, J, J, I))
			term4 *= float(wigner_6j(K0, K1, 1, F3, F1, F0))
			term4 = N(term3)


	return term1+term2*(term3+term4)

def T_A(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3, freq):
	

	# Einstein Coefficient Calculation

	omega = 2*np.pi*freq # Angular frequency

	if n0 > n1:
		
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

	elif n0 < n1:	

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



	
	A_Einstein = omega**3/ (3*np.pi*epsilon0*hbar*c**3)
	A_Einstein *= np.abs( dipole_element(n,n_l,l,l_l, J,J_l) )**2
	
	B_Einstein = np.pi**2 * c**3 / (omega**3*hbar) * A_Einstein


	term1 = (2*J_l+1)*B_Einstein
	term2 = 0
	
	for Kr in range(3):

		term2 += np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_l+1)*(2*F_lprime+1)*(2*K+1)*(2*K_l+1)*(2*Kr+1))
		term2 *= (-1)**(K_l + F_lprime - F_l)
		term2 *= float(wigner_9j(F, F_l, 1, F_prime, F_lprime, 1, K, K_l, Kr) ) * float(wigner_6j(J, J_l, 1, F_l, F, I))
		term2 *= float(wigner_6j(J, J_l, 1, F_lprime, F_prime, I) ) * float(wigner_3j(K, K_l, Kr, 0, 0, 0) )
		term2 *= rad_field_tensor(Kr, n0, n1, l0, l1, freq)

	return term1*term2



def T_S(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3, freq):

	# Einstein Coefficients

	omega = 2*np.pi*freq
	

	if n0 > n1:
		
		n_u = n0
		l_u = l0
		J_u = J0
		K_u = K0
		F_u = F0
		F_uprime = F1

		n = n1
		l = l1
		J = J1
		K = K1
		F = F2
		F_prime = F3

	elif n0 < n1:	

		n_u = n1
		l_u = l1
		J_u = J1
		K_u = K1
		F_u = F2
		F_uprime = F3

		n = n0
		l = l0
		J = J0
		K = K0
		F = F0
		F_prime = F1



	A_Einstein = omega**3/ (3*np.pi*epsilon0*hbar*c**3)
	A_Einstein *= np.abs( dipole_element(n_u, n, l_u, l, J_u, J) )**2
	
	B_Einstein = np.pi**2 * c**3 / (omega**3*hbar) * A_Einstein

	
	term1 = (2*J_u+1)*B_Einstein
	term2 = 0
	
	for Kr in range(3):

		term2 = np.sqrt(3*(2*F+1)*(2*F_prime+1)*(2*F_u+1)*(2*F_uprime+1)*(2*K+1)*(2*K_u+1)*(2*Kr+1))
		term2 *= (-1)**(Kr+K_u+F_uprime-F_u)
		term2 *= float(wigner_9j(F, F_u, 1, F_prime, F_uprime, 1, K, K_u, Kr))
		term2 *= float(wigner_6j(J_u, J, 1, F, F_u, I) )
		term2 *= float( wigner_6j(J_u, J, 1, F_prime, F_uprime, I) )
		term2 *= float( wigner_3j(K, K_u, Kr, 0, 0, 0) )
		term2 *= rad_field_tensor(Kr, n0, n1, l0, l1, freq)

	return term1*term2

def T_E(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3, freq):

	if n0 > n1:
		
		n_u = n0
		l_u = l0
		J_u = J0
		K_u = K0
		F_u = F0
		F_uprime = F1

		n = n1
		l = l1
		J = J1
		K = K1
		F = F2
		F_prime = F3

	elif n0 < n1:	

		n_u = n1
		l_u = l1
		J_u = J1
		K_u = K1
		F_u = F2
		F_uprime = F3

		n = n0
		l = l0
		J = J0
		K = K0
		F = F0
		F_prime = F1


	# Einstein Coefficients

	omega = 2*np.pi*freq
	
	A_Einstein = omega**3/ (3*np.pi*epsilon0*hbar*c**3)
	A_Einstein *= np.abs( dipole_element(n0, n1, l0, l1, J0, J1) )**2
	
	B_Einstein = np.pi**2 * c**3 / (omega**3*hbar) * A_Einstein
	

	term1 = 0

	if K0 == K1:
		
		term1 = 2*J_u + 1
		term1 *= A_Einstein
		term1 *= np.sqrt( (2*F+1)*(2*F_prime+1)*(2*F_u+1)*(2*F_uprime+1) )
		term1 *= (-1)**(1+K0+F1+F3)
		term1 *= float(wigner_6j(F,F_prime,K, F_uprime, F_u, 1))
		term1 *= float(wigner_6j(J_u, J, 1, F, F_u, I))
		term1 *= float(wigner_6j(J_u, J, 1, F_prime, F_uprime, I))
	
	return term1

def R_A(n, l, J, I, K, K_prime, F0, F1, F2, F3, freq):

	# This isn't complete. I need to create a criteria where only Einstein coefficients where we go to a more
	# excited state are considered. Just an fyi.


	B_sum = 0 # Sum of the Einstein coefficients

	term1 = 0
	term2 = 0
	term3 = 0
	term4 = 0

	for n_level in range(Nmax+1):
		for l_level in range(Nmax):

			J_level = np.arange(np.abs(l_level - S), l_level + S, 1) # Calculating the values of J given values of L.

			for j_index in range(2):
				
				omega = 2*np.pi*freq # Angular frequency

				J_l = J_level[j_index] # Lower J level

				# Calculating the Einstein coefficients

				if n_level > n:

					A_Einstein = omega**3 / (3*np.pi*epsilon0*hbar*c**3)
					A_Einstein *= np.abs( dipole_element(n, n_level, l, l_level, J, J_u) )**2
	
					B_Einstein = np.pi**2 * c**3 / (omega**3*hbar) * A_Einstein

					B_sum += B_Einstein	
	
				
					for Kr in range(3):
						
						term2 = np.sqrt(3*(2*K+1)*(2*K_prime+1)*(2*Kr+1))
						term2 *= (-1)**(1+J_u-I+F0)
						term2 *= float( wigner_6j(J, J, Kr, 1, 1, J_u) )
						term2 *= float( wigner_3j(K, K_prime, Kr, 0,0,0) )
	
						if F0 == F2:
							term3 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
							term3 *= float( wigner_6j(J, J, Kr, F3, F1, I) )
							term3 *= float( wigner_6j(K, K_prime, Kr, F3, F1, F0) )
							term3 *= rad_field_tensor(Kr, n0, n1, l0, l1, freq)
	
						elif F1 == F3:
							term4 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
							term4 *= (-1)**(F2 - F1 + K + K_prime + Kr)
							term4 *= float( wigner_6j(J, J, Kr, F2, F0, I) )
							term4 *= float( wigner_6j(K, K_prime, Kr, F2, F0, F1) )
							term4 *= rad_field_tensor(Kr, n0, n1, l0, l1, freq)

		
	term1 = (2*J+1)*B_sum # Term with the sum of Einstein coefficients.

	tot_term = term1*term2*(term3 + term4)
	
	return tot_term



def R_E(n, l, J, I, K, K_prime, F0, F1, F2, F3, freq):


	# This isn't complete. I need to create a criteria where only Einstein coefficients where we go to a more
	# excited state are considered. Just an fyi.


	A_sum = 0 # Sum of Einstein A coefficients.

	if K == K_prime and F0 == F2 and F1 == F3:

		for n_level in range(Nmax+1):
			for l_level in range(Nmax):

				J_level = np.arange(np.abs(l_level - S), l_level + S, 1) # Calculates the values of J for a given L.
				for j_index in range(2):
					
					J_l = J_level[j_index] # Determines the lower level

					if n > n_level:

						omega = 2*np.pi*freq # Angular frequency.
	
						# Calculates the Einstein A coefficient and adds them to a sum.

						A_Einstein = omega**3*e0**2 / (3*np.pi*epsilon0*hbar*c**3)
						A_Einstein *= np.abs( dipole_element(n, n_level, l, l_level, J, J_l) )**2
						A_sum += A_Einstein

	return A_sum

def R_S(n0, l0, J0, I, K0, K1, F0, F1, F2, F3, freq):

	B_sum = 0

	term1 = 0
	term2 = 0
	term3 = 0
	term4 = 0

	for n_level in range(Nmax+1):
		for l_level in range(Nmax):
			J_level = np.arange(np.abs(l_level - S), l_level + S, 1)

			for j_index in range(2):
				
				omega = 2*np.pi*freq # Angular frequency

				J_l = J_level[j_index] # Upper level J value

				if n > n_level:
	
					A_Einstein = omega**3/ (3*np.pi*epsilon0*hbar*c**3)
					A_Einstein *= np.abs( dipole_element(n, n_level, l, l_level, J, J_u) )**2
	
					B_Einstein = np.pi**2 * c**3 / (omega**3*hbar) * A_Einstein

					B_sum += B_Einstein	
	
				
					for Kr in range(3):
						
						term2 = np.sqrt(3*(2*K+1)*(2*K_prime+1)*(2*Kr+1))
						term2 *= (-1)**(1+J_l - I + F0 +Kr)
						term2 *= float( wigner_6j(J, J, Kr, 1, 1, J_l) )
						term2 *= float( wigner_3j(K, K_prime, Kr, 0,0,0) )
	
						if F0 == F2:
							term3 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
							term3 *= float(wigner_6j(J, J, Kr, F3, F1, I))
							term3 *= float(wigner_6j(K, K_prime, Kr, F3, F1, F0))
							term3 *= rad_field_tensor(Kr, n0, n1, l0, l1)
	
						elif F1 == F3:
							term4 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
							term4 *= (-1)**(F2 - F1 + K + K_prime + Kr)
							term4 *= float(wigner_6j(J, J, Kr, F2, F0, I))
							term4 *= float(wigner_6j(K, K_prime, Kr, F2, F0, F1))
							term4 *= rad_field_tensor(Kr, n0, n1, l0, l1)

		
	term1 = (2*J+1)*B_sum
	
	return term1*term2*(term3 + term4)

'''

Lines of code that record each index with the values that we need.

'''

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
	


	
		



