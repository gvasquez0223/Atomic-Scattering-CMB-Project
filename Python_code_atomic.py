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

h = 6.626e-34 # Planck constant m^2 kg/s
hbar = h/(2*np.pi)
eV_to_J = 1.602e-19 # Converting from eV to Joules
e0 = 1.602e-19 # Electron charge
a0 = 5.29e-11 # Bohr radius 
epsilon0 = 8.854e-12 # Epsilon 0 in units of F m^-1

S = 0.5 # Spin Quantum Number
I = 0.5 # Nuclear Quantum Number

numK = 3
numN = 10
numL = numN 
numJ = 2
Nvar = 2

# We want to develop a program where we can index our matrices correctly.

rho_QK = np.zeros((numN, numL, numJ, numK, Nvar, Nvar), dtype = np.complex)
Lambda = np.zeros((numN, numN, numL, numL, numJ, numJ,numK, numK, Nvar, Nvar, Nvar, Nvar), dtype = np.complex)

def energy(n, l, J, I, F):
	return -13.6*eV_to_J/(n+1)**2


# Computes the dipole matrix elements for <alpha J I F f | d_{q}^{1} |alpha' J' I F' f>


def dipole_element(n0, n1, l0, l1, J0, J1):

	e0 = 1.602e-19 # Electron charge

	# Computing the second term before determining <alpha J ||vec{d}||alpha' J'>
	term1 = e0*(-1)**(J1+1)*np.sqrt(2*J1+1)
	term1 *= float(wigner_3j(J0, J1, 1, 0, 0, 0))


	# Computes the overlap integral
	if l0 == l1 + 1 or l0 == l1 - 1:

		term2 = (-1)**(n1-1)/(4* factorial(2*l0+1))
		term2 *= np.exp( 0.5*gammaln(n0+l0) + 0.5*gammaln(n1+l0-1) - 0.5*gammaln(n0-l0-1) - 0.5*gammaln(n1-1) )
		term2 *= (4*n0*n1)**(l0+1) * (n0-n1)**(n0+n1-2*l0-2) / (n0+n1)**(n0+n1)
		term2 *= ( hyp2f1(-n0+l0+1, -n1+l0, 2*l0, -4*n0*n1/(n0-n1)**2 ) - (n0-n1)**2/(n0+n1)**2 * hyp2f1(-n0+l0-1, -n1+l0, 2*l0, -4*n0*n1/(n0-n1)**2 ) )

	elif:
		term2 = 0

	return term1*term2
		



	

'''
def dipole_element(n0, n1, l0, l1, J0, J1, F0, F1, f0, f1):
	
	I = 0.5 # Nuclear spin component
	e0 = 1.602e-19 # Electron charge

	# Computing the first term before determining <alpha J I F||vec{d}||alpha' J' I F'>
	term1 = (-1)**(J0+I-f0)
	term1 *= np.sqrt( (2*J0+1)*(2*F0+1)*(2*F1+1) )
	term1 *= float(wigner_3j(F0, F1, 1, -f0, f1, 0))
	term1 *= float(wigner_6j(J0, J1, 1, F1, F0, I))

	# Computing the second term before determining <alpha J ||vec{d}||alpha' J'>
	term2 = e0*(-1)**(J1+1)*np.sqrt(2*J1+1)
	term2 *= float(wigner_3j(J0, J1, 1, 0, 0, 0))

	'''
	# To compute <alpha||\vec{d} || alpha'> we need to compute an overlap integral
	# between the two radial w.f.s and the radial vector r.

	# First radial part 
	WF0 = (1/n0) *np.sqrt( gamma(n0-l0) / gamma(n0+l0+1) ) 
	L0= genlaguerre(n0-l0-1, 2*l0+1)

	# Second radial part
	WF1 = (1/n1) *np.sqrt( gamma(n1-l1) / gamma(n1+l1+1) ) 
	L1 = genlaguerre(n1-l1-1, 2*l1+1)

	'''

	# Computing <alpha|| \vec{d} || alpha'> we can use the definition given

	if l0 == l1 + 1 or l0 == l1 - 1:
	
		term3 = complex(0,1)
		term3 *= np.sqrt(l0)*(-1)**(n1-1)/(4* factorial(2*l0+1))
		term3 *= np.exp( 0.5*gammaln(n0+l0) + 0.5*gammaln(n1+l0-1) - 0.5*gammaln(n0-l0-1) - 0.5*gammaln(n1-1) )
		term3 *= (4*n0*n1)**(l0+1) * (n0-n1)**(n0+n1-2*l0-2) / (n0+n1)**(n0+n1)
		term3 *= ( hyp2f1(-n0+l0+1, -n1+l0, 2*l0, -4*n0*n1/(n0-n1)**2 ) - (n0-n1)**2/(n0+n1)**2 * hyp2f1(-n0+l0-1, -n1+l0, 2*l0, -4*n0*n1/(n0-n1)**2 ) )

	elif:
		term3 = 0
		

	# Computing overlap integral
	overlap = integrate.quad(lambda x: (2*x/n0)**(l0+1)* np.exp(-x/n0)*(2*x/n0)**(l1+1)* np.exp(-x/n1)*L0(x/n0)*L1(x/n1) * x**3,0,100 )
	

	# Integral result multiplied by numerical factors
	int_result = overlap[0]
	int_result *= WF0*WF1


	return term1*term2*term3
'''


def Nhat(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3):

	Q = 0

	g = J0*(J0+1) + 2.25 - l0*(l0+1)
	g = g/( 2*J0*(J0+1))
	g += 1 

	term1 = 0
	term2 = 0
	term3 = 0
	term4 = 0

	if K0==K1 and F0==F2 and F1==F3:
		term1 = energy(n0, l0, J0, I, F0) - energy(n1, l1, J1, I, F1)
		term1 = term1/h

	if n0 == n1 and l0==l1 and J0==J1:
		J = J0

		term2 =  g*(-1)**(F1+F3+Q)
		term2 *= np.sqrt( (2*K0+1)*(2*K1+1) )
		term2 *= wigner_3j(K0, K1, 1, -Q, Q, 0)



		if F1==F3:
			term3 = (-1)**(1+J+I+F0)
			term3 *= np.sqrt(J*(J+1)*(2*J+1)*(2*F0+1)*(2*F2+1))
			term3 *= wigner_6j(F0, F2, 1, J, J, I)
			term3 *= wigner_6j(K0, K1, 1, F2, F0, F1)
			term3 = N(term3)

		if F0==F2:
			term4 = (-1)**(K0-K1)
			term4 *= (-1)**(1+J+I+F0)
			term4 *= np.sqrt(J*(J+1)*(2*J+1)*(2*F1+1)*(2*F3+1))
			term4 *= wigner_6j(F1, F3, 1, J, J, I)
			term4 *= wigner_6j(K0, K1, 1, F3, F1, F0)
			term4 = N(term3)


	return term1+term2+term3+term4

def T_A(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3, freq):
	

	# Einstein Coefficients

	omega = 2*np.pi*freq
	
	A_Einstein = omega**3*e0**2 / (3*np.pi*epsilon0*hbar*c**3)
	A_Einstein *= np.abs( dipole_element(n0,n1,l0,l1, J0,J1) )**2
	
	B_Einstein = np.pi**2 * c**3 / (omega**3*hbar) * A_Einstein


	term1 = (2*J1+1)*B_Einstein
	term2 = 0
	
	for Kr in range(3):
		for Qr in range(3):
			term2 += np.sqrt(3*(2*F0+1)*(2*F1+1)*(2*F2+1)*(2*F3+1)*(2*K0+1)*(2*K1+1)*(2*Kr+1))
			term2 *= (-1)**(K1 + F2 + F3)
			term2 *= wigner_9j(F0, F2, 1, F1, F3, 1, K0, K1, Kr) * wigner_6j(J0, J1, 1, F2, F0, I)
			term2 *= wigner_6j(J0, J1, 1, F3, F1, I)*wigner_3j(K0, K1, Kr, 0, 0, Qr)
			term2 *= rad_field_tensor(Kr, n0, n1, l0, l1)

	return term1*term2



def T_S(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3, freq):

	# Einstein Coefficients

	omega = 2*np.pi*freq
	
	A_Einstein = omega**3*e0**2 / (3*np.pi*epsilon0*hbar*c**3)
	A_Einstein *= np.abs( dipole_element(n0, n1, l0, l1, J0, J1) )**2
	
	B_Einstein = np.pi**2 * c**3 / (omega**3*hbar) * A_Einstein

	
	term1 = (2*J1+1)*B_Einstein
	term2 = 0
	
	for Kr in range(3):
		for Qr in range(3):
			term2 = np.sqrt(3*(2*F0+1)*(2*F1+1)*(2*F2+1)*(2*F3+1)*(2*K0+1)*(2*K1+1)*(2*Kr+1))
			term2 *= (-1)**(Kr+K1+F3-F2)
			term2 *= wigner_9j(F0, F2, 1, F1, F3, 1, K0, K1, Kr)
			term2 *= wigner_6j(J1, J0, 1, F0, F2, I)
			term2 *= wigner_6j(J1, J0, 1, F1, F3, I)
			term2 *= wigner_3j(K0, K1, Kr, 0, 0, -Qr)
			term2 *= rad_field_tensor(Kr, n0, n1, l0, l1)

	return term1*term2

def T_E(n0, n1, l0, l1, J0, J1, K0, K1, F0, F1, F2, F3, freq):

	# Einstein Coefficients

	omega = 2*np.pi*freq
	
	A_Einstein = omega**3*e0**2 / (3*np.pi*epsilon0*hbar*c**3)
	A_Einstein *= np.abs( dipole_element(n0, n1, l0, l1, J0, J1) )**2
	
	B_Einstein = np.pi**2 * c**3 / (omega**3*hbar) * A_Einstein
	

	term1 = 0

	if K==K1:
		
		term1 = 2*J1 + 1
		term1 *= A_Einstein
		term1 *= np.sqrt( (2*F0+1)*(2*F1+1)*(2*F2+1)*(2*F3+1) )
		term1 *= (-1)**(1+K0+F1+F3)
		term1 *= wigner_6j(F0,F1,K0, F3, F2, 1)
		term1 *= wigner_6j(J1, J0, 1, F0, F2, I)
		term1 *= wigner_6j(J1, J0, 1, F1, F3, I)
	
	return term1

def R_A(n0, l0, J0, I, K0, K1, F0, F1, F2, F3, freq):

B_sum = 0

term1 = 0
term2 = 0
term3 = 0
term4 = 0

	for n_level in range(Nmax+1):
		for l_level in range(Nmax):
			for j_index in range(2):
				J_level = np.arange(np.abs(l_level - S), l_level + S, 1)

				omega = 2*np.pi*freq
	
				A_Einstein = omega**3*e0**2 / (3*np.pi*epsilon0*hbar*c**3)
				A_Einstein *= np.abs( dipole_element(n, n_index, l, l_index, J, J_level[j_index]) )**2
	
				B_Einstein = np.pi**2 * c**3 / (omega**3*hbar) * A_Einstein

				B_sum += B_Einstein	
	
				
				for Kr in range(3):
					for Qr in range(3):
						
						term2 = np.sqrt(3*(2*K0+1)*(2*K1+1)*(2*Kr+1))
						term2 *= (-1)**(1+J_level[j_index]-I+F0)
						term2 *= wigner_6j(J0, J0, Kr, 1, 1, J_level[j_index])
						term2 *= wigner_3j(K0, K1, Kr, 0,0,Qr)
	
						if F0 == F2:
							term3 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
							term3 *= wigner_6j(J0, J0, Kr, F3, F1, I)
							term3 *= wigner_6j(K0, K1, Kr, F3, F1, F0)
							term3 *= rad_field_tensor(Kr, n0, n1, l0, l1)
	
						elif F1 == F3:
							term4 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
							term4 *= (-1)**(F2 - F1 + K0 + K1 + Kr)
							term4 *= wigner_6j(J0, J0, Kr, F2, F0, I)
							term4 *= wigner_6j(K0, K1, Kr, F2, F0, F1)
							term4 *= rad_field_tensor(Kr, n0, n1, l0, l1)

		
	term1 = (2*J0+1)*B_sum
	
return term1*term2*(term3 + term4)



def R_E(n0, l0, J0, I, K0, K1, F0, F1, F2, F3, freq):

A_sum = 0

	if K0 == K1 and F0 == F2 and F1 == F3:

		for n_level in range(Nmax+1):
			for l_level in range(Nmax):
				for j_index in range(2):
					J_level = np.arange(np.abs(l_level - S), l_level + S, 1)

					omega = 2*np.pi*freq
	
					A_Einstein = omega**3*e0**2 / (3*np.pi*epsilon0*hbar*c**3)
					A_Einstein *= np.abs( dipole_element(n, n_index, l, l_index, J, J_level[j_index]) )**2
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
			for j_index in range(2):
				J_level = np.arange(np.abs(l_level - S), l_level + S, 1)

				omega = 2*np.pi*freq

				if J_level[j_index]
	
				A_Einstein = omega**3*e0**2 / (3*np.pi*epsilon0*hbar*c**3)
				A_Einstein *= np.abs( dipole_element(n, n_index, l, l_index, J, J_level[j_index]) )**2
	
				B_Einstein = np.pi**2 * c**3 / (omega**3*hbar) * A_Einstein

				B_sum += B_Einstein	
	
				
				for Kr in range(3):
					for Qr in range(3):
						
						term2 = np.sqrt(3*(2*K0+1)*(2*K1+1)*(2*Kr+1))
						term2 *= (-1)**(1+J_level[j_index] - I + F0 +Kr)
						term2 *= wigner_6j(J0, J0, Kr, 1, 1, J_level[j_index])
						term2 *= wigner_3j(K0, K1, Kr, 0,0,Qr)
	
						if F0 == F2:
							term3 = 0.5*np.sqrt( (2*F1+1)*(2*F3+1) )
							term3 *= wigner_6j(J0, J0, Kr, F3, F1, I)
							term3 *= wigner_6j(K0, K1, Kr, F3, F1, F0)
							term3 *= rad_field_tensor(Kr, n0, n1, l0, l1)
	
						elif F1 == F3:
							term4 = 0.5*np.sqrt( (2*F0+1)*(2*F2+1) )
							term4 *= (-1)**(F2 - F1 + K0 + K1 + Kr)
							term4 *= wigner_6j(J0, J0, Kr, F2, F0, I)
							term4 *= wigner_6j(K0, K1, Kr, F2, F0, F1)
							term4 *= rad_field_tensor(Kr, n0, n1, l0, l1)

		
	term1 = (2*J0+1)*B_sum
	
return term1*term2*(term3 + term4)



		


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




	
					



