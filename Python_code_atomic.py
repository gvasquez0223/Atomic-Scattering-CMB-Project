import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.wigner import wigner_3j, wigner_6j
from sympy import N

# Constants

h = 6.626e-34 # Planck constant m^2 kg/s
eV_to_J = 1.602e-19

S = 0.5 # spin of the electron
I = 0.5

# Constaints on the number of entrees

numK = 2
numQ = 1
numN = 5
numL = numN - 1
numJ = 2*numL + 1
Nvar = 2*numL +1

def energy(n, l, J, I, F):
	return -13.6*eV_to_J/(n+1)**2

def Nhat(n0, n1, l0, l1, J0, J1, K0, K1, Q0, Q1, F0, F1, F2, F3):

	g = J0*(J0+1) + 2.25 - l0*(l0+1)
	g = g/( 2*J0*(J0+1))
	g += 1 

	term1 = 0
	term2 = 0
	term3 = 0
	term4 = 0

	if K0==K1 and Q0==Q1 and F0==F2 and F1==F3:
		term1 = energy(n0, l0, J0, I, F0) - energy(n1, l1, J1, I, F1)
		term1 = term1/h

	if n0 == n1 and l0==l1 and J0==J1 and Q0 == Q1:
		Q = Q0 
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

rho_QK = np.zeros((numN, numL, numJ, numK, numQ, Nvar, Nvar))
S_QK = np.zeros((numN, numL, numJ, numK, numQ, Nvar, Nvar))
Lambda = np.zeros((numN, numN, numL, numL, numJ, numJ, numK, numK, numQ, numQ, Nvar, Nvar, Nvar, Nvar))


for i1 in range(numN):
	for i2 in range(numN):
		for l1 in range(numL):
			for l2 in range(numL):

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
								for q1 in range(numQ):
									for q2 in range(numQ):
										for x1 in range(len(F1)):
											for x2 in range(len(F2)):
												for x3 in range(len(F3)):
													for x4 in range(len(F4)):

														Lambda[i1,i2,l1,l2,j1,j2,k1,k2,q1,q2,x1,x2,x3,x4] = Nhat(i1,i2,l1,l2,J1[j1],J2[j2],k1,k2,q1,q2,F1[x1],F2[x2],F3[x3],F4[x4])

