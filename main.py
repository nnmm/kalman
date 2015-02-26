# coding: utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
#import seaborn as sns
import sys


# maybe rewrite this using iterators?
# also make it more general so a, b, psi, phi and u can depend on t
def kalman(data, qmat, rmat, a, b, psi, phi, mu, sigma0, u):
	# Initialization
	# number of time steps
	n = data.shape[0]+1
	p = mu.shape[0]
	q = psi.shape[0]
	r = u.shape[0]

	ident = np.eye(p)
	# forecasting output X_{t|t-1}
	xtt1 = np.empty((n, p))
	# filtering output X_{t|t}
	xtt = np.empty((n, p))
	xtt[0] = mu
	# forecasting autocov Σ_{t|t-1}
	sigmatt1 = np.empty((n, p, p))
	# filtering autocov Σ_{t|t}
	sigmatt = np.empty((n, p, p))
	sigmatt[0] = sigma0
	# Kalman gain
	k = np.empty((n, p, q))


	for t, yt in enumerate(data, start=1):
		# (4.24) – is the u_t in the poly A_t from the pdf?
		xtt1[t] = phi.dot(xtt[t-1]) + a.dot(u)
		# (4.25)
		sigmatt1[t] = phi.dot(sigmatt[t]).dot(phi.T) + qmat
		# (4.26)
		aux_k = np.linalg.inv(psi.dot(sigmatt1[t]).dot(psi.T) + rmat)
		k[t] = sigmatt1[t].dot(psi.T).dot(aux_k)
		# (4.27) – again, what's u and B?
		xtt[t] = xtt1[t] + k[t].dot(yt - psi.dot(xtt1[t]) - b.dot(u))
		# (4.28)
		sigmatt[t] = (ident - k[t].dot(psi)).dot(sigmatt1[t])
	
	# Actually, I think we are only interested in xtt
	return xtt

def plot_trajectories(trjc, data):
	x = np.linspace(0, 1, len(trjc))
	plt.plot(trjc[:,0], trjc[:,1])
	plt.scatter(data[:,0], data[:,1])
	plt.show()

def main(argv):
	# with open(argv[0], 'r') as data:
	data = np.loadtxt('../traj1.dat')

	# https://www.cl.cam.ac.uk/~rmf25/papers/Understanding%20the%20Basis%20of%20the%20Kalman%20Filter.pdf
	
	# sample period
	delta = 1
	# state space dimension
	p = 4
	# observation space dimension
	q = 2
	# "exogenous input" dimension
	r = 3
	# Variance of the white noise A
	vara = 1
	# Variance of the white noise V
	varv = 1


	# Φ_t in the poly is Φ from the pdf
	# X_k = Φ X_{k-1} + Π A_{k-1}
	phi = np.eye(p)
	phi[0, 2] = delta
	phi[1, 3] = delta


	# Ψ_t in the poly
	psi = np.zeros((q, p))
	psi[0, 0] = 1
	psi[1, 1] = 1

	# Q = cov(W_k) = cov(Π A_{k-1})
	qmat = np.zeros((p, p))
	qmat[0, 0] = vara*(delta**4)/4.0
	qmat[1, 1] = vara*(delta**4)/4.0
	qmat[0, 2] = vara*(delta**3)/2.0
	qmat[1, 3] = vara*(delta**3)/2.0
	qmat[2, 0] = vara*(delta**3)/2.0
	qmat[3, 1] = vara*(delta**3)/2.0
	qmat[2, 3] = vara*(delta**2)
	qmat[3, 3] = vara*(delta**2)

	rmat = np.eye(q)*(varv*varv)

	# As per question 5
	sigma0 = np.eye(p)*100

	# As per question 5
	mu = np.zeros((p,))

	# we don't have any of that
	a = np.zeros((p, r))
	b = np.zeros((q, r))
	u = np.zeros((r,))

	trjc = kalman(data, qmat, rmat, a, b, psi, phi, mu, sigma0, u)[:,:2]
	print(trjc)
	plot_trajectories(trjc, data)
	

if __name__ == '__main__':
	# if len(sys.argv) != 2:
	# 	print("Pass data file as parameter")
	# 	sys.exit(2)
	main(sys.argv[1:])