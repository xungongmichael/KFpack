import numpy as np 
cimport numpy as np 
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def KalmanFilter(np.float a1,np.float p1,\
	np.ndarray[double, ndim=1] C,\
	np.ndarray[double, ndim=1] D,\
	np.ndarray[double, ndim=1] T,\
	np.ndarray[double, ndim=1] R,\
	np.ndarray[double, ndim=1] Z,\
	np.ndarray[double, ndim=1] Q,\
	np.ndarray[double, ndim=1] H,\
	np.ndarray[double, ndim=1] y):
	
	cdef int N = len(y)

	cdef np.ndarray[double, ndim=1] a_t  = np.zeros(N+1)
	cdef np.ndarray[double, ndim=1] a_tt = np.zeros(N)
	cdef np.ndarray[double, ndim=1] P_t  = np.zeros(N+1)
	cdef np.ndarray[double, ndim=1] P_tt = np.zeros(N)

	cdef np.ndarray[double, ndim=1] v = np.zeros(N)
	cdef np.ndarray[double, ndim=1] F = np.zeros(N)
	cdef np.ndarray[double, ndim=1] K = np.zeros(N)
	cdef np.ndarray[double, ndim=1] L = np.zeros(N)
	
	a_t[0] = a1 
	P_t[0] = p1

	cdef int i 
	for i in range(0,N):
		v[i] = y[i]-Z[i]*a_t[i]-D[i]
		F[i] = Z[i]*P_t[i]*Z[i]+H[i]
		K[i] = T[i]*P_t[i]*Z[i]/F[i]

		a_tt[i]  = a_t[i] + P_t[i]*Z[i]*v[i]/F[i]
		P_tt[i]  = P_t[i] - P_t[i]*Z[i]*Z[i]*P_t[i]/F[i] 

		P_t[i+1] = T[i]*P_t[i]*T[i]+R[i]*Q[i]*R[i]-K[i]*F[i]*K[i]
		a_t[i+1] = T[i]*a_t[i] + K[i]*v[i]+C[i]

	return a_t,P_t,v,F,K

@cython.boundscheck(False)
@cython.wraparound(False)
def KalmanSmooth(np.float a1,np.float p1,\
	np.ndarray[double, ndim=1] C,\
	np.ndarray[double, ndim=1] D,\
	np.ndarray[double, ndim=1] T,\
	np.ndarray[double, ndim=1] R,\
	np.ndarray[double, ndim=1] Z,\
	np.ndarray[double, ndim=1] Q,\
	np.ndarray[double, ndim=1] H,\
	np.ndarray[double, ndim=1] y):
	
	cdef np.ndarray[double, ndim=1] a 
	cdef np.ndarray[double, ndim=1] P 
	cdef np.ndarray[double, ndim=1] v 
	cdef np.ndarray[double, ndim=1] F 
	cdef np.ndarray[double, ndim=1] K 

	a,P,v,F,K = KalmanFilter(a1,p1,C,D,T,R,Z,Q,H,y)
	a = a[:-1]
	P = P[:-1]

	cdef int N = len(v)
	cdef np.ndarray[double, ndim=1] r     = np.zeros(N)
	cdef np.ndarray[double, ndim=1] Nn    = np.zeros(N)
	cdef np.ndarray[double, ndim=1] Alpha = np.zeros(N)
	cdef np.ndarray[double, ndim=1] V     = np.zeros(N)

	cdef int i 
	cdef float L
	cdef float k 
	for i in range(N-1,-1,-1):
		k = P[i]/F[i]
		L = 1-k
		r[i-1]  = Z[i]*v[i]/F[i]+L*r[i]
		Alpha[i]= a[i]+P[i]*r[i-1]
		Nn[i-1] = Z[i]*Z[i]/F[i]+L*Nn[i]*L
		V[i]    = P[i]-P[i]*Nn[i-1]*P[i]

	return Alpha,V

@cython.boundscheck(False)
@cython.wraparound(False)
def KalmanSimu(np.float a1,np.float p1,\
	np.ndarray[double, ndim=1] C,\
	np.ndarray[double, ndim=1] D,\
	np.ndarray[double, ndim=1] T,\
	np.ndarray[double, ndim=1] R,\
	np.ndarray[double, ndim=1] Z,\
	np.ndarray[double, ndim=1] Q,\
	np.ndarray[double, ndim=1] H,\
	np.ndarray[double, ndim=1] y):

	cdef int N = len(y)
	cdef np.ndarray[double, ndim=1] ita_plus = np.sqrt(Q)*np.random.normal(0,1,N)
	cdef np.ndarray[double, ndim=1] eps_plus = np.sqrt(H)*np.random.normal(0,1,N)
	cdef np.ndarray[double, ndim=1] alpha_plus = np.zeros(N)

	alpha_plus[0] = np.sqrt(p1)* np.random.normal(0,1)

	cdef int i 
	for i in range(0,N-1):
		alpha_plus[i+1] = T[i]*alpha_plus[i]+R[i]*ita_plus[i]

	cdef np.ndarray[double, ndim=1] y_plus = C + Z*alpha_plus + eps_plus

	cdef np.ndarray[double, ndim=1] alpha_plus_hat
	cdef np.ndarray[double, ndim=1] alpha_hat
	cdef np.ndarray[double, ndim=1] V

	alpha_plus_hat,V = KalmanSmooth(a1,p1,C,D,T,R,Z,Q,H,y_plus)
	alpha_hat,V      = KalmanSmooth(a1,p1,C,D,T,R,Z,Q,H,y)

	cdef np.ndarray[double, ndim=1] alpha_simu = alpha_hat + alpha_plus - alpha_plus_hat

	return alpha_simu









# @cython.boundscheck(False)
# @cython.wraparound(False)
# def LogLikelihood(np.ndarray[double, ndim=1] Pars):
# 	cdef int N
# 	cdef float omega = Pars[0]
# 	cdef float phi   = Pars[1]
# 	cdef float sigma = Pars[2]

# 	cdef float mean 
# 	cdef float var
# 	cdef float a1 = mean
# 	cdef float p1 = sigma**2/(1-phi**2)

# 	cdef np.ndarray[double, ndim=1] T  = np.ones(N)*phi
# 	cdef np.ndarray[double, ndim=1] C  = np.ones(N)*omega
# 	cdef np.ndarray[double, ndim=1] D  = np.ones(N)*mean
# 	cdef np.ndarray[double, ndim=1] R  = np.ones(N)*sigma
# 	cdef np.ndarray[double, ndim=1] Z  = np.ones(N)
# 	cdef np.ndarray[double, ndim=1] Q  = np.ones(N)
# 	cdef np.ndarray[double, ndim=1] H  = np.ones(N)*var

# 	cdef np.ndarray[double, ndim=1] a 
# 	cdef np.ndarray[double, ndim=1] P 
# 	cdef np.ndarray[double, ndim=1] v 
# 	cdef np.ndarray[double, ndim=1] F 
# 	cdef np.ndarray[double, ndim=1] K 

# 	cdef np.ndarray[double, ndim=1] x
# 	a,P,v,F,K=KalmanFilter(a1,p1,C,D,T,R,Z,Q,H,x)

# 	return 0.5*np.sum(np.log(F)) + 0.5*np.sum(v*v/F)
























# @cython.boundscheck(False)
# @cython.wraparound(False)
# def KalmanFilter(np.ndarray[double, ndim=2] a1, np.ndarray[double, ndim=2] p1,\
# 	np.ndarray[double, ndim=2] T,np.ndarray[double, ndim=2] R,np.ndarray[double, ndim=2] Z,\
# 	np.ndarray[double, ndim=2] Q,np.ndarray[double, ndim=2] H,np.ndarray[double, ndim=2] y):
	
# 	cdef int p = y.shape[1]
# 	cdef int m = a1.shape[0]
# 	cdef int r = Q.shape[1]
# 	cdef int N = y.shape[0]


# 	cdef np.ndarray[double, ndim=2] a = np.zeros([(N+1)*m,1])
# 	a[0:m,:] = a1


# 	cdef np.ndarray[double, ndim=2] P = np.zeros([(N+1)*m, m])
# 	P[0:m,:] = p1

# 	cdef np.ndarray[double, ndim=2] v = np.zeros([p*N, 1])
# 	cdef np.ndarray[double, ndim=2] F = np.zeros([N*p, p])
# 	cdef np.ndarray[double, ndim=2] K = np.zeros([N*m, p])
# 	cdef np.ndarray[double, ndim=2] L = np.zeros([N*m, m])

# 	cdef int i
# 	for i in range(0,N):
# 		v[i*p:(i+1)*p,:] = y[i,:].reshape(p,1)- np.dot(Z[i*p:(i+1)*p,:], a[i*m:(i+1)*m,:])
# 		F[i*p:(1+i)*p,:] = np.dot(np.dot(Z[i*p:(1+i)*p,:],P[i*m:(i+1)*m,:]),Z[i*p:(1+i)*p,:].T)
# 		K[i*m:(i+1)*m,:] = np.dot(np.dot(np.dot(T[i*m:(i+1)*m,:],P[i*m:(i+1)*m,:]),Z[i*p:(1+i)*p,:].T),np.linalg.inv(F[i*p:(1+i)*p,:]))
# 		L[i*m:(i+1)*m,:] = T[i*m:(i+1)*m,:] - np.dot(K[i*m:(i+1)*m,:],Z[i*p:(1+i)*p,:])

# 		a[(i+1)*m:(i+2)*m,:] = np.dot(T[i*m:(i+1)*m,:],a[i*m:(i+1)*m,:]) + np.dot(K[i*m:(i+1)*m,:],v[i*p:(i+1)*p,:])
# 		P[(i+1)*m:(i+2)*m,:] = np.dot(np.dot(T[i*m:(i+1)*m,:],P[i*m:(i+1)*m,:]),L[i*m:(i+1)*m,:].T) + np.dot(np.dot(P[i*m:(i+1)*m,:],Q[i*r:(i+1)*r,:]),P[i*m:(i+1)*m,:].T)

# 	return a[:-m,:],P[:-m,:],v,F,K,L





# @cython.boundscheck(False)
# @cython.wraparound(False)
# def kamfilter(double[::1] y,int N):
# 	cdef int a1 = 0
# 	cdef int P1 = 10**7
# 	cdef float sigma_e = 15099
# 	cdef float sigma_n = 1469.1

# 	cdef double[::1] a = np.zeros(N+1)
# 	cdef double[::1] P = np.zeros(N+1)
# 	cdef double[::1] v = np.zeros(N)
# 	cdef double[::1] F = np.zeros(N)
# 	cdef double[::1] K = np.zeros(N)

# 	cdef int i
	
# 	a[0] = a1
# 	P[0] = P1 

# 	cdef double* vptr = &v[0]
# 	cdef double* Fptr = &F[0]
# 	cdef double* Kptr = &K[0]
# 	cdef double* Pptr = &P[0]
# 	cdef double* aptr = &a[0]
# 	cdef double* yptr = &y[0]

# 	for i in range(0,N):
# 		vptr[i] = yptr[i]-aptr[i]
# 		Fptr[i] = Pptr[i]+sigma_e
# 		Kptr[i] = Pptr[i]/Fptr[i]
# 		Pptr[i+1] = Pptr[i]*(1-Kptr[i])+sigma_n
# 		aptr[i+1] = aptr[i] + Kptr[i]*vptr[i]

# 	return a[:-1],P[:-1],v,F,K


# @cython.boundscheck(False)
# @cython.wraparound(False)
# def kalsmooth(double[::1]y, int N):
# 	cdef double[::1] a 
# 	cdef double[::1] P 
# 	cdef double[::1] v 
# 	cdef double[::1] F 
# 	cdef double[::1] K 

# 	a,P,v,F,K = kamfilter(y,N)

# 	cdef double[::1] r 		= np.zeros(N)
# 	cdef double[::1] alpha 	= np.zeros(N)
# 	cdef double[::1] Nn 	= np.zeros(N)
# 	cdef double[::1] V 		= np.zeros(N)
# 	cdef int i 

# 	cdef float Ki
# 	cdef float L
# 	for i in range(N-1,-1,-1):
# 		Ki = P[i]/F[i]
# 		L = 1-Ki
# 		r[i-1]=v[i]/F[i]+L*r[i]

# 		alpha[i]=a[i]+P[i]*r[i-1]
# 		Nn[i-1] = 1/F[i]+L**2*Nn[i]
# 		V[i]=P[i]-P[i]**2*Nn[i-1]

# 	return alpha,V


















