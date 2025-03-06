import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg.interpolative as sli
from numpy import linalg as LA

def ID_test():

#  The number of Gaussian panels on each smooth component.
     Npan      = 200
# The number of Gaussian nodes on each panel.     
     Ngau      = 10    
     # The tolerance in the HSS compression.
     acc       = 1e-8

# create geometry
     [C,ww,tt] = construct_geom(Npan,Ngau)
      
# create indices of interest
     ind = np.arange(1750,1999,1)
     indoffd = np.arange(0,1749,1)

# create matrix we are trying to approximate         
     Aall = construct_A_offd(C,ww,ind,indoffd)
     
# create the ID factorization
     [k, J, proj] = sli.interp_decomp(np.transpose(Aall), acc)
     
# validate the result
     B = sli.reconstruct_skel_matrix(np.transpose(Aall), k, J)
     B = np.transpose(B)
     P = sli.reconstruct_interp_matrix(J, proj)
     P = np.transpose(P)
     
     err = LA.norm(Aall-np.dot(P,B))
     print('err is:', err)
          
# plot the geometry          
     plt.figure()    
     plt.plot(C[0,ind],C[3,ind],'r-')
     plt.plot(C[0,indoffd],C[3,indoffd],'b-')
     plt.plot(C[0,ind[J[1:k]]],C[3,ind[J[1:k]]],'cx')
     plt.show()
        
     return 

def construct_A_offd(C,ww,ind1,ind2):

     n1   = int(len(ind1))
     n2   = int(len(ind2))
     dd1  = np.outer(C[0,ind1] ,np.ones((1,n2))) - \
            np.outer(np.ones((1,n1)), C[0,ind2])
     dd2  = np.outer(C[3,ind1],np.ones((1,n2))) - \
            np.outer(np.ones((1,n1)), C[3,ind2])
     ddsq = dd1*dd1 + dd2*dd2
     tmp1 = C[4,ind2]/np.sqrt(C[1,ind2]*C[1,ind2] + C[4,ind2]*C[4,ind2])
     nn1  = np.outer(np.ones((1,n1)),tmp1)
     tmp2 = C[1,ind2]/np.sqrt(C[1,ind2]*C[1,ind2] + C[4,ind2]*C[4,ind2])
     nn2  = np.outer(np.ones((1,n1)),-tmp2)
     B    = -(1/(2*np.pi))*(nn1*dd1 + nn2*dd2)/ddsq
     B    = B*np.outer(np.ones((1,n1)),ww[ind2])

     return B



def  construct_geom(Npan,Ngau):

     [tt,ww] = get_gauss_nodes(Ngau,Npan)
     ntot = int(Npan*Ngau)
     C = np.zeros((6,ntot))
     C[0,:] = (1+0.3*np.cos(5*2*np.pi*tt))*np.cos(2*np.pi*tt)
     C[1,:] = 2*np.pi*(-np.sin(2*np.pi*tt)*(0.3*np.cos(5*2*np.pi*tt)+1)- \
               1.5*np.sin(5*2*np.pi*tt)*np.cos(2*np.pi*tt))
     C[2,:] =  (2*np.pi)**2*(3*np.sin(2*np.pi*tt)*np.sin(5*2*np.pi*tt)+ \
               np.cos(2*np.pi*tt)*(-7.8*np.cos(5*2*np.pi*tt)-1))
     C[3,:] =  np.sin(2*np.pi*tt)*(1+0.3*np.cos(5*2*np.pi*tt))
     C[4,:] =  2*np.pi*(-1.5*np.sin(2*np.pi*tt)*np.sin(5*2*np.pi*tt) \
               +0.3*np.cos(5*2*np.pi*tt)*np.cos(2*np.pi*tt)+np.cos(2*np.pi*tt))
     C[5,:]  = (2*np.pi)**2*(np.sin(2*np.pi*tt)*(-7.8*np.cos(5*2*np.pi*tt)-1) \
               -3*np.sin(5*2*np.pi*tt)*np.cos(2*np.pi*tt))

     ww = ww*np.sqrt(C[1,:]*C[1,:] + C[4,:]*C[4,:])

     return [C,ww,tt]


def get_gauss_nodes(Ngau,Npan):

     tsta         = 0 
     tend         = 1 
     h = (tend-tsta)/Npan
     ttpanels = np.linspace(tsta,tend,Npan+1)
     tt = np.zeros(Npan*Ngau)
     ww = np.zeros(Npan*Ngau)
# reference quad     
     [tref,wref] = lgwt(Ngau,0,1)
     
     nvec = np.arange(0,Ngau)

     for j in range(Npan):
       h       = ttpanels[j+1] - ttpanels[j]
       ind     = j*Ngau + nvec
       tt[ind] = ttpanels[j] + h*np.flip(tref)
       ww[ind] = h*np.flip(wref)

     return[tt,ww]

# gauss legendre quadrature nodes and weights via
# Newton iteration


def lgwt(N,a,b):
  """ 
   This script is for computing definite integrals using Legendre-Gauss 
   Quadrature. Computes the Legendre-Gauss nodes and weights  on an interval
   [a,b] with truncation order N
  
   Suppose you have a continuous function f(x) which is defined on [a,b]
   which you can evaluate at any x in [a,b]. Simply evaluate it at all of
   the values contained in the x vector to obtain a vector f. Then compute
   the definite integral using np.sum(f*w)
  
   Written by Greg von Winckel - 02/25/2004
   translated to Python - 10/30/2022
  """
  N = N-1
  N1 = N+1
  N2 = N+2
  eps = np.finfo(float).eps  
  xu = np.linspace(-1,1,N1)
  
  # Initial guess
  y = np.cos((2*np.arange(0,N1)+1)*np.pi/(2*N+2))+(0.27/N1)*np.sin(np.pi*xu*N/N2)

  # Legendre-Gauss Vandermonde Matrix
  L = np.zeros((N1,N2))
  
  # Compute the zeros of the N+1 Legendre Polynomial
  # using the recursion relation and the Newton-Raphson method
  
  y0 = 2.
  one = np.ones((N1,))
  zero = np.zeros((N1,))

  # Iterate until new points are uniformly within epsilon of old points
  while np.max(np.abs(y-y0)) > eps:
      
    L[:,0] = one
    
    L[:,1] = y
    for k in range(2,N1+1): 
      L[:,k] = ((2*k-1)*y*L[:,k-1]-(k-1)*L[:,k-2])/k
    
    lp = N2*(L[:,N1-1]-y*L[:,N2-1])/(1-y**2)   
    
    y0 = y
    y = y0-L[:,N2-1]/lp
    
    
  
  # Linear map from[-1,1] to [a,b]
  x=(a*(1-y)+b*(1+y))/2
  
  # Compute the weights
  w=(b-a)/((1-y**2)*lp**2)*(N2/N1)**2
  return [x,w]
  
  
  


ID_test()
