import numpy as np
import numba
from jax import grad, jit, vmap
import jax.numpy as jnp

alpha0 = 0.1
Ndelay = 5
Nmax = 10000
finc = 1.1
fdec = 0.5
fa = 0.99
Nnegmax = 2000


def Dists(pos, EI, EJ, dim=2, lnorm=2):
    NN = len(pos) // dim
    Pos = jnp.reshape(pos, [NN, dim])
    PmP = Pos[EJ] - Pos[EI]
    DS = jnp.sum(jnp.abs(PmP)**lnorm, axis=1)**(1./lnorm)
    return DS

def Energy(pos, KS, RLS, EI, EJ, dim=2, Epow=2, lnorm=2):
    #NE = len(EI)
#     NN = len(pos) // dim
#     Pos = jnp.reshape(pos, [NN, dim])
#     PmP = Pos[EJ] - Pos[EI]
#     DS = jnp.sum(jnp.abs(PmP)**lnorm, axis=1)**(1./lnorm)
    DS = Dists(pos, EI, EJ, dim=dim, lnorm=lnorm)
    ES = 0.5 * KS * (DS - RLS)**Epow
    return jnp.sum(ES)

JErg = jit(Energy)
XGrad = grad(Energy)
JXGrad = jit(XGrad)

@numba.jit()
def optimize_fire(x0,f,df,params,atol=1e-4,dt = 0.002,logoutput=False):
    error = 10*atol 
    dtmax = 10*dt
    dtmin = 0.02*dt
    alpha = alpha0
    Npos = 0
    
    [KS, RLS, EI, EJ, BIJ, dim, Epow, lnorm, fixedNodes] = params

    x = x0.copy()
    V = np.zeros(x.shape)
    F = -np.array(df(x, KS, RLS, EI, EJ))
    F[fixedNodes] = 0.

    for i in range(Nmax):

        P = (F*V).sum() # dissipated power
        
        if (P>0):
            Npos = Npos + 1
            if Npos>Ndelay:
                dt = min(dt*finc,dtmax)
                alpha = alpha*fa
        else:
            Npos = 0
            dt = max(dt*fdec,dtmin)
            alpha = alpha0
            V = np.zeros(x.shape)

        V = V + 0.5*dt*F
        #V = (1-alpha)*V + alpha*F*linalg.norm(V)/linalg.norm(F)
        nV = np.sum(V**2)
        nF = np.sum(F**2)
        V = (1-alpha)*V + alpha*F*nV/nF
        x = x + dt*V
        F = -np.array(df(x, KS, RLS, EI, EJ))
        F[fixedNodes] = 0.
        V = V + 0.5*dt*F

        error = max(abs(F))
        if error < atol: break
        if logoutput: print(f(x, KS, RLS, EI, EJ),error)
    return [x,f(x, KS, RLS, EI, EJ),i]

@numba.jit()
def CalcState(x0, params, f, df, logoutput=False):
    [xmin,fmin,Niter] = optimize_fire(x0,f,df,params,atol=1e-5,dt=0.02,logoutput=logoutput)
    return xmin

#Free State function, for when the inputs/outputs are node displacements from their equilibrium positions
@numba.jit()
def FreeState_node(x0, params, SourceNodes, SourcePos, f, df):
    [KS, RLS, EI, EJ, BIJ, dim, Epow, lnorm, fixedNodes] = params
    KFree = KS.copy()
    RLFree = RLS.copy()
    params = [KFree, RLFree, EI, EJ, BIJ, dim, Epow, lnorm, fixedNodes]
    
    pos0 = x0.copy()
    for i in range(len(SourceNodes)):
        pos0[(SourceNodes[i]*dim)] = SourcePos[i][0]
        pos0[(SourceNodes[i]*dim)+1] = SourcePos[i][1]

    xmin = CalcState(pos0, params, f, df, logoutput=False)

    return xmin

@numba.jit()
def FreeState_edge(x0, params, SourceEdges, SourceStrains, f, df):
    
    [KS, RLS, EI, EJ, BIJ, dim, Epow, lnorm, fixedNodes] = params
    KFree = KS.copy()
    #KFree[SourceEdges] = 20.
    RLFree = RLS.copy()
    #RLFree[SourceEdges] = RLS[SourceEdges] * (1 + SourceStrains)
    params = [KFree, RLFree, EI, EJ, BIJ, dim, Epow, lnorm, fixedNodes]
    
    pos0 = x0.copy()
    for i in range(len(SourceEdges)):
        pos0[EI[SourceEdges[i]]*dim : (EI[SourceEdges[i]]+1)*dim] -= BIJ[SourceEdges[i]] * SourceStrains[i]*RLS[SourceEdges[i]]/2
        pos0[EJ[SourceEdges[i]]*dim : (EJ[SourceEdges[i]]+1)*dim] += BIJ[SourceEdges[i]] * SourceStrains[i]*RLS[SourceEdges[i]]/2
    
    xmin = CalcState(pos0, params, f, df)
    return xmin

if __name__ == "__main__":
    pass
