import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
import numpy.linalg as la
from Node import Node
# from utils.optimize import Dists, FreeState_node, JErg, JXGrad
from utils.optimize import *


class Elastic(object):
    
    def __init__(self, nodes = np.array([[0.,0.], [0.,1.]]), 
                 edges = np.array([[0,1]]),
                 boundary = [],
                 sources = [],
                 targets = []):
        # nodes: positions of all nodes of the system (including fixed points/boundaries)
        # edges: connectivity graph of shape (NE, 2) with index values of nodes
        # boundary: list of Nodes which never move
        # sources: list of Nodes which are restricted in both the free and clamped states
        # targets: list of Nodes which are restricted only in the clamped state
                 
        # setup network architecture
        self.NN, self.dim = nodes.shape
        self.NE = len(edges)
        self.EI, self.EJ = np.transpose(edges)
        
        # populate unit vectors for every edge
        BIJ = []
        for i in range(self.NE):
            ei = self.EI[i]
            ej = self.EJ[i]
            bij = nodes[ej] - nodes[ei]
            BIJ.append(bij/la.norm(bij))
            
        # get the free state positions, when only input constraints are applied
        self.BIJ = np.array(BIJ)
        
        # set initial configuration of learning degrees of freedom
        self._RLS = np.ones(self.NE, dtype=np.float32)
        self._KS = np.ones(self.NE, dtype=np.float32)
        
        # set boundary, sources, and targets
        self.boundary = boundary
        self.sources = sources
        self.targets = targets
        
        # energy scaling and loss normalization
        self.Epow = 2.
        self.lnorm = 2.
        
        # training hyperparameters
        self._eta = 1.
        self._alpha = 1.
        self.currentstep = -1
        self.stop_train = False
        
        # history dataframe
        self.history = pd.DataFrame(columns=['step', 
                                             'free_in', 'free_out', 'clamp_in', 'clamp_out', 'cost', 'loss',
                                              'RLS', 'KS', 'exts_f', 'exts_c'])
        
        # initial equilibrium state
        self.x0 = self.eq_state(nodes)
        
    @property
    def KS(self):
        return self._KS
        
    @KS.setter
    def KS(self, value):
        self._KS = value
        self.x0 = self.eq_state(self.x0)
        
    @property
    def RLS(self):
        return self._RLS
        
    @RLS.setter
    def RLS(self, value):
        self._RLS = value
        self.x0 = self.eq_state(self.x0)
        
    @property
    def eta(self):
        return self._eta
    
    @eta.setter
    def eta(self, value):
        self._eta = value
        
    @property
    def alpha(self):
        return self._alpha
    
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
    
    def plot_state(self, pos, ax=None, save=None):
        if ax == None:
            fig, ax = plt.subplots()

        if pos.size == self.NN*self.dim:
            pos = pos.reshape(self.NN,self.dim)
        else:
            raise Exception("Invalid position size. Expecting size {}, got size {}".format(self.NN*self.dim, pos.size))
            
        ax.set_aspect('equal')
        ax.axis('off')
        
        for bn in self.boundary:
            ax.scatter([pos[bn.index][0]], [pos[bn.index][1]], c='gray', label='source node: {}'.format(bn.index), s = 100, zorder=10)
        
        for sn in self.sources:
            ax.scatter([pos[sn.index][0]], [pos[sn.index][1]], c='b', label='source node: {}'.format(sn.index), s = 100, zorder=10)
        
        for tn in self.targets:
            ax.scatter([pos[tn.index][0]], [pos[tn.index][1]], c='r', label='target node: {}'.format(tn.index), s=100, zorder=10)
        
        
        for j in range(self.NN):
            ax.scatter([pos[j][0]], [pos[j][1]], s=50, c='k')

        for i in range(self.NE):
            ei = self.EI[i]
            ej = self.EJ[i]
            ax.plot([pos[ei][0], pos[ej][0]], [pos[ei][1], pos[ej][1]], 'k')
        
        if save:
            fig.tight_layout()
            fig.savefig(save)
        if ax==None:
            plt.show()
                     
    def eq_state(self, pos):
        # initial node positions (before equilibration)
        if pos.size == self.NN*self.dim:
            x00 = pos.flatten()
        else:
            raise Exception("Invalid position size. Expecting size {}, got size {}".format(self.NN*self.dim, pos.size))
            
        fixed = []
        for bn in self.boundary:
            for d in range(self.dim):
                if bn.fixed[d]:
                    fixed.append(bn.index*self.dim + d)
                
        
        fixedNodes = np.array([bn.index for bn in self.boundary])
        fixedPos = np.array([bn.position for bn in self.boundary])
        
        params = [self._KS, self._RLS, self.EI, self.EJ, self.BIJ, self.dim, self.Epow, self.lnorm, fixed]
        computeProblem = True
        while computeProblem:
            x0 = np.array(FreeState_node(x00, params, fixedNodes, fixedPos, JErg, JXGrad))
            if np.isnan(x0).any():
                # if you get an error, jiggle the nodes a little bit
                x00 += np.random.normal(0.0, 0.01)
            else:
                computeProblem = False
        return x0
            
    def free_state(self, inputpos=None, plot=False):
        # inputpos: array of shape (# sources, dim)
        # if None, keep the position(s) already in the source node object(s)
        if inputpos is not None:
            if inputpos.shape == (len(self.sources), self.dim):
                for ix in range(len(self.sources)):
                    self.sources[ix].position = inputpos[ix]
            else:
                raise Exception("Invalid inputpos shape. Expecting array of shape {}, got shape {}".format((len(self.sources), self.dim), inputpos.shape))
        
        fixed = []
        for bn in self.boundary:
            for d in range(self.dim):
                if bn.fixed[d]:
                    fixed.append(bn.index*self.dim + d)
                    
        for sn in self.sources:
            for d in range(self.dim):
                if sn.fixed[d]:
                    fixed.append(sn.index*self.dim + d)

        
        fixedNodes = np.concatenate(([bn.index for bn in self.boundary], [sn.index for sn in self.sources]), axis=0)
        fixedPos = np.concatenate(([bn.position for bn in self.boundary], [sn.position for sn in self.sources]), axis=0)
        
        params = [self._KS, self._RLS, self.EI, self.EJ, self.BIJ, self.dim, self.Epow, self.lnorm, fixed]
        FS = np.array(FreeState_node(self.x0, params, fixedNodes, fixedPos, JErg, JXGrad))
        
        if plot:
            self.plot_state(FS)
        
        return FS
    
    def clamped_state(self, inputpos=None, outputpos=None, plot=False):
        # inputpos/outputpos: arrays of shape (# sources/targets, dim)
        # if None, keep the position(s) already in the source/target node object(s)
        if inputpos is not None:
            if inputpos.shape == (len(self.sources), self.dim):
                for ix in range(len(self.sources)):
                    self.sources[ix].position = inputpos[ix]
            else:
                raise Exception("Invalid inputpos shape. Expecting array of shape {}, got shape {}".format((len(self.sources), self.dim), inputpos.shape))
                
        if outputpos is not None:
            if outputpos.shape == (len(self.targets), self.dim):
                for ix in range(len(self.targets)):
                    self.targets[ix].position = outputpos[ix]
            else:
                raise Exception("Invalid outputpos shape. Expecting array of shape {}, got shape {}".format((len(self.targets), self.dim), outputpos.shape))
        
        fixed = []
        for bn in self.boundary:
            for d in range(self.dim):
                if bn.fixed[d]:
                    fixed.append(bn.index*self.dim + d)
                    
        for sn in self.sources:
            for d in range(self.dim):
                if sn.fixed[d]:
                    fixed.append(sn.index*self.dim + d)
                    
        for tn in self.targets:
            for d in range(self.dim):
                if tn.fixed[d]:
                    fixed.append(tn.index*self.dim + d)

        fixedNodes = np.concatenate(([bn.index for bn in self.boundary], [sn.index for sn in self.sources], [tn.index for tn in self.targets]), axis=0)
        fixedPos = np.concatenate(([bn.position for bn in self.boundary], [sn.position for sn in self.sources], [tn.position for sn in self.targets]), axis=0)
        
        if len(fixedNodes) == self.NN:
            # if the system is fully constrained, just use constraints as the position (compute won't work)
            # reorder fixedPos first
            orderedfixed = np.array([pos for _, pos in sorted(zip(fixedNodes, fixedPos))])
            CS = orderedfixed.flatten()
        else:
            params = [self._KS, self._RLS, self.EI, self.EJ, self.BIJ, self.dim, self.Epow, self.lnorm, fixed]
            CS = np.array(FreeState_node(self.x0, params, fixedNodes, fixedPos, JErg, JXGrad))
        
        if plot:
            self.plot_state(CS)
        
        return CS
    
    def get_exts(self, pos):
        if pos.size == self.NN*self.dim:
            return np.array(Dists(pos, self.EI, self.EJ, self.dim, self.lnorm))
        else:
            raise Exception("Invalid position size. Expecting size {}, got size {}".format(self.NN*self.dim, pos.size))
    
    def free_output(self, inputpos):
        #evaluate model (shortcut for trained models)
        FS = self.free_state(inputpos)
        FSdim = FS.reshape(self.NN, self.dim)
        return np.array([FSdim[tn.index] for tn in self.targets])
    
    def loss(self, freeout, desiredout):
        #squared error of the outputs
        #freeout, desiredout: arrays of shape (# targets, dim)
        if freeout.shape != len(self.targets), self.dim:
            raise Exception("Invalid freeout shape. Expecting shape {}, got shape {}".format((len(self.targets), self.dim), freeout.shape)
        if desiredout.shape != len(self.targets), self.dim:
            raise Exception("Invalid desiredout shape. Expecting shape {}, got shape {}".format((len(self.targets), self.dim), desiredout.shape)
        isfixed = np.invert(np.array([tn.fixed for tn in self.targets]))
        diffouts = desiredout - freeout
        np.place(diffouts, isfixed, 0.)
        return np.sum(np.square(diffouts))
    
    def cost(self, freepos, clamppos):
        #energy cost between free and clamped network states
        #freepos, clamppos: 1D arrays of size (NN*dim)
        EC = Energy(clamppos, self.KS, self.RLS, self.EI, self.EJ, self.dim, self.Epow, self.lnorm)
        EF = Energy(freepos, self.KS, self.RLS, self.EI, self.EJ, self.dim, self.Epow, self.lnorm)
        return EC - EF
    
    def update(self, exts_f, exts_c, rule = 'RLS', clip = [0.5, 1.5]):
        row = self.history.loc[self.currentstep]

        #update rule
        if rule == 'RLS':
            dLs = self._alpha*np.multiply(self.KS, (exts_c - exts_f))
            self.RLS += dLs
            #clip rest lengths between max and min values
            if clip is not None:
                self.RLS = np.clip(self.RLS, clip[0], clip[1])
        elif rule == 'KS':
            if clip is not None:
                self.KS = np.clip(self.KS, clip[0], clip[1])
        else:
            raise Exception('Learning rule not recongized')

        #update equilibrium state
        self.x0 = self.eq_state(self.x0)
        
    def initialize_step(self):
        self.currentstep += 1
        #start a new row
        self.history.loc[self.currentstep] = [self.currentstep] + [None]*(len(self.history.columns)-1)
     
    def learning_step(self, iopair):
        #iopair: tuple of: array of shape(# sources, dim), array of shape(#targets, dim)
        desiredin, desiredout = iopair
        if desiredin.shape != (len(self.sources), self.dim):
            raise Exception("Invalid input shape. Expecting shape {}, got shape {}".format((len(self.sources), self.dim), desiredin.shape)
        if desiredout.shape != (len(self.targets), self.dim):
            raise Exception("Invalid output shape. Expecting shape {}, got shape {}".format((len(self.targets), self.dim), desiredout.shape)
        
        self.initialize_step()
        eq = self.x0.reshape(self.NN,self.dim)
        refin = [eq[sn.index] for sn in self.sources]
        refout = [eq[tn.index] for tn in self.targets]
        row = self.history.loc[self.currentstep]
        row.RLS = json.dumps(self.RLS.tolist())
        row.KS = json.dumps(self.KS.tolist())
        
        #apply free state
        FS = self.free_state(desiredin)
        exts_f = self.get_exts(FS)
        row.exts_f = json.dumps(exts_f.tolist())
        FSdim = FS.reshape(self.NN,self.dim)
        row.free_in = json.dumps([FSdim[sn.index].tolist() for sn in self.sources])
        freeout = np.array([FSdim[tn.index] for tn in self.targets])
        row.free_out = json.dumps([FSdim[tn.index].tolist() for tn in self.targets])
        
        #apply clamped state
        clampout = self._eta*desiredout + (1-self._eta)*freeout
        
        CS = self.clamped_state(desiredin, clampout)
        if np.isnan(CS).any():
            self.plot_state(self.x0)
            self.stop_train = True
        exts_c = self.get_exts(CS)
        row.exts_c = json.dumps(exts_c.tolist())
        CSdim = CS.reshape(self.NN,self.dim)
        row.clamp_in = json.dumps([CSdim[sn.index].tolist() for sn in self.sources])
        row.clamp_out = json.dumps([CSdim[tn.index].tolist() for tn in self.targets])
        
        row.loss = self.loss(freeout, desiredout)
        row.cost = self.cost(FS, CS)
        self.update(exts_f, exts_c)
        
        self.history.loc[self.currentstep] = row #fix this later
        
    def train(self, iopairs = [], Nsteps = 100):
        #iopairs: list of tuples, length (# datapoints)
        #Nsteps: integer, how many times to cycle through all data points
        pairindex = 0
        
        while (self.currentstep+1 < Nsteps*len(iopairs)) and self.stop_train==False:
            pairindex = pairindex % len(iopairs)
            self.learning_step(iopairs[pairindex])
#             all_outs = []
#             for inp in np.array(iopairs)[:,0]:
#                 FS = self.free_state(inp)
#                 all_outs.append(FS[6])
#             self.history.at[self.currentstep, 'all_outs'] = json.dumps(all_outs)
            pairindex += 1

if __name__=='__main__':
    import warnings
    warnings.filterwarnings("ignore")
    
    pos = np.array([[0.,0.], [0.,1.], [0., 2.5]])
    bdy = [Node(0, pos[0])]
    source = [Node(2, pos[2])]
    target = [Node(1, pos[1])]

    e = Elastic(nodes = pos, edges = np.array([[0,1], [1,2]]), boundary = bdy, sources = source, targets=target)

    print('Equilibrium state:', e.x0)
    print('Free state:', e.free_state())
