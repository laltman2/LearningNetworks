import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import numpy.linalg as la
from Node import Node
from utils.optimize import FreeState_node, JErg, JXGrad

class Elastic():
    
    def __init__(self, nodes = np.array([[0.,0.], [0.,1.]]), 
                 edges = np.array([[0,1]]),
                 boundary = [Node()],
                 sources = [Node()],
                 targets = [Node()]):
        # nodes: positions of all nodes of the system (including fixed points/boundaries)
        # edges: connectivity graph of shape (NE, 2) with index values of nodes
        # boundary: list of Nodes which never move
        # sources: list of Nodes which are restricted in both the free and clamped states
        # targets: list of Nodes which are restricted only in the clamped state
                 
        # setup network architecture
        self.dim = nodes.shape[1]
        self.NN = len(nodes)
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
        self.RLS = np.ones(self.NE, dtype=np.float32)
        self.KS = np.ones(self.NE, dtype=np.float32)
        
        # set boundary, sources, and targets
        self.boundary = boundary
        self.sources = sources
        self.targets = targets
        
        # energy scaling and loss normalization
        self.Epow = 2.
        self.lnorm = 2.
        
        # training hyperparameters
        self.eta = 1.
        self.alpha = 1.
        self.currentstep = -1
        self.stop_train = False
        
        # history dataframe
        self.history = pd.DataFrame(columns=['step', 
                                             'free_in', 'free_out', 'clamp_in', 'clamp_out', 'cost',
                                              'RLS', 'KS' 'exts_f', 'exts_c'])
        
        # initial equilibrium state
        self.x0 = self.eq_state(nodes)
    
    def plot_state(self, pos, ax=None, save=None):
        if ax == None:
            fig, ax = plt.subplots()

        pos = pos.reshape(self.NN,2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        for sn in self.sources:
            ax.scatter([pos[sn.index][0]], [pos[sn.index][1]], c='b', label='source node: {}'.format(sn.index), zorder=10)
        
        for tn in self.targets:
            ax.scatter([pos[tn.index][0]], [pos[tn.index][1]], c='r', label='target node: {}'.format(tn.index), zorder=10)
        
        
        for j in range(self.NN):
            ax.scatter([pos[j][0]], [pos[j][1]], c='k')

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
        x00 = pos.flatten()
        
        fixed = []
        for bn in self.boundary:
            for d in range(self.dim):
                if bn.fixed[d]:
                    fixed.append(bn.index*self.dim + d)
                
        
        fixedNodes = np.array([bn.index for bn in self.boundary])
        fixedPos = np.array([bn.position for bn in self.boundary])
        
        params = [self.KS, self.RLS, self.EI, self.EJ, self.BIJ, self.dim, self.Epow, self.lnorm, fixed]
#         print('params', params)
        x0 = np.array(FreeState_node(x00, params, fixedNodes, fixedPos, JErg, JXGrad))
        return x0
            
    def free_state(self, inputpos=None, plot=False):
        # inputpos: array of shape (# sources, dim)
        # if None, keep the position(s) already in the source node object(s)
        if inputpos:
            for ix in range(len(self.sources)):
                self.sources[ix].position = inputpos[ix]
        
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
        
        params = [self.KS, self.RLS, self.EI, self.EJ, self.BIJ, self.dim, self.Epow, self.lnorm, fixed]
        FS = np.array(FreeState_node(self.x0, params, fixedNodes, fixedPos, JErg, JXGrad))
        
        if plot:
            self.plot_state(FS)
        
        return FS


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