import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
import numpy.linalg as la
from LearningNetworks.Node import Node
from LearningNetworks.EdgeTask import EdgeTask
from LearningNetworks.NodeTask import NodeTask
# from utils.optimize import Dists, FreeState_node, JErg, JXGrad
from LearningNetworks.utils.optimize import *


class Elastic(object):
    
    def __init__(self, nodes = np.array([[0.,0.], [0.,1.]]), 
                 edges = np.array([[0,1]]),
                 boundary = [],
                 tasks=[]):
        # nodes: positions of all nodes of the system (including fixed points/boundaries)
        # edges: connectivity graph of shape (NE, 2) with index values of nodes
        # boundary: list of Nodes which never move
        # tasks: list of Tasks with specified node indices, iopairs, fixed bools
                 
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
        self._boundary = boundary
        self.tasks = tasks
        
        self.fixededges = []
        
        # energy scaling and loss normalization
        self.Epow = 2.
        self.lnorm = 2.
        
        # training hyperparameters
        self._eta = 1.
        self._alpha = 1.
        self.currentstep = -1
        self.stop_train = False
        
        # history dataframe
        self.history = pd.DataFrame(columns=['step', 'task',
                                             'free_in', 'free_out', 'clamp_in', 'clamp_out', 
                                             'cost', 'loss',
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
    def boundary(self):
        return self._boundary
        
    @RLS.setter
    def boundary(self, value):
        self._boundary = value
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
    
    def plot_state(self, pos, ax=None, save=None, **kwargs):
        if ax == None:
            fig, ax = plt.subplots()

        if pos.size == self.NN*self.dim:
            pos = pos.reshape(self.NN,self.dim)
        else:
            raise Exception("Invalid position size. Expecting size {}, got size {}".format(self.NN*self.dim, pos.size))
            
        ax.set_aspect('equal')
        ax.axis('off')
        
        for bn in self._boundary:
            ax.scatter([pos[bn.index][0]], [pos[bn.index][1]], c='gray', label='boundary node: {}'.format(bn.index), s = 100, zorder=10, **kwargs)
        
        for task in self.tasks:
            if isinstance(task, NodeTask):
                for sn in task.sourcenodes:
                    ax.scatter([pos[sn][0]], [pos[sn][1]], c='b', label='source node: {}'.format(sn), s = 100, zorder=10, **kwargs)

                for tn in task.targetnodes:
                    ax.scatter([pos[tn][0]], [pos[tn][1]], c='r', label='target node: {}'.format(tn), s=100, zorder=10, **kwargs)

            if isinstance(task, EdgeTask):
                for se in task.sourceedges:
                    ei = self.EI[se]
                    ej = self.EJ[se]
                    ax.plot([pos[ei][0], pos[ej][0]], [pos[ei][1], pos[ej][1]], color='lightblue', linewidth=3, zorder=10, **kwargs)
#                     ax.scatter([pos[self.EI[se]][0]], [pos[self.EI[se]][1]], c='lightblue', label='source edge: {}, side 1'.format(se), s = 100, zorder=10, **kwargs)
#                     ax.scatter([pos[self.EJ[se]][0]], [pos[self.EJ[se]][1]], c='lightblue', label='source edge: {}, side 2'.format(se), s = 100, zorder=10, **kwargs)

                for te in task.targetedges:
                    ei = self.EI[te]
                    ej = self.EJ[te]
                    ax.plot([pos[ei][0], pos[ej][0]], [pos[ei][1], pos[ej][1]], color='pink', linewidth=3, zorder=10, **kwargs)
#                     ax.scatter([pos[self.EI[te]][0]], [pos[self.EI[te]][1]], c='pink', label='target edge: {}, side 1'.format(te), s=100, zorder=10, **kwargs)
#                     ax.scatter([pos[self.EJ[te]][0]], [pos[self.EJ[te]][1]], c='pink', label='target node: {}, side 2'.format(te), s=100, zorder=10, **kwargs)
        
        for j in range(self.NN):
            ax.scatter([pos[j][0]], [pos[j][1]], s=50, c='k', **kwargs)

        for i in range(self.NE):
            ei = self.EI[i]
            ej = self.EJ[i]
            ax.plot([pos[ei][0], pos[ej][0]], [pos[ei][1], pos[ej][1]], 'k', **kwargs)
        
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
        for bn in self._boundary:
            for d in range(self.dim):
                if bn.fixed[d]:
                    fixed.append(bn.index*self.dim + d)
                
        
        fixedNodes = np.array([bn.index for bn in self._boundary])
        fixedPos = np.array([bn.position for bn in self._boundary])
        
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
            
    def free_state(self, taskIndex=0, inputs=None, pairIndex=0, plot=False, **kwargs):
        # taskIndex: choose from your list of tasks, defaults to 0
        # inputs: array of shape (# sources, dim)
        # if None, fall back to pairIndex (from IOpairs list in task)
        
        if not self.tasks:
            raise Exception("No tasks added")
            return
        
        task = self.tasks[taskIndex]
        sources = task.sources(inputs, pairIndex)
        
        fixed = []
        for bn in self._boundary:
            for d in range(self.dim):
                if bn.fixed[d]:
                    fixed.append(bn.index*self.dim + d)
             
        if isinstance(task, NodeTask):
            for sn in sources:
                for d in range(self.dim):
                    if sn.fixed[d]:
                        fixed.append(sn.index*self.dim + d)

            fixedNodes = np.concatenate(([bn.index for bn in self._boundary], [sn.index for sn in sources]), axis=0)
            fixedPos = np.concatenate(([bn.position for bn in self._boundary], [sn.position for sn in sources]), axis=0)

            params = [self._KS, self._RLS, self.EI, self.EJ, self.BIJ, self.dim, self.Epow, self.lnorm, fixed]
            FS = np.array(FreeState_node(self.x0, params, fixedNodes, fixedPos, JErg, JXGrad))
        
        
        if isinstance(task, EdgeTask):
            for se in sources:
                for d in range(self.dim):
                    fixed.append(self.EI[se.index]*self.dim + d)
                    fixed.append(self.EJ[se.index]*self.dim + d)

            fixedEdges = [se.index for se in sources]
            fixedStrains = [se.strain for se in sources]

            params = [self._KS, self._RLS, self.EI, self.EJ, self.BIJ, self.dim, self.Epow, self.lnorm, fixed]
            FS = np.array(FreeState_edge(self.x0, params, fixedEdges, fixedStrains, JErg, JXGrad))
        
        if plot:
            self.plot_state(FS, **kwargs)
        
        return FS
    
    def clamped_state(self, taskIndex=0, IOpair=None, pairIndex=0, plot=False, **kwargs):
        # taskIndex: choose from your list of tasks, defaults to 0
        # IOpair: tuple: array of shape (# sources, dim),  array of shape (#targets, dim)
        # if None, fall back to pairIndex (from IOpairs list in task)
        if not self.tasks:
            raise Exception("No tasks added")
            return

        task = self.tasks[taskIndex]
        
        if IOpair:
            inputs, outputs = IOpair
        else:
            inputs, outputs = None, None
            
        sources = task.sources(inputs, pairIndex)
        targets = task.targets(outputs, pairIndex)
        
        fixed = []
        for bn in self._boundary:
            for d in range(self.dim):
                if bn.fixed[d]:
                    fixed.append(bn.index*self.dim + d)
              
        if isinstance(task, NodeTask):
            for sn in sources:
                for d in range(self.dim):
                    if sn.fixed[d]:
                        fixed.append(sn.index*self.dim + d)

            for tn in targets:
                for d in range(self.dim):
                    if tn.fixed[d]:
                        fixed.append(tn.index*self.dim + d)

            fixedNodes = np.concatenate(([bn.index for bn in self._boundary], [sn.index for sn in sources], [tn.index for tn in targets]), axis=0)
            fixedPos = np.concatenate(([bn.position for bn in self._boundary], [sn.position for sn in sources], [tn.position for tn in targets]), axis=0)

            if len(fixed) > (self.NN-1)*self.dim:
                # if the system is (almost?) fully constrained, just use constraints as the position (compute won't work)
    #             print('Too many constraints, falling back to default positions')
                # remove duplicates and reorder
                _, ix = np.unique(fixedNodes, return_index=True)
                orderedfixed = fixedPos[ix]
    #             orderedfixed = np.array([pos for _, pos in sorted(zip(fixedNodes, fixedPos))])
                CS = orderedfixed.flatten()
            else:
                params = [self._KS, self._RLS, self.EI, self.EJ, self.BIJ, self.dim, self.Epow, self.lnorm, fixed]
                CS = np.array(FreeState_node(self.x0, params, fixedNodes, fixedPos, JErg, JXGrad))
        
        if isinstance(task, EdgeTask):
            for se in sources:
                for d in range(self.dim):
                    fixed.append(self.EI[se.index]*self.dim + d)
                    fixed.append(self.EJ[se.index]*self.dim + d)
            
            for te in targets:
                for d in range(self.dim):
                    fixed.append(self.EI[te.index]*self.dim + d)
                    fixed.append(self.EJ[te.index]*self.dim + d)
        
            fixedEdges = np.concatenate(([se.index for se in sources], [te.index for te in targets]), axis=0)
            fixedStrains = np.concatenate(([se.strain for se in sources], [te.strain for te in targets]), axis=0)
            
            params = [self._KS, self._RLS, self.EI, self.EJ, self.BIJ, self.dim, self.Epow, self.lnorm, fixed]
            CS = np.array(FreeState_edge(self.x0, params, fixedEdges, fixedStrains, JErg, JXGrad))
            
        if plot:
            self.plot_state(CS, **kwargs)
        
        return CS
    
    def get_exts(self, pos):
        if pos.size == self.NN*self.dim:
            return np.array(Dists(pos, self.EI, self.EJ, self.dim, self.lnorm))
        else:
            raise Exception("Invalid position size. Expecting size {}, got size {}".format(self.NN*self.dim, pos.size))
    
    def free_output(self, taskIndex=0, inputs=None, pairIndex=0):
        #evaluate model (shortcut for trained models)
        FS = self.free_state(taskIndex, inputs, pairIndex)
        if isinstance(self.tasks[taskIndex], NodeTask):
            FSdim = FS.reshape(self.NN, self.dim)
            freeout = np.array([FSdim[tn] for tn in self.tasks[taskIndex].targetnodes])
        if isinstance(self.tasks[taskIndex], EdgeTask):
            DSF = self.get_exts(FS)
            freeout = np.array([(DSF/self.RLS - 1.)[te] for te in self.tasks[taskIndex].targetedges])
        return freeout
    
    def loss(self, taskIndex, freeout, desiredout):
        #squared error of the outputs
        #taskIndex: choose from your list of tasks, defaults to 0
        #freeout, desiredout: arrays of equal shape
        
        if freeout.shape != desiredout.shape:
            raise Exception('outputs are not of the same shape')
            return
        
        task = self.tasks[taskIndex]
        diffouts = desiredout - freeout
        if isinstance(task, NodeTask):
            _,tf = task.fixed
            notfixed = np.invert(tf)
            np.place(diffouts, notfixed, 0.)
        return np.sum(np.square(diffouts))
    
    def cost(self, freepos, clamppos):
        #energy cost between free and clamped network states
        #freepos, clamppos: 1D arrays of size (NN*dim)
        EC = Energy(clamppos, self.KS, self.RLS, self.EI, self.EJ, self.dim, self.Epow, self.lnorm)
        EF = Energy(freepos, self.KS, self.RLS, self.EI, self.EJ, self.dim, self.Epow, self.lnorm)
        return EC - EF
    
    def update(self, exts_f, exts_c, rule = 'RLS', clip = [0.5, 1.5], **kwargs):
        
        row = self.history.loc[self.currentstep]
        
        #update rule
        if rule == 'RLS':
            dLs = self._alpha*np.multiply(self.KS, (exts_c - exts_f))
            dLs[self.fixededges] = 0
            self.RLS += dLs
            #clip rest lengths between max and min values
            if clip is not None:
                self.RLS = np.clip(self.RLS, clip[0], clip[1])
            else:
                self.RLS[self.RLS < 0] = 0
        elif rule == 'KS':
            dks = (-1)*self._alpha*(np.subtract(exts_c, self.RLS)**2 - np.subtract(exts_f, self.RLS)**2)
            dks[self.fixededges] = 0
            self.KS += dks
            if clip is not None:
                self.KS = np.clip(self.KS, clip[0], clip[1])
            else:
                self.KS[self.KS < 0] = 0
        else:
            raise Exception('Learning rule not recongized')

        #update equilibrium state
        self.x0 = self.eq_state(self.x0)
        
    def initialize_step(self, verbose=False, **kwargs):
        self.currentstep += 1
        #start a new row
        self.history.loc[self.currentstep] = [self.currentstep] + [None]*(len(self.history.columns)-1)
        if verbose:
            print('Step {} initialized'.format(self.currentstep))
     
    def learning_step(self, taskIndex=0, pairIndex=0, plot=False, **kwargs):
        #iopair: tuple of: array of shape(# sources, dim), array of shape(#targets, dim)
        
        task = self.tasks[taskIndex]
        
        inputs, outputs = task.IOpairs[pairIndex]
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        
        self.initialize_step(**kwargs)
        eq = self.x0.reshape(self.NN,self.dim)
        if isinstance(task, NodeTask):
            refin = [eq[sn] for sn in task.sourcenodes]
            refout = [eq[tn] for tn in task.targetnodes]
        row = self.history.loc[self.currentstep]
        row.RLS = json.dumps(self.RLS.tolist())
        row.KS = json.dumps(self.KS.tolist())
        row.task = taskIndex
        
        #apply free state
        FS = self.free_state(taskIndex,inputs,plot=plot)
        exts_f = self.get_exts(FS)
        row.exts_f = json.dumps(exts_f.tolist())
        
        if isinstance(task, NodeTask):
            FSdim = FS.reshape(self.NN,self.dim)
            row.free_in = json.dumps([FSdim[sn].tolist() for sn in task.sourcenodes])
            freeout = np.array([FSdim[tn] for tn in task.targetnodes])
            row.free_out = json.dumps([FSdim[tn].tolist() for tn in task.targetnodes])
        
        if isinstance(task, EdgeTask):
            freestrains = exts_f/self.RLS - 1.
            row.free_in = json.dumps([freestrains[se].tolist() for se in task.sourceedges])
            freeout = np.array([freestrains[te] for te in task.targetedges])
            row.free_out = json.dumps([freestrains[te].tolist() for te in task.targetedges])
        
        #apply clamped state
        clampout = self._eta*outputs + (1-self._eta)*freeout
        newIOpair = (inputs, clampout)
        
        CS = self.clamped_state(taskIndex,newIOpair,plot=plot)
        if np.isnan(CS).any():
            self.plot_state(self.x0)
            self.stop_train = True
        exts_c = self.get_exts(CS)
        row.exts_c = json.dumps(exts_c.tolist())
        
        if isinstance(task, NodeTask):
            CSdim = CS.reshape(self.NN,self.dim)
            row.clamp_in = json.dumps([CSdim[sn].tolist() for sn in task.sourcenodes])
            row.clamp_out = json.dumps([CSdim[tn].tolist() for tn in task.targetnodes])
            
        if isinstance(task, EdgeTask):
            clampstrains = exts_c/self.RLS - 1.
            row.clamp_in = json.dumps([clampstrains[se].tolist() for se in task.sourceedges])
            row.clamp_out = json.dumps([clampstrains[te].tolist() for te in task.targetedges])

        row.loss = self.loss(taskIndex, freeout, outputs)
        row.cost = self.cost(FS, CS)
        self.update(exts_f, exts_c, **kwargs)
        
        self.history.loc[self.currentstep] = row #fix this later
        
        
    def train(self, Nsteps = 100, tasks = None, plot=False, **kwargs):
        # tasks: which task indices to train for (if none, does all)
        # Nsteps: how many times to cycle through all data points in all trainable tasks
        
        if tasks is None:
            tasks = np.arange(len(self.tasks))
        tasks = np.atleast_1d(tasks)
        
        totalTrainStep=0
        while (totalTrainStep < Nsteps) and self.stop_train==False:
            for taskIndex in tasks:
                for pairIndex in range(len(self.tasks[taskIndex].IOpairs)):
                    self.learning_step(taskIndex, pairIndex, plot=plot, **kwargs)
            totalTrainStep += 1
                    

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
