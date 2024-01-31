import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
from LearningNetworks.Edge import Edge

class EdgeTask(object):

    def __init__(self, sourceedges=[], targetedges=[], IOpairs=[]):
        #source/target edges are edge indices
        #IOpair: input/output strain pair
        #tuple of array of shape(# sources), array of shape(#targets)
        #IOpairs: list of IOpair tuples, length (# datapoints)

        self.sourceedges = sourceedges
        self.targetedges = targetedges

        if IOpairs:
            IOpairs = np.array(IOpairs)
            #check shape/structure of IOpairs matches sources/targets
            shapeOK = True
            for IOpair in IOpairs:
                sI, tO = IOpair
                sI = np.array(sI)
                tO = np.array(tO)
                if (len(sI) != len(self.sourceedges)) or (len(tO) != len(self.targetedges)):
                    shapeOK = False
                    raise Exception("Shape of IOpair {} does not match source/target nodes. Expecting shape {},{}".format(IOpair, (len(self.sourceedges)), (len(self.targetedges))))
            if shapeOK:
                self.IOpairs = IOpairs
                                   
      
    def sources(self, inputs=None, pairIndex=0, EIEJ = None):
        # generate source edge objects
                                    
        sources = []
        if inputs is None:
            inputs,_ = self.IOpairs[pairIndex]
        for ix in range(len(self.sourceedges)):
            se = np.array(self.sourceedges[ix])
            strain = inputs[ix]
            sources.append(Edge(index=se, strain=strain))
        return sources
    
    
    def targets(self, outputs=None, pairIndex=0, EIEJ = None):
        # generate target node objects for a given list of outputs
        # if no outputs are given, fall back to pairIndex which chooses from self.IOpairs
        targets = []
        if outputs is None:
            _,outputs = self.IOpairs[pairIndex]
        for ix in range(len(self.targetedges)):
            te = self.targetedges[ix]
            strain = outputs[ix]
            targets.append(Edge(index=te, strain=strain))
        return targets