import numpy as np
import pandas as pd
import json
from matplotlib import pyplot as plt
from LearningNetworks.Node import Node

class NodeTask(object):

    def __init__(self, sourcenodes=[], targetnodes=[], IOpairs=[], postype='absolute', fixed=None, dim=2):
        #source/targetnodes are indices
        #IOpair: input/output position pair
        #tuple of array of shape(# sources, dim), array of shape(#targets, dim)
        #IOpairs: list of IOpair tuples, length (# datapoints)
        #postype: "absolute" or "relative". whether node positions are relative to native state or in absolute units
        #fixed: indicate which degrees of freedom to hold fixed
        #if None, default to all true. Otherwise,
        #tuple of array of shape(# sources, dim), array of shape(# targets, dim)

        self.sourcenodes = sourcenodes
        self.targetnodes = targetnodes
        self.dim = dim
        
        self.postype = postype

        if IOpairs:
            IOpairs = np.array(IOpairs)
            #check shape/structure of IOpairs matches sources/targets
            shapeOK = True
            for IOpair in IOpairs:
                sI, tO = IOpair
                sI = np.array(sI)
                tO = np.array(tO)
                if (sI.shape != (len(self.sourcenodes), self.dim)) or (tO.shape != (len(self.targetnodes),self.dim)):
                    shapeOK = False
                    raise Exception("Shape of IOpair {} does not match source/target nodes. Expecting shape {},{}".format(IOpair, (len(self.sourcenodes), self.dim), (len(self.targetnodes),self.dim)))
            if shapeOK:
                self.IOpairs = IOpairs
                
        if not fixed:
            sfixed = np.full((len(self.sourcenodes), self.dim), True)
            tfixed = np.full((len(self.targetnodes), self.dim), True)
            self.fixed = (sfixed, tfixed)
        else:
            fixed = np.array(fixed)
            #check shape/structure of fixed matches sources/targets
            sfixed, tfixed = fixed
            if (sfixed.shape == (len(self.sourcenodes), self.dim)) and (tfixed.shape == (len(self.targetnodes),self.dim)):
                self.fixed = fixed
            else:
                raise Exception("Shape of fixed {} does not match source/target nodes. Expecting shape {},{}".format(fixed, (len(self.sourcenodes), self.dim), (len(self.targetnodes),self.dim)))
            
      
    def sources(self, inputs=None, pairIndex=0):
        # generate source node objects for a given list of inputs
        # if no inputs are given, fall back to pairIndex which chooses from self.IOpairs
        
        sources = []
        if inputs is None:
            inputs,_ = self.IOpairs[pairIndex]
        if inputs.shape != (len(self.sourcenodes), self.dim):
            raise Exception("Invalid inputs shape. Expecting array of shape {}, got shape {}".format((len(self.sourcenodes), self.dim), inputs.shape))
            return None
        for ix in range(len(self.sourcenodes)):
            sn = self.sourcenodes[ix]
            sf, _ = self.fixed
            fx = sf[ix]
            pos = inputs[ix]
            sources.append(Node(index=sn, position=pos, fixed=fx))
        return sources
    
    
    def targets(self, outputs=None, pairIndex=0):
        # generate target node objects for a given list of outputs
        # if no outputs are given, fall back to pairIndex which chooses from self.IOpairs
        targets = []
        if outputs is None:
            _,outputs = self.IOpairs[pairIndex]
        if outputs.shape != (len(self.targetnodes), self.dim):
            raise Exception("Invalid outputs shape. Expecting array of shape {}, got shape {}".format((len(self.targetnodes), self.dim), outputs.shape))
            return None
        for ix in range(len(self.targetnodes)):
            tn = self.targetnodes[ix]
            _,tf = self.fixed
            fx = tf[ix]
            pos = outputs[ix]
            targets.append(Node(index=tn, position=pos, fixed=fx))
        return targets