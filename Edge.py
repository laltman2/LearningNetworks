import numpy as np

class Edge():
    # class for denoting a node's constraints or degrees of freedom
    
    def __init__(self, edgeindex = 0, nodeindexpair = np.array([0,1]), strain = 1.):
        self.index = edgeindex
        self.nindex = nodeindexpair
        self.strain = strain

    def __str__(self):
        return f'Edge: {self.index}, connecting nodes: {self.nindex}, strain: {self.strain}'

if __name__=='__main__':
    e = Edge()
    print(e)
