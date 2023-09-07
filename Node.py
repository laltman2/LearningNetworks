import numpy as np

class Node():
    # class for denoting a node's constraints or degrees of freedom
    
    def __init__(self, index = 0, position = np.array([0., 0.]), fixed = np.array([True, True])):
        self.index = index
        self.position = position
        self.fixed = fixed

if __name__=='__main__':
    n = Node()
    print(n.index, n.position, n.fixed)
