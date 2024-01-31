import numpy as np

class Edge():
    # class for denoting a node's constraints or degrees of freedom
    
    def __init__(self,index = 0, nodeindexpair = np.array([0,1]), strain = 1.):
        self.index = index
        self.nindex = nodeindexpair
        self.strain = strain

    def __str__(self):
        strpair = f'Edge: connecting nodes: {self.nindex}, strain: {self.strain}'
        if self.index:
            strpair += f'Edge index: {self.index}'
        return strpair
    
if __name__=='__main__':
    e = Edge()
    print(e)
