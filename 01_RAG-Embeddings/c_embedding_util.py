import numpy as np
from numpy import dot
from numpy.linalg import norm

class EmbeddingUtil():

    def cos_sim(self, a, b):
        '''余弦距离 -- 越大越相似'''
        return dot(a, b) / (norm(a) * norm(b))
    
    def l2(self, a, b):
        '''欧氏距离 -- 越小越相似'''
        x = np.asarray(a) - np.asarray(b)
        return norm(x)