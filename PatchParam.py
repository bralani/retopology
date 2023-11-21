import numpy as np
from Permutation import Permutation

class PatchParam:
    def __init__(self):
        self.pattern_id = -1
        self.l = np.array([], dtype=np.int32)
        self.permutation = Permutation()
        self.p = np.array([], dtype=np.int32)
        self.q = np.array([], dtype=np.int32)
        self.x = -1
        self.y = -1
        self.z = -1
        self.w = -1

    def get_num_sides(self):
        return len(self.l)

    def get_l_permuted(self):
        return self.permutation(self.l)
