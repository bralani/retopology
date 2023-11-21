import numpy as np

class Permutation:
    def __init__(self):
        self.num_sides = 0
        self.id = 0
        self.init(0)
    
    def init(self, num_sides_, id_ = 0):
        self.num_sides = num_sides_
        self.id = id_
    
    def is_flipped(self):
        return self.id >= self.num_sides
        
    def __eq__(self, rhs):
        return self.num_sides == rhs.num_sides and self.id == rhs.id
    
    # loop utility
    def next(is_forward = True):
        id += 1 if is_forward else -1
    
    def is_valid(self):
        return 0 <= self.id and self.id < 2 * self.num_sides
    
    # access
    def __getitem__(self, index):
        if self.id < self.num_sides:
            return (index + self.id) % self.num_sides
        else:
            return self.num_sides - 1 - (index + self.id - self.num_sides) % self.num_sides

    def __call__(self, l):
        l_permuted = np.zeros(self.num_sides, dtype=np.int32)
        for i in range(self.num_sides):
            l_permuted[i] = l[self[i]]
        return l_permuted    
