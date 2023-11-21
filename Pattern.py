import numpy as np
from PatchVertexTag import PatchVertexTag
from abc import abstractmethod
from typing import TypeVar, Generic, List, Tuple

VariableIndicators = List[List[Tuple[PatchVertexTag, PatchVertexTag]]]

NumSides = TypeVar('NumSides')
PatternID = TypeVar('PatternID')
class Pattern(Generic['NumSides', 'PatternID']):
    @staticmethod
    @abstractmethod
    def get_constraint_matrix():
        # Implement logic to get constraint matrix using NumPy or similar
        pass

    @staticmethod
    @abstractmethod
    def get_constraint_rhs(l):
        # Implement logic to get constraint right-hand side using NumPy or similar
        pass

    @staticmethod
    @abstractmethod
    def get_variable(param, index):
        # Implement logic to get variable from PatchParam
        pass

    @staticmethod
    @abstractmethod
    def get_default_parameter(l, param):
        # Implement logic to get default parameter using NumPy or similar
        pass

    @staticmethod
    @abstractmethod
    def generate_subtopology(param, patch):
        # Implement logic to generate subtopology based on PatchT
        pass

    @staticmethod
    @abstractmethod
    def get_variable_indicators():
        # Implement logic to return variable indicators
        pass

    @staticmethod
    @abstractmethod
    def get_param_str(param):
        # Implement logic to get parameter string
        pass
