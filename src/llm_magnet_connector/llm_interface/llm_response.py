from dataclasses import dataclass


@dataclass
class OptimizerParameters:
    """
    This class contains values for all optimizer parameters presented to the LLM.
    
    Attributes:
        order: The order of the curve.
        ell: The length of the curve [mm].
        rbendmin: The minimum bending radius respected in the curve [mm].
        t1: Edge regression constraint tolerance. Should be < 0 [mm].
    """
    order: int
    ell: float
    rbendmin: float
    t1: float 
    
@dataclass
class BadnessCriteria:
    """
    This class contains boolean values for all badness / goodness criteria presented to the LLM.
    True means that the criterion is satisfied according to the LLM, and hence the curve bad.
    """
    unrealizable_kinks: bool
    overlapping: bool
    unreasonable_length: bool
    ends_not_smooth: bool
    

class LLMResponse:
    """
    This class contains the key values from the LLM response, i.e., the assessed badness criteria and the newly selected optimizer parameters.
    """
    def __init__(self, optimizer_parameters: OptimizerParameters, badnessCriteria: BadnessCriteria):
        self.optimizer_parameters = optimizer_parameters
        self.badnessCriteria = badnessCriteria
    
    def __str__(self):
        return f"{self.optimizer_parameters}, {self.badnessCriteria}"
    
    