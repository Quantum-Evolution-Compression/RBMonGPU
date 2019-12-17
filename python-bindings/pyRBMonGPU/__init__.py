from ._pyRBMonGPU import (
    Operator,
    Spins,
    MonteCarloLoop,
    ExactSummation,
    ExpectationValue,
    HilbertSpaceDistance,
    get_S_matrix,
    get_O_k_vector,
    psi_angles,
    setDevice
)

from .Psi import Psi
from .PsiDeep import PsiDeep

from .new_neural_network import (
    new_neural_network,
    new_deep_neural_network
)

from .LearningByGradientDescent import LearningByGradientDescent
