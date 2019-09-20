from ._pyRBMonGPU import (
    Operator,
    Spins,
    MonteCarloLoop,
    ExactSummation,
    # ExpectationValue,
    # HilbertSpaceDistance,
    setDevice
)

from .Psi import Psi
from .PsiDynamical import PsiDynamical
from .PsiDeep import PsiDeep

from .new_neural_network import (
    new_neural_network,
    add_hidden_spin,
    add_hidden_layer,
    new_static_neural_network,
    new_deep_neural_network
)

from .LearningByGradientDescent import LearningByGradientDescent
