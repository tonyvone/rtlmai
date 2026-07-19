"""RTLM-N: Distributed Neural Intelligence State.

Edge nodes convert local experience into compact neural intelligence states
inside a shared canonical adaptation basis. States merge without moving raw
data, and a model factory materializes many task-, tenant-, and power-specific
models from one merged state.

The mathematical object (for supported surfaces):

    H_i = sum_j h_j h_j^T        R_i = sum_j h_j r_j^T      (frozen features)
    G_i = sum_j J_j^T J_j        g_i = sum_j J_j^T r_j      (projected Jacobian)

    Merge:        H = sum_i H_i, R = sum_i R_i  (associative, commutative)
    Materialize:  W* = (H + lambda I)^{-1} R
                  c* = (G + lambda I)^{-1} g

Every result carries an explicit guarantee class (EXACT-FROZEN,
LINEARIZED-BOUNDED, SKETCHED-BOUNDED, EMPIRICAL-NEURAL, UNSUPPORTED).
"""

from rtlm_n.core.guarantees import GuaranteeClass
from rtlm_n.accumulators.state import NeuralIntelligenceState
from rtlm_n.bases.basis import CanonicalBasis
from rtlm_n.models.backbone import TinyTransformer, BaseModelRegistry
from rtlm_n.distributed.node import EdgeNode
from rtlm_n.distributed.merge import MergeEngine, IncompatibleStateError, DuplicateStateError
from rtlm_n.materialization.factory import ModelFactory, MaterializedModel
from rtlm_n.metering.meter import EnergyMeter, DeviceProfile

__version__ = "0.1.0"

__all__ = [
    "GuaranteeClass",
    "NeuralIntelligenceState",
    "CanonicalBasis",
    "TinyTransformer",
    "BaseModelRegistry",
    "EdgeNode",
    "MergeEngine",
    "IncompatibleStateError",
    "DuplicateStateError",
    "ModelFactory",
    "MaterializedModel",
    "EnergyMeter",
    "DeviceProfile",
]
