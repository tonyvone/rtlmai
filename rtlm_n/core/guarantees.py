"""Guarantee classes for every RTLM-N result.

Claim discipline is a core design rule: every materialized model, merged
state, and transported state must carry one of these labels. Nothing is
allowed to imply exactness it does not have.
"""

from enum import Enum


class GuaranteeClass(str, Enum):
    #: Centralized equivalence for a declared frozen representation and
    #: convex objective. Distributed merged-state materialization is
    #: bitwise-equivalent (up to floating point) to centralized training.
    EXACT_FROZEN = "EXACT-FROZEN"

    #: Exact solution of a local linear/quadratic approximation of the
    #: network around the reference parameters, with a measured
    #: linearization error that must be reported.
    LINEARIZED_BOUNDED = "LINEARIZED-BOUNDED"

    #: Compressed (sketched) state with a declared approximation error.
    SKETCHED_BOUNDED = "SKETCHED-BOUNDED"

    #: Experimentally useful without an equivalence proof (e.g. backbone
    #: migration through a learned representation bridge).
    EMPIRICAL_NEURAL = "EMPIRICAL-NEURAL"

    #: Incompatible bases, unsafe numerics, representation drift, or an
    #: invalid claim. Materialization must refuse.
    UNSUPPORTED = "UNSUPPORTED"


#: Ordering from strongest to weakest claim. Merging two states yields the
#: weaker of the two guarantee classes.
_STRENGTH = {
    GuaranteeClass.EXACT_FROZEN: 4,
    GuaranteeClass.LINEARIZED_BOUNDED: 3,
    GuaranteeClass.SKETCHED_BOUNDED: 2,
    GuaranteeClass.EMPIRICAL_NEURAL: 1,
    GuaranteeClass.UNSUPPORTED: 0,
}


def weaker(a: GuaranteeClass, b: GuaranteeClass) -> GuaranteeClass:
    """Return the weaker (more conservative) of two guarantee classes."""
    return a if _STRENGTH[a] <= _STRENGTH[b] else b
