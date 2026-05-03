from .shapr import ShapR
from .shap import Shap, KernelShap
from .ground_truth_shap import GroundTruthShap, BruteForceKernelShap
from .random import Random
from .lime import Lime
from .maple import Maple
class _LazyBreakDown:
    """Defer importing `breakdown` (blist / pyBreakDown) until used."""

    def __new__(cls, clf, data, **kwargs):
        from .breakdown import BreakDown as _BreakDownImpl

        return _BreakDownImpl(clf, data, **kwargs)


BreakDown = _LazyBreakDown


class _LazyL2X:
    """Defer importing `l2x` (and TensorFlow/Keras) until the explainer is used."""

    def __new__(cls, clf, data, **kwargs):
        from .l2x import L2X as _L2XImpl

        return _L2XImpl(clf, data, **kwargs)


L2X = _LazyL2X