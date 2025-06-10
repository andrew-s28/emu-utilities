import importlib.metadata

from .adjoint_gradient import load_adjoint_gradient
from .attribution import load_attribution
from .convolution import control_variance, lagged_variance, load_1d_convolution, load_2d_convolution, spatial_variance
from .forward_gradient import load_forward_gradient
from .sampling import load_sample
from .tracer import load_tracer

__all__ = [
    "load_adjoint_gradient",
    "load_attribution",
    "load_1d_convolution",
    "load_2d_convolution",
    "control_variance",
    "lagged_variance",
    "spatial_variance",
    "load_forward_gradient",
    "load_sample",
    "load_tracer",
]

try:
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"  # Fallback for development mode
