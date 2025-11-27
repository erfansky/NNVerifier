"""
model_representation.py

Defines the internal model representation used by the pipeline:
- Layer base class and concrete layer types (DenseLayer, ConvLayer stub)
- Activation base class and concrete activations (ReLUActivation for now)
- Model and Property dataclasses to hold parsed network + verification property
- Validation helpers to check weight/bias dimensions

This module is intentionally SMT Solver agnostic so it can be reused with different solvers.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Sequence, Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod



# Exceptions
class ModelFormatError(Exception):
    """Raised when the model representation is invalid (shape mismatch, missing fields)."""
    pass


# Activation hierarchy
class Activation(ABC):
    """
    Base class for activations.

    Note: this class purposely does NOT perform solver-specific encoding.
    Solver-specific encode methods will be implemented in the encoder module,
    which will check the activation type and perform the corresponding SMT encoding.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def is_relu(self) -> bool:
        return self.name.lower() == "relu"


class ReLUActivation(Activation):
    @property
    def name(self) -> str:
        return "relu"

    def __repr__(self) -> str:
        return f"ReLUActivation()"


# Placeholder: future activations (Sigmoid, Tanh, LeakyReLU, etc.)
class SigmoidActivation(Activation):
    @property
    def name(self) -> str:
        return "sigmoid"

    def __repr__(self) -> str:
        return "SigmoidActivation()"


# Activation factory helper
def activation_from_name(name: str) -> Activation:
    if name is None:
        raise ValueError("Activation name cannot be None")
    n = name.strip().lower()
    if n == "relu":
        return ReLUActivation()
    if n == "sigmoid":
        return SigmoidActivation()
    raise ValueError(f"Unknown activation name: {name}")



# Layer hierarchy
class Layer(ABC):
    """Abstract base layer class."""

    @abstractmethod
    def num_outputs(self) -> int:
        ...

    @abstractmethod
    def __repr__(self) -> str:
        ...


@dataclass
class DenseLayer(Layer):
    units: int
    activation: Activation
    weights: List[List[float]]  # shape: [units][prev_units]
    bias: List[float]            # shape: [units]

    def __post_init__(self):
        # Basic validation of shapes
        if len(self.weights) != self.units:
            raise ModelFormatError(
                f"DenseLayer: expected {self.units} weight rows, found {len(self.weights)}"
            )
        if len(self.bias) != self.units:
            raise ModelFormatError(
                f"DenseLayer: expected bias length {self.units}, found {len(self.bias)}"
            )
        # ensure all weight rows have the same length
        row_lengths = {len(row) for row in self.weights}
        if len(row_lengths) != 1:
            raise ModelFormatError("DenseLayer: inconsistent weight row sizes")
        # store previous layer size for reference (not validated here)
        self._prev_units = next(iter(row_lengths))

    def prev_units(self) -> int:
        return self._prev_units

    def num_outputs(self) -> int:
        return self.units

    def __repr__(self) -> str:
        return (
            f"DenseLayer(units={self.units}, activation={self.activation}, "
            f"weights_shape=({self.units},{self.prev_units()}), bias_shape=({len(self.bias)},))"
        )


@dataclass
class ConvLayer(Layer):
    """
    A small placeholder for convolutional layers.
    We include parameters that are commonly needed; full conv support is more involved
    (strides, padding, dilation, multi-dimensional kernels, input shape).
    For now this class serves as a structural placeholder to allow extendability.
    """
    filters: int
    kernel_size: Tuple[int, int]             # (kh, kw)
    activation: Activation
    weights: Optional[Any] = None            # To be defined: list of kernels per filter
    bias: Optional[List[float]] = None
    input_shape: Optional[Tuple[int, int, int]] = None  # (H, W, C)

    def num_outputs(self) -> int:
        # A precise output dimension depends on stride/padding; we return number of feature maps
        return self.filters

    def __repr__(self) -> str:
        return (
            f"ConvLayer(filters={self.filters}, kernel_size={self.kernel_size}, "
            f"activation={self.activation}, input_shape={self.input_shape})"
        )


# Model and Property classes
@dataclass
class OutputConstraint:
    """
    Represents a single output property. For now we support only the form:
       y: ["<", threshold]
    but the structure is extendable to support other operators.
    """
    operator: str   # "<", ">", "<=", ">=", "==", etc.
    threshold: float

    def __post_init__(self):
        if self.operator not in {"<", ">", "<=", ">=", "=="}:
            raise ValueError(f"Unsupported operator: {self.operator}")

    def __repr__(self):
        return f"OutputConstraint({self.operator} {self.threshold})"


@dataclass
class PropertySpec:
    """
    Holds input bounds and output constraint.

    - input_bounds: mapping from input name (e.g., "x1") to [lower, upper]
    - output_constraint: dictionary mapping output name (like "y") to OutputConstraint
    """
    input_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    output_constraints: Dict[str, OutputConstraint] = field(default_factory=dict)
    
    def __post_init__(self):
        self.validate_inputs()
    
    def validate_inputs(self):
        for name, bounds in self.input_bounds.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ModelFormatError(f"Input bounds for {name} must be a list/tuple of two numbers")
            if bounds[0] > bounds[1]:
                raise ModelFormatError(f"Input bounds for {name} are invalid: lower > upper")


@dataclass
class NetworkModel:
    """
    The top-level model object containing layers and optional metadata.
    - layers: list of Layer objects in forward order
    - input_names: optional list of input variable names (if None, encoder will assign names)
    """
    layers: List[Layer]
    input_names: Optional[List[str]] = None

    def total_inputs(self) -> int:
        """
        Determine input dimensionality from the first layer. For Dense nets,
        the first DenseLayer.weights shape gives previous layer size.
        Otherwise return len(input_names) if provided.
        """
        if self.input_names is not None:
            return len(self.input_names)

        if not self.layers:
            return 0

        first = self.layers[0]
        if isinstance(first, DenseLayer):
            return first.prev_units()
        # For conv layer or others, user should provide input_names or input_shape in future
        raise ModelFormatError(
            "Cannot infer input dimension: provide input_names in the YAML or use a Dense first layer."
        )

    def total_outputs(self) -> int:
        if not self.layers:
            return 0
        return self.layers[-1].num_outputs()

    def __repr__(self) -> str:
        layers_str = "\n  ".join(repr(l) for l in self.layers)
        return f"NetworkModel(\n  layers=[\n  {layers_str}\n  ],\n  input_names={self.input_names}\n)"



# Small utility for building a model from raw python dicts (helpful for unit tests)
def build_dense_layer_from_raw(raw: dict) -> DenseLayer:
    """
    raw is expected to have keys: 'units', 'activation', 'weights', 'bias'
    """
    units = int(raw["units"])
    activation = activation_from_name(raw.get("activation", "relu"))
    weights = raw["weights"]
    bias = raw["bias"]
    # ensure floats
    weights = [[float(v) for v in row] for row in weights]
    bias = [float(v) for v in bias]
    return DenseLayer(units=units, activation=activation, weights=weights, bias=bias)


def example_model() -> Tuple[NetworkModel, PropertySpec]:
    """
    Recreates the example given in the homework prompt (10-input -> 5 -> 3 -> 1).
    This is useful for quick local tests before building the YAML parser.
    """
    # layer 1 (Dense 5)
    layer1 = build_dense_layer_from_raw({
        "units": 5,
        "activation": "relu",
        "weights": [
            [0.0, 0.1, -0.2, 0.4, 0.5, -0.3, 0.3, 0.0, -0.5, 0.2],
            [0.3, -0.4, 0.0, 0.1, -0.2, 0.5, 0.4, 0.2, -0.3, 0.1],
            [0.5, 0.0, 0.3, -0.1, 0.2, -0.2, 0.4, 0.1, 0.0, 0.5],
            [-0.3, 0.2, 0.5, -0.4, 0.1, -0.1, 0.0, 0.3, -0.2, 0.4],
            [0.2, -0.5, 0.4, 0.1, -0.3, 0.2, 0.1, 0.5, -0.2, 0.0]
        ],
        "bias": [0.1, 0.0, -0.1, 0.2, 0.0]
    })

    # layer 2 (Dense 3)
    layer2 = build_dense_layer_from_raw({
        "units": 3,
        "activation": "relu",
        "weights": [
            [0.2, -0.1, 0.3, 0.0, 0.1],
            [-0.3, 0.4, 0.2, 0.5, -0.2],
            [0.1, -0.4, 0.0, 0.3, 0.2]
        ],
        "bias": [0.0, 0.1, -0.1]
    })

    # layer 3 (Dense 1)
    layer3 = build_dense_layer_from_raw({
        "units": 1,
        "activation": "relu",
        "weights": [[0.5, -0.4, 0.3]],
        "bias": [0.1]
    })

    model = NetworkModel(layers=[layer1, layer2, layer3],
                         input_names=[f"x{i+1}" for i in range(10)])
    prop = PropertySpec(
        input_bounds={f"x{i+1}": (0.0, 1.0) for i in range(10)},
        output_constraints={"y": OutputConstraint("<", 1.0)}
    )
    return model, prop
