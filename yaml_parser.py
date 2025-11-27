"""
yaml_parser.py

Reads a .yml file describing a neural network and verification properties,
and converts it into internal Python objects defined in model_representation.py.

Responsibilities:
- Load YAML
- Validate required fields
- Map layer types to classes (DenseLayer, ConvLayer, etc.)
- Map activation names to Activation objects (ReLUActivation, etc.)
- Build NetworkModel and PropertySpec objects
"""
import os
import yaml
from typing import Dict, Any, List, Tuple

from model_representation import (
    NetworkModel, PropertySpec, OutputConstraint,
    DenseLayer, ConvLayer, activation_from_name,
    build_dense_layer_from_raw, ModelFormatError
)



# Layer Factory
LAYER_FACTORY = {
    "Dense": build_dense_layer_from_raw,
    "Conv": ConvLayer,  # placeholder for future conv layers
}



# Main Parser
class YamlParser:
    def __init__(self, yaml_file: str):
        """
        :param yaml_file: path to the .yml file
        """
        self.yaml_file = yaml_file
        self.raw_data: Dict[str, Any] = {}

    def load_yaml(self):
        """Load YAML file and store raw dictionary."""
        base_path = os.path.dirname(__file__)  # directory where yaml_parser.py is
        full_path = os.path.join(base_path, self.yaml_file)
        
        with open(full_path, "r") as f:
            self.raw_data = yaml.safe_load(f)
        if not self.raw_data:
            raise ModelFormatError("YAML file is empty or malformed")

    def parse(self) -> Tuple[NetworkModel, PropertySpec]:
        """Parse raw YAML into NetworkModel and PropertySpec objects."""
        
        if not self.raw_data:
            self.load_yaml()

        if "model" not in self.raw_data:
            raise ModelFormatError("Missing 'model' section in YAML")

        model_data = self.raw_data["model"]
        if "layers" not in model_data:
            raise ModelFormatError("Missing 'layers' in model section")

        layers: List[Any] = []
        for idx, layer_raw in enumerate(model_data["layers"]):
            if "type" not in layer_raw:
                raise ModelFormatError(f"Layer {idx} missing 'type' field")
            layer_type = layer_raw["type"]
            if layer_type not in LAYER_FACTORY:
                raise ModelFormatError(f"Unknown layer type: {layer_type}")

            if layer_type == "Dense":
                layer_obj = build_dense_layer_from_raw(layer_raw)
            elif layer_type == "Conv":
                # future conv layer, placeholder
                layer_obj = ConvLayer(
                    filters=layer_raw.get("filters", 1),
                    kernel_size=tuple(layer_raw.get("kernel_size", (3, 3))),
                    activation=activation_from_name(layer_raw.get("activation", "relu")),
                    weights=layer_raw.get("weights"),
                    bias=layer_raw.get("bias"),
                    input_shape=tuple(layer_raw.get("input_shape", (1, 1, 1)))
                )
            else:
                raise ModelFormatError(f"Layer type {layer_type} not implemented")

            layers.append(layer_obj)

        # parse input names
        input_names = list(self.raw_data.get("property", {}).get("inputs", {}).keys())
        if not input_names:
            input_names = None

        network_model = NetworkModel(layers=layers, input_names=input_names)

        # Parse property section
        if "property" not in self.raw_data:
            raise ModelFormatError("Missing 'property' section in YAML")
        prop_data = self.raw_data["property"]

        input_bounds_raw: Dict[str, List[float]] = prop_data.get("inputs", {})
        input_bounds: Dict[str, Tuple[float, float]] = {}
        for name, bounds in input_bounds_raw.items():
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise ModelFormatError(f"Invalid bounds for input {name}")
            input_bounds[name] = (float(bounds[0]), float(bounds[1]))

        output_constraints_raw: Dict[str, Any] = prop_data.get("outputs", {})
        output_constraints: Dict[str, OutputConstraint] = {}
        for out_name, constr in output_constraints_raw.items():
            if not isinstance(constr, list) or len(constr) != 2:
                raise ModelFormatError(f"Invalid output constraint for {out_name}")
            operator, threshold = constr
            output_constraints[out_name] = OutputConstraint(operator=operator, threshold=float(threshold))

        prop_spec = PropertySpec(input_bounds=input_bounds, output_constraints=output_constraints)
        prop_spec.validate_inputs()

        return network_model, prop_spec