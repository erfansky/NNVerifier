"""
z3_encoder.py

Converts a NetworkModel and PropertySpec into Z3 constraints.

Responsibilities:
- Create Z3 variables for inputs and hidden neurons
- Encode Dense layers
- Encode exact ReLU activation
- Encode output constraints (currently '<')
- Expose the Z3 Solver object for solving
"""

from z3 import Real, Solver, And, Or
from typing import Dict, List

from model_representation import NetworkModel, PropertySpec, DenseLayer, ReLUActivation, OutputConstraint


class Z3Encoder:
    def __init__(self, model: NetworkModel, prop: PropertySpec):
        self.model = model
        self.prop = prop
        self.solver = Solver()
        self.var_layers: List[List] = []  # List of neuron variables per layer
        self.input_vars: Dict[str, Real] = {}  # Mapping input names -> Z3 Real variables

    # create input variables
    def create_input_variables(self):
        if self.model.input_names is None:
            # generate default names: x0, x1, ...
            self.model.input_names = [f"x{i}" for i in range(self.model.total_inputs())]
        for name in self.model.input_names:
            self.input_vars[name] = Real(name)
        # Add input bounds
        for name, (lb, ub) in self.prop.input_bounds.items():
            if name not in self.input_vars:
                raise ValueError(f"Input name {name} not in model input variables")
            var = self.input_vars[name]
            self.solver.add(var >= lb)
            self.solver.add(var <= ub)

    # encode dense layer
    def encode_dense_layer(self, layer: DenseLayer, prev_vars: List):
        """
        Encode a dense layer: h = ReLU(W * prev + b)
        Returns a list of Z3 variables representing this layer's neurons.
        """
        layer_vars = []
        for i in range(layer.units):
            # linear combination: sum_j w_ij * prev_j + b_i
            lin_expr = layer.bias[i]
            for j, prev_var in enumerate(prev_vars):
                lin_expr += layer.weights[i][j] * prev_var
            # create Z3 variable for this neuron
            neuron_var = Real(f"h{len(self.var_layers)}_{i}")
            layer_vars.append(neuron_var)

            # encode activation
            if isinstance(layer.activation, ReLUActivation):
                # Exact ReLU encoding
                # neuron_var = max(0, lin_expr)
                self.solver.add(
                    Or(
                        And(lin_expr >= 0, neuron_var == lin_expr),
                        And(lin_expr < 0, neuron_var == 0)
                    )
                )
            else:
                raise NotImplementedError(f"Activation {layer.activation.name} not implemented yet")

        return layer_vars

    # encode the full network
    def encode_network(self):
        self.create_input_variables()
        prev_vars = list(self.input_vars.values())
        for layer in self.model.layers:
            if isinstance(layer, DenseLayer):
                layer_vars = self.encode_dense_layer(layer, prev_vars)
            else:
                raise NotImplementedError(f"Layer type {type(layer)} not implemented yet")
            self.var_layers.append(layer_vars)
            prev_vars = layer_vars

    # encode output constraints
    def encode_output_constraints(self):
        """
        Currently supports '<' operator for single output.
        """
        if len(self.var_layers) == 0:
            raise ValueError("Network not encoded yet")
        output_layer_vars = self.var_layers[-1]
        if len(output_layer_vars) != len(self.prop.output_constraints):
            raise ValueError("Number of outputs does not match property specification")
        for i, (name, constr) in enumerate(self.prop.output_constraints.items()):
            out_var = output_layer_vars[i]
            if constr.operator == "<":
                self.solver.add(out_var >= constr.threshold)
            else:
                # extendable for >, <=, >=, == later
                raise NotImplementedError(f"Output operator {constr.operator} not implemented yet")

    # run full encoding pipeline
    def encode(self):
        self.encode_network()
        self.encode_output_constraints()
        return self.solver

