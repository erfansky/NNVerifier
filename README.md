# Constraint-Based Neural Network Verification (Using Z3 SMT Solver)

This project implements a modular, extensible framework for **constraint-based neural network verification** using the **Z3 SMT Solver**.  
The goal is to check whether a neural network satisfies a given input–output specification by converting the network into a set of SMT constraints and querying Z3.

The project supports ReLU networks, dense and convolutional layers (extendable), and linear inequality specifications.  
All modules are written in a highly extensible OOP style for future expansion.

---

## 1. Command-Line Usage

The main entry point of the project is:

```
python main.py <path_to_yaml_file> [<output_file>]
```

### Arguments

| Argument | Required | Description |
|---------|----------|-------------|
| `<path_to_yaml_file>` | Yes | Path to the neural network description in YAML format. |
| `<output_file>` | No | If provided, verification output is saved to a file; otherwise printed to console. |

### Output Format

#### UNSAT — Verified Network
If Z3 returns `unsat`, verification succeeded:

```
UnSAT: No counter example found, Neural Network verified.
running time: <running time>
```

#### SAT — Counterexample Exists
If Z3 returns `sat`:

```
SAT: Neural Network verification faild!
Counter example: <variable assignments>
running time: <running time>
```

---

## 2. Project Structure

```
project/
│
├── examples/
│   ├── correct.yml
│   ├── incorrect.yml
│   └── ...
│
├── representation.py
├── yaml_parser.py
├── z3_encoders.py
├── main.py
└── README.md
```

---

## 3. examples/*.yml — Network Definition Files

All network descriptions and constraints are written in YAML format.

Each file typically specifies:

- Input bounds (domain)
- Network architecture  
  - Dense layers
  - CNN layers
  - Activation functions (ReLU supported now, extendable later)
- Output specification  
  - Linear constraints of the form:  
    `expression < value`

---

## 4. model/representation.py — Internal Data Structures

This module defines the **core OOP classes** used to represent the parsed neural network and verification constraints.

### Key Classes

- `NeuralNetworkRepresentation`: represents the network plus input domain and verification properties.
- `DenseLayer` / `ConvLayer`: store weights, biases, activation functions.
- `ActivationType`: enum of supported activations (`relu` now, others can be added).
- `LinearConstraint`: represents linear inequalities for outputs.

---

## 5. yaml_parser.py — Parsing YAML Into Internal Objects

Responsibilities:

- Reads YAML file
- Validates structure and types
- Converts layers and activations into objects from `representation.py`
- Converts input/output bounds into objects
- Raises errors for malformed or missing fields

Output: fully populated internal representation ready for Z3 encoding.

---

## 6. z3_encoders.py — SMT Encoding for Z3

Responsibilities:

- Create Z3 Real variables for all neurons and inputs
- Encode:
  - Dense layers
  - Exact ReLU activation
  - Input bounds
  - Output constraints (`<`)
- Return a Z3 solver object ready to check satisfiability

Exact ReLU encoding:

```
out >= 0
out >= linear_expr
out = linear_expr if linear_expr >= 0 else 0
```

---

## 7. main.py — Program Orchestrator

- Reads CLI arguments
- Calls `yaml_parser` to load model
- Calls `z3_encoders` to encode and run Z3 solver
- Formats SAT/UNSAT results
- Prints or saves output
- Measures running time

---

## 8. Running Examples

```
python main.py examples/correct.yml
python main.py examples/correct.yml output.txt
```

---

## 9. Requirements

- Python 3.8+
- z3-solver Python package (`pip install z3-solver`)

---

## 10. Extensibility

- Add new activation functions or layers easily
- Extend output constraint types
- Support for CNN, pooling layers in future
- Supports branch-and-bound and interval-based verification later
