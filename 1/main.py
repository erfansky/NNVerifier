import sys
import time
from yaml_parser import YamlParser
from z3_encoder import Z3Encoder


def format_sat_result(model, running_time):
    """
    Format SAT output (counterexample found).
    """
    lines = []
    lines.append("SAT: Neural Network verification failed!")
    lines.append("Counter example:")

    # Print each variable assignment (inputs only)
    for d in model.decls():
        # we print only inputs (x1, x2, ...)
        name = d.name()
        if name.startswith("x"):
            value = model[d]
            lines.append(f"  {name} = {value}")

    lines.append(f"Running time: {running_time:.6f} seconds")
    return "\n".join(lines)


def format_unsat_result(running_time):
    """
    Format UNSAT output (verified).
    """
    return (
        "UnSAT: No counter example found, Neural Network verified.\n"
        f"Running time: {running_time:.6f} seconds"
    )


def save_or_print(result_text, output_path=None):
    """
    If output_path is provided, save result to file.
    Otherwise print to console.
    """
    if output_path:
        with open(output_path, "w") as f:
            f.write(result_text)
        print(f"Result saved to {output_path}")
    else:
        print(result_text)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_yaml> [output_file]")
        return

    yaml_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        parser = YamlParser(yaml_path)
        model, prop = parser.parse()
    except Exception as e:
        print("Error while parsing YAML:", e)
        return

    encoder = Z3Encoder(model, prop)

    start_time = time.time()
    solver = encoder.encode()
    result = solver.check()
    running_time = time.time() - start_time

    # Interpret result
    if str(result) == "unsat":
        result_text = format_unsat_result(running_time)
    elif str(result) == "sat":
        model_cex = solver.model()
        result_text = format_sat_result(model_cex, running_time)
    else:
        # unknown
        result_text = (
            "UNKNOWN: Solver could not determine satisfiability.\n"
            f"Running time: {running_time:.6f} seconds"
        )
        
    save_or_print(result_text, output_path)


if __name__ == "__main__":
    main()
