from yaml_parser import YamlParser,ModelFormatError
from z3_encoder import Z3Encoder

def main():
    parser = YamlParser("examples/incorrect.yml")
    try:
        model, prop = parser.parse()
        encoder = Z3Encoder(model, prop)
        solver = encoder.encode()
        result = solver.check()
        print("Solver result:", result)
        if result.r == 1:  # sat
            print("Counterexample:", solver.model())
    except ModelFormatError as e:
        print("Error parsing YAML:", e)



if __name__ == "__main__":
    main()