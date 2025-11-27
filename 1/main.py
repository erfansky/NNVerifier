import z3
from yaml_parser import YamlParser,ModelFormatError
import os

def main():
    print(os.getcwd())
    parser = YamlParser("examples/three_hidden.yml")
    try:
        model, prop = parser.parse()
        print("Parsed model:")
        print(model)
        print("Parsed property:")
        print(prop)
    except ModelFormatError as e:
        print("Error parsing YAML:", e)



if __name__ == "__main__":
    main()