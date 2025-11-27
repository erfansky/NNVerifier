import z3
from model_representation import example_model


def main():
    model, prop = example_model()
    print(model)
    print("Inputs:", model.total_inputs())
    print("Outputs:", model.total_outputs())
    print("Property inputs:", prop.input_bounds)
    print("Property outputs:", prop.output_constraints)



if __name__ == "__main__":
    main()