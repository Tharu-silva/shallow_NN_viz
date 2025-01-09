class Shallow_NN:
    def __init__(self, activation, num_units=3):

        # Initialise hidden units
        self.hidden_units = []
        for _ in range(num_units):
            # (Hidden Unit, Output weight)
            self.hidden_units.append(Hidden_unit(activation))

        self.output_bias = -1

    """
    Computes the scalar output of the network given a scalar input 

    Args:
        input (float): Scalar input

    Returns:
        output (float): Scalar output
    """
    def compute_output(self, input):
        # Compute the weighted contribution of each HU
        output = 0
        for unit in self.hidden_units:
            output += unit.get_contribution(input) * unit.outputWeight
    
        output += self.output_bias
        return output

class Hidden_unit:
    def __init__(self, activation):
        self.activation = activation
        self.inputWeight = -1
        self.outputWeight = -1
        self.bias = -1 

    """
    Computes the contribution of the hidden unit

    Args:
        input (float): Scalar input
    
    Returns:
        contribution (float): Contribution of the unit
    """
    def get_contribution(self, input):
        contribution = self.bias + self.inputWeight * input # Pre-activations
        return self.activation(contribution)
    

def ReLU(input):
    return max(0, input)

def main():
    network = Shallow_NN(ReLU)

    # Set biases of HUs
    network.hidden_units[0].bias = -0.20
    network.hidden_units[1].bias = -0.90
    network.hidden_units[2].bias = 1.10

    # Set input weights of HUs
    network.hidden_units[0].inputWeight = 0.40
    network.hidden_units[1].inputWeight = 0.90
    network.hidden_units[2].inputWeight = -0.70

    # Set output weights of HUs
    network.hidden_units[0].outputWeight = -1.30
    network.hidden_units[1].outputWeight = 1.30
    network.hidden_units[2].outputWeight = 0.66

    # Set output bias
    network.output_bias = -0.23


    print(network.compute_output(0)) # ~0.5
    

if __name__ == "__main__":
    main()