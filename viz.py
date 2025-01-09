import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from network import Shallow_NN, ReLU 

def create_network_visualization():

    # Create input range
    x = np.linspace(0.0, 2.0, 200)

    # Create subplot layout (4x3 grid)
    fig = make_subplots(
        rows=4, cols=3,
        subplot_titles=(
            'Preactivation Unit 1', 'Preactivation Unit 2', 'Preactivation Unit 3',
            'Activation Unit 1', 'Activation Unit 2', 'Activation Unit 3',
            'Contribution Unit 1', 'Contribution Unit 2', 'Contribution Unit 3',
            'Network Output', '', ''
        )
    )

    # Different color for each hidden unit
    colors = ['blue', 'red', 'green']

    for i in range(3):
        # Preactivations
        fig.add_trace(
            go.Scatter(x=x, y=[network.hidden_units[i].get_pre_activation(xi) for xi in x],
                    name=f'Preactivation {i+1}',
                    line=dict(color=colors[i])),
            row=1, col=i+1
        )
        
        # Activations
        fig.add_trace(
            go.Scatter(x=x, y=[network.hidden_units[i].get_activation(xi) for xi in x],
                    name=f'Activation {i+1}',
                    line=dict(color=colors[i])),
            row=2, col=i+1
        )
        
        # Contributions
        fig.add_trace(
            go.Scatter(x=x, y=[network.hidden_units[i].get_contribution(xi) for xi in x],
                    name=f'Contribution {i+1}',
                    line=dict(color=colors[i])),
            row=3, col=i+1
        )

    # Network output
    fig.add_trace(
        go.Scatter(x=x, y=[network.compute_output(xi) for xi in x],
                name='Network Output'),
        row=4, col=1
    )

    # Update layout
    fig.update_layout(
        height=1000,
        width=1200,
        showlegend=False
    )

    return fig

if __name__ == '__main__':
    # Initialize network
    network = Shallow_NN(ReLU)

    # Set default values and apply them to network
    default_biases = [-0.20, -0.90, 1.10]
    default_input_weights = [0.40, 0.90, -0.70]
    default_output_weights = [-1.30, 1.30, 0.66]
    default_output_bias = -0.23
    
    # Apply defaults to network
    for i in range(3):
        network.hidden_units[i].bias = default_biases[i]
        network.hidden_units[i].inputWeight = default_input_weights[i]
        network.hidden_units[i].outputWeight = default_output_weights[i]
    network.output_bias = default_output_bias

    fig = create_network_visualization()
    fig.show()
