import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_layers, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layers (list of ints): the sizes of the hidden layers
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
         # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(state_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        # define x 
        x = state
        # Forward through each layer in `hidden_layers`, with ReLU activation
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            
        # and the final output layer    
        x = self.output(x)
        
        return x
