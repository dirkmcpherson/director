import numpy as np
import torch.nn as nn


class DenseModel(nn.Module):
    def __init__(
            self, 
            output_shape, 
            input_size, 
            layers, 
            node_size, 
            activation=nn.ELU
        ):
        super().__init__()
        self._output_shape = output_shape
        self._input_size = input_size
        self._layers = layers
        self._node_size = node_size
        self.activation = activation
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._input_size, self._node_size)]
        model += [self.activation()]
        for i in range(self._layers-1):
            model += [nn.Linear(self._node_size, self._node_size)]
            model += [self.activation()]
        model += [nn.Linear(self._node_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)