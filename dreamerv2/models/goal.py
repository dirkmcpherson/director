import numpy as np
import torch
import torch.nn as nn
import torch.distributions as td
from dreamerv2.utils.ptutils import OneHotDist
from dreamerv2.models.dense import DenseModel

class GoalEncoder(DenseModel):
    def __init__(
            self,
            output_shape,
            input_size, 
            info,
        ):
        """
        :param output_shape: tuple containing shape of expected output
        :param input_size: size of input features
        :param info: dict containing num of hidden layers, size of hidden layers, activation function, output distribution etc.
        """
        super().__init__(output_shape, input_size, info)
        self.category_size = info['category_size']
        self.class_size = info['class_size']
        self.uniform_dist = None

    def forward(self, input):
        dist_inputs = self.model(input)
        # shape = dist_inputs.shape
        # logits = torch.reshape(dist_inputs, shape = (*shape[:-1], self.category_size, self.class_size))
        logits = dist_inputs
        logits = logits.view(*logits.shape[:-1], self.category_size, self.class_size)
        # print(logits.shape)
        if self.dist == 'onehotst':
            if (self.uniform_dist == None):
                # populate uniform distribution if not already done
                self.uniform_dist = td.independent.Independent(
                    td.OneHotCategoricalStraightThrough(probs=torch.ones_like(logits) / (self.class_size * self.category_size)),
                    len(self._output_shape)
                )
            return td.independent.Independent(td.OneHotCategoricalStraightThrough(logits=logits), len(self._output_shape))
        else:
            return super().forward()