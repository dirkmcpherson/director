'''
@author james.staley625703@tufts.edu

Wrappers for torch modules to work with director based off danijar hafner's director tensorflow code.
'''

import torch as th
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import IPython

class OneHotDist(td.OneHotCategorical):
    def __init__(self, logits, dtype=th.float32):
        super().__init__(logits=logits)
        self._logits = logits
        self.dtype = dtype

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return super()._parameter_properties(dtype)

    def sample(self, sample_shape=()):
        raise DeprecationWarning('Use td.OneHotCategoricalStraightThrough instead.')
        if not isinstance(sample_shape, (list, tuple)):
            sample_shape = (sample_shape,)
        logits = self._logits.to(self.dtype)
        shape = tuple(logits.shape)
        
        # make a square matrix of logits
        logits = logits.reshape([np.prod(shape[:-1]), shape[-1]])

        # sample from the categorical distribution
        indices = td.Categorical(logits=logits).sample(sample_shape)
        
        # convert the indices to one-hot vectors
        sample = th.zeros(indices.shape + (shape[-1],), dtype=self.dtype)
        IPython.embed()

        # sample.scatter_(-1, indices.unsqueeze(-1), 1.0)

        if np.prod(sample_shape) != 1:
            sample = sample.transpose((1, 0, 2))
        
        sample = sample.detach().reshape(sample_shape + shape)

        # Straight through biased gradient estimator.
        # probs = self._pad(super()._probs, sample.shape)

        
        # indices = tf.random.categorical(logits, np.prod(sample_shape), seed=None)
        # sample = tf.one_hot(indices, shape[-1], dtype=self.dtype)
        # if np.prod(sample_shape) != 1:
        # sample = sample.transpose((1, 0, 2))
        # sample = tf.stop_gradient(sample.reshape(sample_shape + shape))
        # # Straight through biased gradient estimator.
        # probs = self._pad(super().probs_parameter(), sample.shape)
        # sample += tf.cast(probs - tf.stop_gradient(probs), sample.dtype)
        # return sample
        return sample

    def _pad(self, tensor, shape):
        while len(tensor.shape) < len(shape):
            tensor = tensor.unsqueeze(0)

    def __repr__(self):
        return 'OneHotDist()'



def print_tensor_memory(tensor):
    # print the tensor shape and memory usage in MB
    print(f"{tensor.shape} uses {get_tensor_memory_bytes(tensor)//1e6} MB (rounded).")

def get_tensor_memory_bytes(tensor):
    return tensor.element_size() * tensor.nelement()