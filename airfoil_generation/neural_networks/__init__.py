import torch.nn as nn

from airfoil_generation.neural_networks.neural_operator import (
    FourierNeuralOperator,
    FourierNeuralOperatorDeterministic,
    FourierNeuralOperatorBasedValueFunction,
)
from airfoil_generation.neural_networks.point_cloud_neural_operator import (
    PointCloudNeuralOperator,
)


def register_module(module: nn.Module, name: str):
    """
    Overview:
        Register the module to the module dictionary.
    Arguments:
        - module (:obj:`nn.Module`): The module to be registered.
        - name (:obj:`str`): The name of the module.
    """
    global MODULES
    if name.lower() in MODULES:
        raise KeyError(f"Module {name} is already registered.")
    MODULES[name.lower()] = module


def get_module(type: str):
    if type.lower() in MODULES:
        return MODULES[type.lower()]
    else:
        raise ValueError(f"Unknown module type: {type}")


MODULES = {
    "FourierNeuralOperator".lower(): FourierNeuralOperator,
    "FourierNeuralOperatorDeterministic".lower(): FourierNeuralOperatorDeterministic,
    "FourierNeuralOperatorBasedValueFunction".lower(): FourierNeuralOperatorBasedValueFunction,
    "PointCloudNeuralOperator".lower(): PointCloudNeuralOperator,
}
