# FuncGenFoil: Airfoil Generation and Editing Model in Function Space
    
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

English | [简体中文(Simplified Chinese)](https://github.com/zjowowen/FuncGenFoil/blob/main/README.zh.md)

**FuncGenFoil**, short for Function-Space Generated Airfoil, is a method for generating airfoils using generative models in function space, such as diffusion models or flow models. This library provides a framework to demonstrate the power of generative models in design and optimization.


## Outline

- [Framework Structure](#framework-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [License](#license)

## Framework Structure

<p align="center">
  <img src="assets/airfoil_generation.pdf" alt="Image Description 1" width="80%" height="auto" style="margin: 0 1%;">
</p>

## Installation

Please install from source:

```bash
git clone https://github.com/zjowowen/FuncGenFoil.git
cd FuncGenFoil
pip install -e .
```

## Quick Start

Here is an example of how to train a airfoil generative flow model in function space.

Download dataset from [here](https://drive.google.com/drive/folders/1LU6p-TeWpH5b1Vvh2GRv_TwetHkyV8jZ?usp=sharing) and save it in the current directory.

To train the model without conditional information:
```bash
python train_unconditional_airfoil_generation.py
```

To train the model with conditional information:
```bash
python train_conditional_airfoil_generation.py
```

To train the model for airfoil editing:
```bash
python train_airfoil_editing.py
```

To evaluate the model for airfoil generation with super-resolution:
```bash
python eval_airfoil_generation_super_resolution.py
```

## License

FuncGenFoil is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for more details.
