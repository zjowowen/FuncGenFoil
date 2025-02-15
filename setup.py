from setuptools import setup, find_packages

with open('README.md', 'r', encoding="utf-8") as f:
    readme = f.read()

setup(
    name='FuncGenFoil',
    version='0.0.1',
    description='Airfoil Generation in Function Space.',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='zjowowen',
    author_email="zjowowen@gmail.com",

    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        'h5py',
        'timm',
        'numpy<=1.26.4',
        'torch>=2.2.0',
        'opencv-python',
        'tensordict',
        'torchrl',
        'di-treetensor',
        'matplotlib',
        'wandb',
        'rich',
        'easydict',
        'tqdm',
        'torchdyn',
        'torchsde',
        'scipy',
        'POT',
        'beartype',
        'diffusers',
        'av',
        'moviepy',
        'imageio[ffmpeg]',
    ],
    dependency_links=[
        'git+https://github.com/zjowowen/GenerativeRL_Preview.git#egg=GenerativeRL',
        'git+https://github.com/rtqichen/torchdiffeq.git#egg=torchdiffeq',
    ],
    extras_require={
        'HuggingFace': [
            'safetensors',
            'huggingface_hub',
        ],
        'formatter': [
            'black',
            'isort',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    license="Apache-2.0",
)
