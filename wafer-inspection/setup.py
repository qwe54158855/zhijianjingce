from setuptools import setup, find_packages

setup(
    name="wafer-inspection",
    version="0.1.0",
    description="Physics-informed scattering module for wafer inspection",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.9",
)
