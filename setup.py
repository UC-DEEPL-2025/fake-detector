from setuptools import setup, find_packages

setup(
    name="dfdetect",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2",
        "torchvision>=0.17",
        "hydra-core>=1.3",
        "omegaconf>=2.3",
        "albumentations>=1.4",
        "opencv-python>=4.9",
        "numpy>=1.24",
        "Pillow>=10.0",
        "tqdm>=4.66",
        "timm>=1.0",
        "scikit-learn>=1.3",
    ],
    python_requires=">=3.8",
)
