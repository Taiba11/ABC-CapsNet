from setuptools import setup, find_packages

setup(
    name="abc-capsnet",
    version="1.0.0",
    author="Taiba Majid Wani, Reeva Gulzar, Irene Amerini",
    author_email="majid@diag.uniroma1.it",
    description="ABC-CapsNet: Attention based Cascaded Capsule Network for Audio Deepfake Detection",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/<your-username>/ABC-CapsNet",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "torchvision>=0.13.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "noisereduce>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
