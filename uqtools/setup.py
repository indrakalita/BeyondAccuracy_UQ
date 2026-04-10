# setup.py
from setuptools import setup, find_packages

setup(
    name="uqtools",  # Package name (used for pip install)
    version="0.1.0",  # Initial version
    author="Your Name",
    author_email="youremail@example.com",
    description="Visualization and health analysis utilities for uncertainty quantification in ML models",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/uq_plotting",  # Update to your repo
    packages=find_packages(),  # Automatically include all packages in the repo
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "matplotlib>=3.5",
        "scipy>=1.7",
        "pandas>=1.3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
    include_package_data=True,
    zip_safe=False,
)
