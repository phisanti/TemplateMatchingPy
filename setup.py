"""Setup script for TemplateMatchingPy."""

from setuptools import setup, find_packages
import os

# Read the README file
current_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(current_dir, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open(os.path.join(current_dir, "requirements.txt"), "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="templatematchingpy",
    version="1.0.0",
    author="TemplateMatchingPy Contributors",
    author_email="santiago.canomuniz@unibas.ch",
    description="Python implementation of ImageJ template matching and stack alignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/phisanti/TemplateMatchingPy",
    project_urls={
        "Bug Tracker": "https://github.com/phisanti/TemplateMatchingPy/issues",
        "Documentation": "https://github.com/phisanti/TemplateMatchingPy/docs",
        "Source Code": "https://github.com/phisanti/TemplateMatchingPy",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
            "myst-parser",
        ],
        "examples": [
            "matplotlib>=3.0.0",
            "jupyter",
        ],
    },
    include_package_data=True,
    package_data={
        "templatematchingpy": ["py.typed"],
    },
    keywords=[
        "image processing",
        "template matching",
        "image alignment",
        "image registration",
        "microscopy",
        "computer vision",
        "opencv",
        "imagej",
        "fiji",
    ],
    zip_safe=False,
)
