import setuptools
import os

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# requirements
lib_path = os.path.dirname(os.path.realpath(__file__))
requirements_path = os.path.join(lib_path, 'requirements.txt')
with open(requirements_path) as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="minervachem",
    version="0.0.3",
    author="Michael Frederick Tynes",
    author_email="mtynes@lanl.gov",
    description="Machine learning for Cheminformatics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.github.com/lanl/minervachem/",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Software Development :: Libraries",
    ],
    packages=setuptools.find_packages(),
    python_requires='<3.12',
    install_requires=install_requires,
)
