# Quantum Inspire Examples

Test CI

[![License](https://img.shields.io/github/license/qutech-delft/quantum-inspire-examples.svg?)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/quantum-inspire-examples/badge/?version=latest)](https://quantum-inspire-examples.readthedocs.io/en/latest/?badge=latest)

The Quantum Inspire Examples consists of a number of Jupyter notebooks and python scripts with a diverse set of Quantum
algorithms that illustrate the possibilities of the Quantum Inspire platform to run more complex algorithms.
The Quantum Inspire examples make use of:

* An API for the [Quantum Inspire](https://www.quantum-inspire.com/) platform (the QuantumInspire SDK);
* Backends for:
  * the [ProjectQ SDK](https://github.com/ProjectQ-Framework/ProjectQ);
  * the [Qiskit SDK](https://qiskit.org/).

For more information on Quantum Inspire see
[https://www.quantum-inspire.com/](https://www.quantum-inspire.com/). Detailed information can be found in the Quantum
Inspire [knowledge base](https://www.quantum-inspire.com/kbase/advanced-guide/).

## Installing from source

The source for the Quantum Inspire examples can be found at Github. For the default installation execute:

```
git clone https://github.com/QuTech-Delft/quantum-inspire-examples
cd quantum-inspire-examples
pip install .
```

This will install everything necessary to run the examples, the Quantum Inspire SDK including the Qiskit and ProjectQ
packages.

## Installing for generating documentation

To install the necessary packages to perform documentation activities:

```
pip install .[rtd]
```

The documentation generation process is dependent on pandoc. When you want to generate the
documentation and pandoc is not yet installed on your system navigate
to [Pandoc](https://pandoc.org/installing.html) and follow the instructions found there to install pandoc.
To build the 'readthedocs' documentation do:

```
cd docs
make html
```

The documentation is then build in 'docs/_build/html' and can be viewed [here](docs/_build/html/index.html).

## Running

For example usage see the python scripts in the [docs/examples/](docs/examples/) directory
and Jupyter notebooks in the [docs/notebooks/](docs/notebooks/) directory when installed from source.

For example, to run the ProjectQ example notebook after installing from source:

```
cd docs/notebooks
jupyter notebook example_projectq.ipynb
```

or when you want to choose which example notebook to run from the browser do:

```
jupyter notebook --notebook-dir="docs/notebooks"
```

and select a Jupyter notebook (file with extension `ipynb`) to run from one of the directories.


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/QuTech-Delft/quantum-inspire-examples/dev?filepath=docs/notebooks)

Another way to browse and run the available notebooks is by clicking the 'launch binder' button above.

## Configure your token credentials for Quantum Inspire

1. Create a Quantum Inspire account at `https://www.quantum-inspire.com/` if you do not already have one.
2. Get an API token from the Quantum Inspire website `https://www.quantum-inspire.com/account`.
3. With your API token run:
```python
from quantuminspire.credentials import save_account
save_account('YOUR_API_TOKEN')
```
After calling `save_account`, your credentials will be stored on disk and token authentication is done automatically
in many of the examples.

## Known issues

* Known issues and common questions regarding the Quantum Inspire platform
  can be found in the [FAQ](https://www.quantum-inspire.com/faq/).

## Bug reports

Please submit bug-reports [on the github issue tracker](https://github.com/QuTech-Delft/quantum-inspire-examples/issues).
