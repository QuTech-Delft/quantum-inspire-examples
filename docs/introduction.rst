Introduction
============

Welcome to the Quantum Inspire Examples. This introduction will shortly introduce the repository, and it will guide you through the structure, installation process and how to contribute. We look forward to working with you!

The Quantum Inspire Examples consist of a number of Jupyter notebooks and python scripts with a diverse set of
Quantum algorithms that illustrate the possibilities of the Quantum Inspire platform to run more complex algorithms.
The Quantum Inspire examples make use of:

* An API for the `Quantum Inspire <https://www.quantum-inspire.com/>`_ platform (the Quantum Inspire SDK);
* Backends for:

  - the `ProjectQ SDK <https://github.com/ProjectQ-Framework/ProjectQ>`_;
  - the `Qiskit SDK <https://qiskit.org/>`_.

For more information on Quantum Inspire see `https://www.quantum-inspire.com/ <https://www.quantum-inspire.com/>`_.
Detailed information can be found in the Quantum Inspire `knowledge base <https://www.quantum-inspire.com/kbase/advanced-guide/>`_.

Quantum Inspire is developed by `QuTech <https://www.qutech.nl/>`_
QuTech is an advanced research center based in Delft, the Netherlands, for quantum computing and quantum internet.
It is a collaboration founded by the Delft University of Technology (`TU Delft <https://www.tudelft.nl/en>`_) and
the Netherlands Organisation for Applied Scientific Research (`TNO <https://www.tno.nl/en>`_).

Installing from source
----------------------

The source for the Quantum Inspire examples can be found at Github. For the default installation execute:

.. code-block:: console

    git clone https://github.com/QuTech-Delft/quantum-inspire-examples
    cd quantum-inspire-examples
    git submodule update --init
    pip install .


This will install everything necessary to run the examples, the Quantum Inspire SDK including the Qiskit and ProjectQ
packages.

Installing for generating documentation
---------------------------------------

To install the necessary packages to perform documentation activities:

.. code-block:: console

    pip install .[rtd]

The documentation generation process is dependent on pandoc. When you want to generate the
documentation and pandoc is not yet installed on your system navigate
to `Pandoc <https://pandoc.org/installing.html>`_ and follow the instructions found there to install pandoc.
To build the 'readthedocs' documentation do:

.. code-block:: console

    cd docs
    make html

The documentation is then build in 'docs/_build/html'.

Running
-------

For example usage see the python scripts in the ``docs/examples/`` directory
and Jupyter notebooks in the ``docs/notebooks/`` directory when installed from source.

For example, to run the ProjectQ example notebook after installing from source:

.. code-block:: console

    cd docs/notebooks
    jupyter notebook example_projectq.ipynb


or when you want to choose which example notebook to run from the browser do:

.. code-block:: console

    jupyter notebook --notebook-dir="docs/notebooks"

and select a Jupyter notebook (file with extension ``ipynb``) to run from one of the directories.

Configure your token credentials for Quantum Inspire
----------------------------------------------------

To make use of Quantum Inspire requires you to register and create an account. To prevent submitting your credentials
with each example you can make use of token authentication.

1. Create a Quantum Inspire account at `https://www.quantum-inspire.com/ <https://www.quantum-inspire.com/>`_ if you do not already have one.
2. Get an API token from the Quantum Inspire website `https://www.quantum-inspire.com/account <https://www.quantum-inspire.com/account>`_.
3. With your API token run:

.. code-block:: console

    from quantuminspire.credentials import save_account
    save_account('YOUR_API_TOKEN')

After calling ``save_account``, your credentials will be stored on disk and token authentication is done automatically
in many of the examples.
