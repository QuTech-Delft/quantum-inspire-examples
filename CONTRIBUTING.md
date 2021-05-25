Contributions to Quantum Inspire Examples
-----------------------------------------

We welcome contributions to our Quantum Inspire Examples (QIE) repository.

By contributing to the repository you state you own the copyright to those contributions and agree to include your
contributions as part of this project under the Apache License version 2.0.

For additions, bug fixes or improvements always make a branch first.

## Uploading examples

After uploading a branch one can make a [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests) to the `dev` branch which will be reviewed by our main developers.
If the contribution is up to standards we will include it in the QIE repository.
We advise to make a sub-directory for each example. A sub-directory is mandatory when the example consist of more
than a single file.

### Uploading notebooks

Additional notebooks can be added to the /docs/notebooks directory.

### Uploading other examples

Additions not in the form of an iPython notebook can added to the /docs/examples directory.

## Code style

* We follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide. Many editors support [autopep8](https://pypi.python.org/pypi/autopep8) that can help with coding style.
  To allow longer lines, make a `.pep8` config file like:
 ```
[pep8]
max-line-length = 120
```
* Use Google Style Python Docstrings for documentation.

### Bugs reports and feature requests

If you don't know how to solve a bug yourself or want to request a feature, you can raise an issue via github’s
[issues](https://github.com/QuTech-Delft/quantum-inspire-examples/issues). Please first search for existing and closed issues,
if your problem or idea is not yet addressed, please open a new issue.
