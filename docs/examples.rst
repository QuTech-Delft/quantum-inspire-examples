Python Examples
===============

cQASM examples
--------------

Grover’s Algorithm: implementation in cQASM and performance analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This example demonstrates how to use the SDK to create a more complex Grover's algorithm in cQASM, and simulate the
circuit on Quantum Inspire.

.. literalinclude:: examples/sat_example/example_main.py

Code for generating and executing the cQASM
"""""""""""""""""""""""""""""""""""""""""""

.. literalinclude:: examples/sat_example/src/run.py

Code for the optimizer
""""""""""""""""""""""

.. literalinclude:: examples/sat_example/src/optimizer.py

Code for the SAT-utilities
""""""""""""""""""""""""""

.. literalinclude:: examples/sat_example/src/sat_utilities.py

ProjectQ examples
-----------------

ProjectQ example 1
^^^^^^^^^^^^^^^^^^

A simple example that demonstrates how to use the SDK to create a circuit to create a Bell state, and simulate the
circuit on Quantum Inspire.

.. literalinclude:: examples/example_projectq_entangle.py

ProjectQ example 2
^^^^^^^^^^^^^^^^^^

An example that demonstrates how to use the SDK to create a more complex circuit to run Grover's algorithm and
simulate the circuit on Quantum Inspire.

.. literalinclude:: examples/example_projectq_grover.py

Qiskit examples
---------------

Qiskit example 1
^^^^^^^^^^^^^^^^

A simple example that demonstrates how to use the SDK to create a circuit to create a Bell state, and simulate the
circuit on Quantum Inspire.

.. literalinclude:: examples/example_qiskit_entangle.py

Qiskit example 2
^^^^^^^^^^^^^^^^

A simple example that demonstrates how to use the SDK to create a circuit to demonstrate conditional gate execution.

.. literalinclude:: examples/example_qiskit_conditional.py

Back to the :doc:`main page <index>`.
