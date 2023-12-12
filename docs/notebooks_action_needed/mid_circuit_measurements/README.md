# Mid-circuit Measurements tools

To improve initialization and readout fidelity, we can employ the MCMs. In particular, we will implement the
post-selection (PS) and the maximum likelihood 3 readout (ML3).
The ML3 RO measurement procedure is implemented at the measurement step. Instead of reading out the state only
once (ML1), we perform three measurements back to back. Depending on the 3 measurement outcomes, we assign the
qubit's state. It allows us to gain additional information about the qubit state over an extended duration by
tracking the trajectory of the state over the three measurements.
Post-selection is implemented by doing a qubit measurement right after the initialization. If the qubit is not
in the desired state, we disregard the job.
