import logging

import numpy as np
import qcodes
import qtt

import inspect

import threading
from collections.abc import Mapping
from functools import partial
from typing import Any, Callable
from abc import ABCMeta

from qcodes import Instrument, find_or_create_instrument
from qtt.instrument_drivers.gates import VirtualDAC
from qtt.instrument_drivers.virtual_instruments import VirtualIVVI, VirtualMeter
import termcolor
from quantify_core.data.handling import to_gridded_dataset


class ReprMixin(metaclass=ABCMeta):
    """Automatically generates a _repr_pretty_-method for any class

    The representation incluses the class attributes and properties, but limits the representation sizes.

    """

    _max_representation_size = 200
    _add_properties = True

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """IPython pretty representation"""
        p.text(self._repr_mixin_() if not cycle else "...")

    def _repr_mixin_(self) -> str:
        return self._fancy_repr_(
            self, max_representation_size=self._max_representation_size, add_properties=self._add_properties
        )

    @staticmethod
    def _fancy_repr_format_item(key: Any, value: Any, max_attribute_value_size: int) -> str:
        ckey = termcolor.colored(key, "cyan")  # pick according to rprint style
        if isinstance(value, Callable):  # type: ignore
            value = getattr(value, "__name__", str(value))
        if isinstance(value, str):
            if len(value) > max_attribute_value_size:
                value = value[:10] + "..." + value[:-10]
            return f"{ckey}={value}"
        if np.isscalar(value) or value is None:
            return f"{ckey}={value!r}"
        return f"{ckey}"

    @staticmethod
    def _fancy_repr_(  # pylint: disable=too-many-arguments
        self: object,
        attributes: list[str] | None = None,
        add_properties: bool = True,
        max_representation_size: int = 200,
        max_attribute_value_size: int = 40,
        show_id: bool = True,
    ) -> str:
        _repr_valid_types = (str, int, float, tuple, type(None))

        def is_property(v: Any) -> bool:
            """Return True of the specified object is a class property"""
            return isinstance(v, property)

        object_variables = vars(self)
        if attributes is None:
            attributes = [key for key in object_variables if not key.startswith("_")]
        items = [(key, object_variables[key]) for key in attributes if isinstance(key, _repr_valid_types)]

        if add_properties:
            property_names = [name for (name, value) in inspect.getmembers(self.__class__, is_property)]
            items += [(p, value) for p in property_names if isinstance(value := getattr(self, p), _repr_valid_types)]

        formatted_attributes = [
            f"{self._fancy_repr_format_item(key, value, max_attribute_value_size)}" for key, value in items
        ]
        cummulatieve_lengths = np.cumsum(list(map(len, formatted_attributes)))

        w = np.flatnonzero(cummulatieve_lengths > max_representation_size)
        if len(w) > 0:
            v_string = ", ".join(formatted_attributes[: w[0]])
            v_string += ",..."
        else:
            v_string = ", ".join(formatted_attributes)

        class_name = termcolor.colored(self.__class__.__name__, "magenta")
        if show_id:
            class_name = f"<{class_name} at 0x{id(self):x}>"

        return f"{class_name}({v_string})"

#%%

import copy
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import qcodes
import qtt.measurements.scans
from dataclasses_json import dataclass_json
from pydantic.utils import deep_update
from importlib import import_module

def drawcrosshair(
    x: Sequence, *, color="c", marker: Optional[str] = "s", ax=None, alpha: float = 0.5, label=None, **kwargs
):
    """Draw crosshair in 1D or 2D image

    x: Position
    color: Color used for drawing
    marker: If not None, then plot a marker at the centre
    alpha: Transparency value
    kwargs: Passed to the plotting methods

    """
    if ax is None:
        ax = plt.gca()

    if len(x) == 1:
        ax.axvline(x[0], color=color, alpha=alpha, label=label, **kwargs)
    elif len(x) == 2:
        ax.axvline(x[0], color=color, alpha=alpha, label=label, **kwargs)
        ax.axhline(x[1], color=color, alpha=alpha, label=None, **kwargs)
        if marker:
            ax.plot(x[0], x[1], color=color, marker=marker, label=None, alpha=alpha, **kwargs)
    else:
        raise Exception("drawcrosshair not valid for input {x}")

def get_attribute_from_module(method_string: str) -> Any:
    """Get method from a module

    Args:
        method_string: Description of the module and attribute
    Returns:
        Attribute of the module

    Example:
        >>> get_attribute_from_module('os.mkdir')

    """
    split_idx = method_string.rfind(".")
    module_name = method_string[:split_idx]
    mod = import_module(module_name)
    f = getattr(mod, method_string[(split_idx + 1) :])
    return f

def resolve_parameter(parameter: str | qcodes.Parameter | Callable) -> qcodes.Parameter | Callable:  # type: ignore
    if isinstance(parameter, qcodes.Parameter):
        return parameter
    elif callable(parameter):
        return parameter
    elif isinstance(parameter, str):
        if "." in parameter:
            instrument_name, parameter_name = parameter.split(".")
            try:
                # try to get from instrument
                instrument: qcodes.Instrument = qcodes.Instrument.find_instrument(instrument_name)
                parameter = instrument.parameters[parameter_name]
                return parameter  # type: ignore
            except:
                # try to get callable from module
                return get_attribute_from_module(parameter)  # type: ignore

        else:
            try:
                gates: qcodes.Instrument = qcodes.Instrument.find_instrument("gates")
                parameter = getattr(gates, parameter)
                return parameter  # type: ignore
            except Exception as ex:
                raise RuntimeError(f"could not resolve parameter {parameter}, {ex}")
    else:
        raise RuntimeError("could not resolve parameter {parameter}")


def _vectorize(value, dimension: int):
    if isinstance(value, (int, float, np.number)):
        return [value] * dimension
    else:
        assert len(value) == dimension
        return value


@dataclass_json
@dataclass
class ScanRange:
    """Definition of multi-dimension scan"""

    gates: list[str]
    scanrange: list[float]
    number_of_points: list[int]
    centre: list[float] = None  # type: ignore

    def _parse_npts(self):
        if isinstance(self.number_of_points, int):
            self.number_of_points = [self.number_of_points] * self.dimension
        elif isinstance(self.number_of_points, float):
            raise ValueError("number_of_points should be int or list of int")

    def __post_init__(self):
        if isinstance(self.gates, (str, qcodes.Parameter)):
            self.gates = [self.gates]
        self._parse_npts()
        self.scanrange = _vectorize(self.scanrange, self.dimension)
        if self.centre is None:
            self.centre = 0
        self.centre = _vectorize(self.centre, self.dimension)

        self._assert_is_list_of(self.scanrange, (int, float, np.number))
        self._assert_is_list_of(self.number_of_points, int)

    def __add__(self, rhs):
        return ScanRange(
            self.gates + rhs.gates,
            self.scanrange + rhs.scanrange,
            self.number_of_points + rhs.number_of_points,
            self.centre + rhs.centre,
        )

    @property
    def parameters(self) -> list[qcodes.Parameter | Callable]:
        return [resolve_parameter(gate) for gate in self.gates]

    @classmethod
    def around_current(cls, gates, scanrange, number_of_points, **kwargs):
        if isinstance(gates, str):
            gates = [gates]
        parameters = [resolve_parameter(gate) for gate in gates]
        centre = [p() for p in parameters]
        return ScanRange(gates, scanrange, number_of_points, centre=centre, **kwargs)

    @classmethod
    def start_stop(cls, gates, start, end, number_of_points, **kwargs):
        """Create ScanRange from starting and ending points"""
        if isinstance(gates, (str, qcodes.Parameter)):
            gates = [gates]

        starts = _vectorize(start, len(gates))
        stops = _vectorize(end, len(gates))
        scanrange = [stop - start for start, stop in zip(starts, stops)]
        centre = [(stop + start) / 2 for start, stop in zip(starts, stops)]
        return ScanRange(gates, scanrange, number_of_points, centre=centre, **kwargs)

    @classmethod
    def _assert_is_list_of(clf, l, types):
        assert isinstance(l, (list, tuple))
        for v in l:
            assert isinstance(v, types)

    def project(self, index: int) -> "ScanRange":
        """Reduce ScanRange to a 1-dimensional range by projection"""
        centre = self.centre[index : index + 1]  # type: ignore
        return ScanRange(
            gates=self.gates[index : index + 1],
            scanrange=self.scanrange[index : index + 1],
            number_of_points=self.number_of_points[index : index + 1],
            centre=centre,
        )

    @property
    def parameter_names(self) -> list[str]:
        return self.gates

    @property
    def dimension(self) -> int:
        return len(self.gates)

    @property
    def scan_centre(self) -> list[float]:
        return self.centre  # type: ignore

    @property
    def start(self) -> list[float]:
        """Return the starting points of the sweeps"""
        return [self.scan_centre[idx] - self.scanrange[idx] / 2 for idx in range(self.dimension)]

    @property
    def end(self) -> list[float]:
        """Return the end points of the sweeps"""
        return [self.scan_centre[idx] + self.scanrange[idx] / 2 for idx in range(self.dimension)]

    def setpoints(self, index: int | None = None):
        if isinstance(index, int):
            r = self.scanrange[index]
            centre = 0.0
            if self.centre is not None:
                centre = self.centre[index]
            return centre + np.linspace(
                -r / 2,
                r / 2,
                self.number_of_points[index],
            )
        else:
            return [self.setpoints(ii) for ii in range(self.dimension)]

    def apply(self, measurement_control):
        settables = self.parameters
        measurement_control.settables(settables)
        grid = self.setpoints()
        measurement_control.setpoints_grid(grid)

    def scan_name(self) -> str:
        """Return short string to use as filename in storage"""
        return "scan-" + "-".join([str(g) for g in self.gates])

    def draw_crosshair(self, ax: Any | None = None, **kwargs):
        dimension = self.dimension
        if ax is None:
            ax = plt.gca()
        cargs: dict[str, Any] = {}
        if "color" not in kwargs:
            cargs["color"] = "c"
        if "alpha" not in kwargs:
            cargs["alpha"] = 0.5

        if dimension == 1 or dimension == 2:
            drawcrosshair(self.centre, ax=ax, **kwargs)
        else:
            raise NotImplementedError(f"draw crosshair dimension {dimension}")

    def _qtt_sweep(self, index, use_range: bool = False):
        step = self.scanrange[index] / self.number_of_points[index]
        if use_range:
            return {"param": self.gates[index], "range": self.scanrange[index], "step": step}
        else:
            return {"param": self.gates[index], "start": self.start[index], "end": self.end[index], "step": step}

    def generate_scanjob(self, scanjob: dict | None = None) -> dict:
        """Generate legacy qtt scanjob"""
        if scanjob is None:
            scanjob = {}
        else:
            scanjob = copy.deepcopy(scanjob)
        scanjob = deep_update(scanjob, {"sweepdata": self._qtt_sweep(0)})
        if self.dimension > 1:
            scanjob = deep_update(scanjob, {"stepdata": self._qtt_sweep(1)})
        return qtt.measurements.scans.scanjob_t(scanjob)

    def generate_scanjob_range(self, scanjob: dict | None = None) -> dict:
        """Generate legacy qtt scanjob"""
        if scanjob is None:
            scanjob = {}
        else:
            scanjob = copy.deepcopy(scanjob)
        scanjob = deep_update(scanjob, {"sweepdata": self._qtt_sweep(0, use_range=True)})
        if self.dimension > 1:
            scanjob = deep_update(scanjob, {"stepdata": self._qtt_sweep(1, use_range=True)})
        return qtt.measurements.scans.scanjob_t(scanjob)    
#%% Dot system
import copy
import itertools
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
from numpy import linalg as la
from qtt.pgeometry import projectiveTransformation
from qtt.utilities.visualization import get_axis



def isdiagonal(HH: np.ndarray) -> bool:
    """Return True if matrix is diagonal."""
    return not (np.any(HH - np.diag(np.diagonal(HH))))


# %%


class GateTransform(ReprMixin):
    def __init__(self, Vmatrix: np.ndarray, sourcenames: list[str], targetnames: list[str]):
        """Class to describe a linear transformation between source and target gates."""
        self.Vmatrix = np.asarray(Vmatrix).astype(float)
        self.sourcenames = sourcenames
        self.targetnames = targetnames

    def transformGateScan(self, vals2D: dict | np.ndarray | Sequence, nn: int = None) -> dict[str, np.ndarray]:
        """Get a list of parameter names and [c1 c2 c3 c4] 'corner' values
        to generate dictionary self.vals2D[name] = matrix of values.

        Args:
            vals2D (dict): keys are the gate names, values are matrices with the gate values.
            nn : Number of zero to pad with if not available
        Returns:
            dict: tranformed gate values.
        """

        if isinstance(vals2D, dict):
            zz = np.zeros(nn if nn is not None else (), dtype=float)
            xx = [vals2D.get(s, zz) for s in self.sourcenames]
            xx = [x.flatten() for x in xx]
            gate_values = np.vstack(xx).astype(float)

        else:
            gate_values = np.array(vals2D).astype(float)

        gate_values_out = projectiveTransformation(self.Vmatrix, gate_values)

        vals2Dout = {}
        for j, n in enumerate(self.targetnames):
            vals2Dout[n] = gate_values_out[j].reshape(nn).astype(float)
        return vals2Dout


# %%

"""  Class holding the data from a quantum dot model """

_digit_regex = re.compile(r"\d")


@dataclass
class DotSystemData:
    number_of_dots: int
    maxelectrons: int
    chemical_potential: np.ndarray = field(default=None)  # type: ignore
    on_site_charging: np.ndarray = field(default=None)  # type: ignore
    inter_site_charging_pairs: list[tuple[int, int]] = field(default_factory=list)  # type: ignore
    inter_site_charging: np.ndarray = field(default=None)  # type: ignore

    tunneling_pairs: list[tuple[int, int]] = field(default_factory=list)  # type: ignore
    tunneling: np.ndarray = field(default=None)  # type: ignore

    def __post_init__(self):
        if self.number_of_dots < 1:
            raise ValueError("number_of_dots should be positive")

        if self.inter_site_charging_pairs is None:
            self.inter_site_charging_pairs = np.array([])
        if self.tunneling_pairs is None:
            self.tunneling_pairs = np.array([])
        self.inter_site_charging_pairs = [tuple(t) for t in self.inter_site_charging_pairs]

        for t in self.inter_site_charging_pairs:
            assert t[0] < self.number_of_dots, "inter_site_charging_pairs should be between dots"
            assert t[1] < self.number_of_dots, "inter_site_charging_pairs should be between dots"
        if self.chemical_potential is None:
            self.chemical_potential = np.zeros(self.number_of_dots)
        if self.on_site_charging is None:
            self.on_site_charging = np.zeros(self.number_of_dots)
        if self.inter_site_charging is None:
            self.inter_site_charging = np.zeros(len(self.inter_site_charging_pairs))
        if self.tunneling is None:
            self.tunneling = np.zeros(len(self.tunneling_pairs))

    @property
    def variable_names(self) -> list[str]:
        varnames = (
            [f"chemical_potential{dot}" for dot in range(self.number_of_dots)]
            + [f"on_site_charging{i}" for i in range(self.number_of_dots)]
            + [f"inter_site_charging_{i}_{j}" for (i, j) in self.inter_site_charging_pairs]
            + [f"tunneling_{i}_{j}" for (i, j) in self.tunneling_pairs]
        )
        return varnames

    @property
    def variable_name_tuples(self) -> Sequence[tuple[str, tuple[int, ...]]]:
        varnames: Sequence[tuple[str, tuple[int, ...]]] = (
            [("chemical_potential", (dot,)) for dot in range(self.number_of_dots)]
            + [("on_site_charging", (dot,)) for dot in range(self.number_of_dots)]
            + [("inter_site_charging_", (i, j)) for (i, j) in self.inter_site_charging_pairs]  # type: ignore
            + [("tunneling_", (i, j)) for (i, j) in self.tunneling_pairs]  # type: ignore
        )
        return varnames

    @property
    def variable_values(self) -> Iterable[tuple[str, tuple]]:
        tt = self.variable_name_tuples
        values = [self.get_variable(*t) for t in tt]  # type:ignore
        return zip(self.variable_names, values)  # type: ignore

    def get_variable(self, name: str, indices: int | tuple[int] | None = None) -> float:
        if indices is None:
            m = _digit_regex.search(name)
            if m is None:
                raise Exception(f"variable {name} not found")
            idx = m.start()
            base = name[:idx]
            dot_strings = name[idx:].split("_")
            dots = tuple(int(d) for d in dot_strings)
        else:
            base = name
            if isinstance(indices, int):
                dots = (indices,)
            else:
                dots = indices

        if len(dots) == 1:
            val = getattr(self, base)[(dots[0])]
        else:
            pairs = getattr(self, base + "pairs")
            values = getattr(self, base[:-1])
            pidx = pairs.index(dots)
            val = values[pidx]
        return float(val)


if __name__ == "__main__":
    self = data = DotSystemData(number_of_dots=2, maxelectrons=3, inter_site_charging_pairs=[(0, 1)])
    data.variable_names
    data
    data.get_variable(name="chemical_potential0")
    data.get_variable(name="chemical_potential", indices=0)
    data.get_variable(name="inter_site_charging_0_1")


class Namespace(SimpleNamespace):
    def get(self, name):
        return getattr(self, name)

    def set(self, name, value):
        setattr(self, name, value)


class DotSystem(ReprMixin):
    """Class to simulate a system of interacting quantum dots.

    For a full model see "Quantum simulation of a Fermi-Hubbard model using a semiconductor quantum dot array":
        https://arxiv.org/pdf/1702.07511.pdf

    Based on the arguments the system calculates the energies of the different
    dot states. Using the energies the ground state, occupancies etc. can be calculated.
    The spin-state of electrons in the dots is ignored.

    The main functionality:

        * Build a Hamiltonian from the number of dots
        * Solve for the eigenvalues and eigenstates of the Hamiltonian
        * Present the results.

    The model used is [reference xxx].

    Attributes:

        number_of_basis_states (int): number of basis states.
        H (array): Hamiltonian of the system.

        energies (array): calculated energy for each state (ordered).
        states (array): eigenstates expressed in the basis states.
        state_occupancies (array): Occupancies in the dots for each state
        state_number_of_electrons (array): for each state the total number of electrons.

    """

    def __init__(
        self,
        name: str = "system",
        number_of_dots: int = 3,
        maxelectrons: int = 2,
        inter_site_charging_pairs: list[tuple[int, int]] | None = None,
        tunneling_pairs: list[tuple[int, int]] | None = None,
        number_of_gates: int | None = None,
        **kwargs,
    ):
        """Create dot system model
        Args:
            name : name of the system.
            number_of_dots: number of dots to simulate.
            maxelectrons : maximum occupancy per dot.
        """
        self.name = name
        if number_of_gates is None:
            number_of_gates = number_of_dots
        self.number_of_gates = number_of_gates
        self.alpha = np.zeros(
            (number_of_dots, number_of_gates)
        )  # virtual gate matrix, mapping gates to chemical potentials

        self._allow_classic_calculation = True

        self.data = DotSystemData(
            number_of_dots=number_of_dots,
            maxelectrons=maxelectrons,
            inter_site_charging_pairs=inter_site_charging_pairs,  # type: ignore
            tunneling_pairs=tunneling_pairs,  # type: ignore
            **kwargs,
        )
        self.matrices = Namespace()
        self.makebasis(self.number_of_dots, self.data.maxelectrons)
        self._make_variables()
        self._make_variable_matrices()
        self.solveH()

    @property
    def number_of_dots(self) -> int:
        return self.data.number_of_dots

    def order_states_by_occupation(self):
        """Order the calculated states by occupation."""
        sortinds = np.argsort(self.state_number_of_electrons)
        self._order_data(sortinds)

    def order_states_by_energy(self):
        """Order the calculated states by energy."""
        sortinds = np.argsort(self.energies)
        self._order_data(sortinds)

    @property
    def states(self):
        return self.eigenstates

    def _order_data(self, sortinds):
        _state_data = [
            "energies",
            "eigenstates",
            "state_probabilities",
            "state_occupancies",
            "state_basis_index",
        ]
        for name in _state_data:
            setattr(self, name, getattr(self, name)[sortinds])

    def show_states(self, n: int = 10):
        """List states of the system with energies."""
        print("\nEnergies/states list for %s:" % self.name)
        print("-----------------------------------")
        for i in range(min(n, self.number_of_basis_states)):
            print(
                f"{i}  - energy: "
                + str(np.around(self.energies[i], decimals=2))
                + " ,      state: "
                + str(np.around(self.state_occupancies[i], decimals=2))
                + " ,      Ne = "
                + str(self.state_number_of_electrons[i])
            )
        print(" ")

    def makebasis(self, ndots: int, maxelectrons: int = 2):
        """Define a basis of occupancy states with a specified number of dots and max occupancy.

        The basis consists of vectors of length (ndots) where each entry in the vector indicates
        the number of electrons in a dot. The number of electrons in the total system is specified
        in `nbasis`.

        Args:
            ndots: number of dots to simulate.
            maxelectrons: maximum occupancy per dot.
        """
        assert self.number_of_dots == ndots
        assert self.data.maxelectrons == maxelectrons
        basis = list(itertools.product(range(maxelectrons + 1), repeat=ndots))
        basis = np.asarray(sorted(basis, key=lambda x: sum(x)))
        self.basis = np.ndarray.astype(basis, int)
        self.number_of_electrons = np.sum(self.basis, axis=1)
        self.number_of_basis_states = len(self.number_of_electrons)
        self.H = np.zeros((self.number_of_basis_states, self.number_of_basis_states), dtype=float)
        self.eigenstates = np.zeros((self.number_of_basis_states, self.number_of_basis_states), dtype=float)

    @staticmethod
    def chemical_potential_name(dot: int) -> str:
        return f"chemical_potential{dot}"

    @staticmethod
    def on_site_charging_name(dot) -> str:
        return "on_site_charging%d" % dot

    @staticmethod
    def inter_site_charging_name(dot1, dot2):
        """Return name for nearest - neighbour charging energy."""
        return f"inter_site_charging_{dot1}_{dot2}"

    @staticmethod
    def tunneling_name(dot1: int, dot2: int) -> str:
        return f"tunneling_{dot1}_{dot2}"

    def _make_variables(self):
        """Create value and matrix for a all variables."""
        for name in self.data.variable_names:
            self.matrices.set(
                name,
                np.full((self.number_of_basis_states, self.number_of_basis_states), 0, dtype=int),
            )

    def _make_variable_matrices(self):
        """Create matrices for the interactions.

        These matrices are used to quickly calculate the Hamiltonian.

        """
        m = np.zeros(self.number_of_dots, dtype=int)

        def mkb(i, j):
            mx = m.copy()
            mx[i] = 1
            mx[j] = -1
            return mx

        potential = range(0, (self.data.maxelectrons + 1))
        on_site_potential = [n * (n - 1) / 2 for n in range(0, self.data.maxelectrons + 1)]
        for i in range(self.number_of_basis_states):
            for j in range(self.number_of_basis_states):
                if i == j:
                    for dot in range(0, self.number_of_dots):
                        number_electrons_in_dot = self.basis[i, dot]
                        self.matrices.get(self.chemical_potential_name(dot))[i, i] = potential[number_electrons_in_dot]
                        self.matrices.get(self.on_site_charging_name(dot))[i, i] = on_site_potential[
                            number_electrons_in_dot
                        ]
                    for dot1, dot2 in self.data.inter_site_charging_pairs:
                        number_electrons_in_dot1 = self.basis[i, dot1]
                        number_electrons_in_dot2 = self.basis[i, dot2]
                        # nearest-neighbour charging energy
                        logging.info("set inter_site_charging for dot %d-%d" % (dot1, dot2))
                        self.matrices.get(self.inter_site_charging_name(dot1, dot2))[i, i] = (
                            number_electrons_in_dot1 * number_electrons_in_dot2
                        )

                else:
                    statediff = self.basis[i, :] - self.basis[j, :]

                    for dot1, dot2 in self.data.tunneling_pairs:
                        if (statediff == mkb(dot1, dot2)).all() or (statediff == mkb(dot2, dot1)).all():
                            self.matrices.get(self.tunneling_name(dot1, dot2))[i, j] = -1  # tunneling term

        self._init_sparse()
        self._makebasis_extra()

    @staticmethod
    def _sparse_matrix_name(variable):
        return f"_sparse_{variable}"

    def _init_sparse(self):
        """Create sparse structures.
        Constructing a matrix using sparse elements can be faster than construction of a full matrix,
        especially for larger systems.
        """
        self.H = np.zeros((self.number_of_basis_states, self.number_of_basis_states), dtype=float)

        for name in self.data.variable_names:
            A = self.matrices.get(name)
            ind = A.flatten().nonzero()[0]
            self.matrices.set("_sparse_indices_" + name, ind)
            self.matrices.set(self._sparse_matrix_name(name), A.flat[ind])

    def makeH(self, sparse=True) -> np.ndarray:
        """Create a new Hamiltonian."""
        self.H.fill(0)
        for name, value in self.data.variable_values:
            if not value == 0:
                if sparse:
                    a = self.matrices.get(self._sparse_matrix_name(name))
                    ind = self.matrices.get("_sparse_indices_" + name)
                    self.H.flat[ind] += a * value
                else:
                    self.H += self.matrices.get(name) * value
        self.solved = False
        return self.H

    def makeHsparse(self, verbose: int = 0) -> np.ndarray:
        """Create a new Hamiltonian from sparse data"""
        return self.makeH(sparse=True)

    def solveH(self, usediag: bool = False) -> tuple:
        """Solve the system by calculating the eigenvalues and eigenstates of the Hamiltonian.

        Args:
            usediag (bool) : If True, then assume the Hamiltonial is diagonal
        Returns:
            Energies and eigenstates
        """
        if usediag:
            self.energies = self.H.diagonal()
            self.eigenstates = np.eye(len(self.energies))
            idx = np.argsort(self.energies)
            energies = self.energies[idx]
            eigenstates = self.eigenstates[idx]
        else:
            energies, eigenstates = la.eigh(self.H)

        self.solved = True
        self._update_state_data(energies, eigenstates)
        return self.energies, self.eigenstates

    @property
    def state_number_of_electrons(self):
        return np.add.reduce(self.state_occupancies, axis=1, dtype=float)

    def _update_state_data(self, energies, eigenstates):
        """Update data derived from eigenvalues and eigenstates"""
        self.state_basis_index = np.arange(len(energies), dtype=int)
        self.energies = energies
        self.eigenstates = eigenstates
        self.state_probabilities = np.square(np.absolute(self.eigenstates))
        self.state_occupancies = np.dot(self.state_probabilities.T, self.basis)
        self.order_states_by_energy()
        self.findcurrentoccupancy(is_ordered=True)

    def is_classic_system(self) -> bool:
        return not np.any(self.data.tunneling) and self._allow_classic_calculation

    def calculate_energies(
        self,
        *,
        potentials: Sequence | np.ndarray | None = None,
        gatevalues: Sequence | np.ndarray | None = None,
    ):
        """Calculate energies of the different states in the system.

        Args:
             potentials : New values for the chemical potentials in the dots.
             gatevalues:
        """
        if self.is_classic_system():
            return self.calculate_energies_classic(potentials=potentials, gatevalues=gatevalues)

        if potentials is None:
            potentials = copy.copy(self.data.chemical_potential[:])
        if gatevalues is None:
            gatevalues = np.zeros(self.number_of_gates)  # type: ignore
        assert len(potentials) == self.number_of_dots
        self.data.chemical_potential[:] = np.asarray(potentials)

        effective_potential = self.data.chemical_potential + np.dot(self.alpha, gatevalues)
        self.data.chemical_potential[:] = effective_potential
        self.makeH()
        self.solveH()
        self.data.chemical_potential[:] = np.asarray(potentials)
        return self.energies

    def calculate_energies_classic(self, *, potentials=None, gatevalues=None):
        """Calculate the energies of all dot states, given a set of gate values.

        Returns:
            Array of energies

        """
        if potentials is None:
            potentials = copy.copy(self.data.chemical_potential[:])
        if gatevalues is None:
            gatevalues = np.zeros(self.number_of_gates)
        assert len(potentials) == self.number_of_dots
        self.data.chemical_potential[:] = np.asarray(potentials)

        if np.any(self.data.tunneling):
            raise Exception(f"tunneling coefficients should be zero for {self.__class__.__name__}")
        energies = np.zeros((self.number_of_basis_states,))
        effective_potential = self.data.chemical_potential + np.dot(self.alpha, gatevalues)
        energies += self.basis.dot(effective_potential)  # chemical potential times number of electrons
        energies += self._coulomb_energy.dot(self.data.inter_site_charging)  # coulomb repulsion
        energies += self._addition_energy_basis.dot(self.data.on_site_charging)  # addition energy
        eigenstates = np.eye(len(energies))
        self.solved = True
        self._update_state_data(energies, eigenstates)

        return self.energies

    def calculate_ground_state(
        self, *, potentials: Sequence[float] | None = None, gatevalues: Sequence[float] | None = None
    ) -> np.ndarray:
        """Calculate the ground state of the dot system, given a set of gate values.

        Args:
             potentials: values for the chemical potentials in the dots.

        Returns:
             Dot occupancies for ground state

        """
        _ = self.calculate_energies(potentials=potentials, gatevalues=gatevalues)
        return self.state_occupancies[0]

    def calculate_ground_state_energy(
        self, *, potentials: Sequence[float] | None = None, gatevalues: Sequence[float] | None = None
    ) -> float:
        """Calculate the ground state of the dot system, given a set of gate values. Returns a state array."""
        energies = self.calculate_energies(potentials=potentials, gatevalues=gatevalues)
        return energies[0]

    def findcurrentoccupancy(self, exact=True, is_ordered: bool = False):
        if self.solved:
            if not is_ordered:
                self.order_states_by_energy()
            if exact:
                # almost exact...
                idx = self.energies == self.energies[0]
                self.OCC = (self.state_occupancies[idx].mean(axis=0)).round(decimals=7)
            else:
                # first order approximation
                self.OCC = np.around(self.state_occupancies[0], decimals=7)
        else:
            self.solveH()
        return self.OCC

    def showMmatrix(self, name: str | None = None, fig=10):
        if name is None:
            name = self.data.variable_names[0]

        ax = get_axis(fig)
        ax.imshow(self.matrices.get(name), interpolation="nearest")
        ax.set_title("Hamiltonian matrix " + name)
        ax.grid("on")

    def show_variables(self):
        print("\nVariable list for %s:" % self.name)
        print("----------------------------")
        for name in self.data.variable_names:
            value = self.data.get_variable(name)
            print(f"{name} = {value}")
        print(" ")

    def _makebasis_extra(self):
        """Define a basis of occupancy states

        These addition structures are used for efficient construction of the Hamiltonian
        """
        self._addition_energy_basis = self.basis.copy()
        self._coulomb_energy = np.zeros((self.number_of_basis_states, self.data.inter_site_charging.size))
        for i in range(self.number_of_basis_states):
            self._addition_energy_basis[i] = 1 / 2 * np.multiply(self.basis[i], self.basis[i] - 1)

            for p, pair in enumerate(self.data.inter_site_charging_pairs):
                num_electrons = [self.basis[i][x] for x in pair]
                self._coulomb_energy[i, p] = np.dot(*num_electrons)

    def makeparamvalues1D(self, paramnames: list[str], startend, number_of_points: int):
        raise NotImplementedError("this functionality has not been ported from qtt")

    def makeparamvalues2D(self, paramnames: list[str], cornervals, number_of_points: list[int]):
        raise NotImplementedError("this functionality has not been ported from qtt")

    def simulate_honeycomb(self, paramvalues2D, verbose=1, usediag=False, multiprocess=True):
        """Simulating a honeycomb by looping over a 2D array of parameter values"""
        raise NotImplementedError("this functionality has not been ported from qtt")


def find_transitions(occs: np.ndarray):
    """Find transitions in occupancy image.

    Returns:
        Transitions and delocalizations
    """
    transitions = np.full([np.shape(occs)[0], np.shape(occs)[1]], 0, dtype=float)
    delocalizations = np.full([np.shape(occs)[0], np.shape(occs)[1]], 0, dtype=float)

    arraysum = np.add.reduce
    d1 = arraysum(np.absolute(occs - np.roll(occs, 1, axis=0)), axis=2)
    d2 = arraysum(np.absolute(occs - np.roll(occs, -1, axis=0)), axis=2)
    d3 = arraysum(np.absolute(occs - np.roll(occs, 1, axis=1)), axis=2)
    d4 = arraysum(np.absolute(occs - np.roll(occs, -1, axis=1)), axis=2)
    transitions = d1 + d2 + d3 + d4
    # fix borders
    transitions[0, :] = 0
    transitions[-1, :] = 0
    transitions[:, 0] = 0
    transitions[:, -1] = 0

    occs1 = occs % 1

    for mi in range(occs.shape[2]):
        m1 = np.minimum(occs1[:, :, mi], np.abs(1 - occs1[:, :, mi]))
        delocalizations[1:-1, 1:-1] += m1[1:-1, 1:-1]

    return transitions, delocalizations



class OneDot(DotSystem):
    def __init__(self, name="onedot", maxelectrons=3, **kwargs):
        """Simulation of a single quantum dot."""
        super().__init__(name=name, number_of_dots=1, maxelectrons=maxelectrons, **kwargs)


class DoubleDot(DotSystem):
    def __init__(self, name="doubledot", maxelectrons=3, **kwargs):
        """Simulation of double-dot system.
        See: DotSystem.
        """
        number_of_dots = 2
        inter_site_charging_pairs = [(i, i + 1) for i in range(number_of_dots - 1)]
        tunneling_pairs = [(i, i + 1) for i in range(number_of_dots - 1)]
        super().__init__(
            name=name,
            number_of_dots=number_of_dots,
            maxelectrons=maxelectrons,
            inter_site_charging_pairs=inter_site_charging_pairs,
            tunneling_pairs=tunneling_pairs,
            **kwargs,
        )

        self.data.chemical_potential = np.array([-120.0, -100.0])  # chemical potential at zero gate voltage
        self.data.on_site_charging = np.array([54.0, 52.8])  # addition energy
        self.data.inter_site_charging == np.array([3.0])
        self.alpha = -np.array([[1.0, 0.25], [0.25, 1.0]])


class TripleDot(DotSystem):
    def __init__(self, name="tripledot", maxelectrons=3, **kwargs):
        """Simulation of triple-dot system."""
        number_of_dots = 3
        inter_site_charging_pairs = [(0, 1), (0, 2), (1, 2)]
        tunneling_pairs = [(i, i + 1) for i in range(number_of_dots - 1)]
        super().__init__(
            name=name,
            number_of_dots=number_of_dots,
            maxelectrons=maxelectrons,
            inter_site_charging_pairs=inter_site_charging_pairs,
            tunneling_pairs=tunneling_pairs,
            **kwargs,
        )

        self.data.chemical_potential = 50 + np.array([27.0, 20.0, 25.0])  # chemical potential at zero gate voltage
        self.data.on_site_charging = np.array([54.0, 52.8, 54.0])  # addition energy
        self.data.inter_site_charging = 3 * np.array([6.0, 1.0, 5.0])
        self.alpha = -np.array([[1.0, 0.25, 0.1], [0.25, 1.0, 0.25], [0.1, 0.25, 1.0]])


class FourDot(DotSystem):
    def __init__(self, name="fourdot", maxelectrons=2, **kwargs):
        """Simulation of 4-dot system."""
        number_of_dots = 4
        inter_site_charging_pairs = [(i, i + 1) for i in range(number_of_dots - 1)]
        tunneling_pairs = [(i, i + 1) for i in range(number_of_dots - 1)]
        super().__init__(
            name=name,
            number_of_dots=number_of_dots,
            maxelectrons=maxelectrons,
            inter_site_charging_pairs=inter_site_charging_pairs,
            tunneling_pairs=tunneling_pairs,
            **kwargs,
        )


class MultiDot(DotSystem):
    def __init__(self, name="multidot", number_of_dots=6, maxelectrons=3, **kwargs):
        """Classical simulation of multi dot"""
        dotpairs = list(itertools.combinations(range(number_of_dots), 2))
        inter_site_charging_pairs = dotpairs
        super().__init__(
            name=name,
            number_of_dots=number_of_dots,
            maxelectrons=maxelectrons,
            inter_site_charging_pairs=inter_site_charging_pairs,
            **kwargs,
        )

        self.data.chemical_potential = 10 * np.sin(np.arange(number_of_dots))  # chemical potential at zero gate voltage
        self.data.on_site_charging = 50 + np.sin(2 + np.arange(number_of_dots))  # addition energy

        coulomb_repulsion = [
            np.Inf,
            18.0,
            3.0,
            0.05,
        ] + [0] * number_of_dots
        self.data.inter_site_charging = np.array([coulomb_repulsion[p[1] - p[0]] for p in dotpairs])
        self.alpha = np.eye(self.number_of_dots)
        
#%% QIVirtualDAC

import logging
from functools import partial
from typing import Any

import numpy as np

import json
from typing import List, Optional

import numpy as np
import qcodes as qc

import time

import numpy as np


def lamda_do_nothing(matrix):
    return matrix


class virtual_gate_matrix:
    def __init__(
        self, name, gates, v_gates, data, forward_conv_lamda=lamda_do_nothing, backward_conv_lamda=lamda_do_nothing
    ):
        self.name = name
        self.gates = gates
        self.v_gates = v_gates
        self._matrix = data

        self.forward_conv_lamda = forward_conv_lamda
        self.backward_conv_lamda = backward_conv_lamda
        self.last_update = time.time()

    @property
    def matrix(self):
        return self.forward_conv_lamda(self._matrix)

    @matrix.setter
    def matrix(self, matrix):
        if self._matrix.shape != matrix.shape:
            raise ValueError("input shape of matrix does not match the one in the virtual gate matrix")
        self._matrix[:, :] = self.backward_conv_lamda(matrix)

    @property
    def inv(self):
        l_inv_f = combine_lamdas(self.forward_conv_lamda, lamda_invert)
        l_inv_b = combine_lamdas(self.backward_conv_lamda, lamda_invert)
        return virtual_gate_matrix(self.name, self.gates, self.v_gates, self._matrix, l_inv_f, l_inv_b)

    def reduce(self, gates, v_gates=None):
        """
        reduce size of the virtual gate matrix

        Args:
            gates (list<str>) : name of the gates where to reduce to reduce the current matrix to.
            v_gates (list<str>) : list with the names of the virtual gates (optional)
        """
        v_gates = self.get_v_gate_names(v_gates, gates)
        v_gate_matrix = np.eye(len(gates))

        for i in range(len(gates)):
            for j in range(len(gates)):
                if gates[i] in self.gates:
                    v_gate_matrix[i, j] = self[v_gates[i], gates[j]]

        return virtual_gate_matrix("dummy", gates, v_gates, v_gate_matrix)

    def get_v_gate_names(self, v_gate_names, real_gates):
        if v_gate_names is None:
            v_gates = []
            for rg in real_gates:
                gate_index = self.gates.index(rg)
                v_gates.append(self.v_gates[gate_index])
        else:
            v_gates = v_gate_names

        return v_gates

    def __getitem__(self, index):
        if isinstance(index, tuple):
            idx_1, idx_2 = index
            idx_1 = self.__evaluate_index(idx_1, self.v_gates)
            idx_2 = self.__evaluate_index(idx_2, self.gates)

            return self.matrix[idx_1, idx_2]
        else:
            raise ValueError("wrong input format provided ['virtual_gate','gate'] expected).")

    def __setitem__(self, index, value):
        self.last_update = time.time()

        if isinstance(index, tuple):
            idx_1, idx_2 = index
            idx_1 = self.__evaluate_index(idx_1, self.v_gates)
            idx_2 = self.__evaluate_index(idx_2, self.gates)

            m = self.matrix
            m[idx_1, idx_2] = value
            self._matrix[:, :] = self.backward_conv_lamda(m)
        else:
            raise ValueError("wrong input format provided ['virtual_gate','gate'] expected).")

    def __evaluate_index(self, idx, options):
        if isinstance(idx, int) >= len(options):
            raise ValueError(f"gate out of range ({idx}),  size of virtual matrix {len(options)}x{len(options)}")

        if isinstance(idx, str):
            if idx not in options:
                raise ValueError(f"{idx} gate does not exist in virtual gate matrix")
            else:
                idx = options.index(idx)

        return idx

    def __len__(self):
        return len(self.gates)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:  # pylint: disable=unused-argument
        if cycle:
            p.text(self.__repr__())
        else:
            descr = f"Virtual gate matrix named {self.name}\nContents:\n"
            content = f"\nGates : {self.gates}\nVirtual gates : {self.v_gates}\nMatrix :\n"

            for row in self.matrix:
                content += f"{row}\n"

            s = descr + content
            p.text(s)


def lamda_invert(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.inv(matrix)


def lamda_norm(matrix_norm):
    matrix_no_norm = np.empty(matrix_norm.shape)

    for i in range(matrix_norm.shape[0]):
        matrix_no_norm[i, :] = matrix_norm[i, :] / matrix_norm[i, i]

    return matrix_no_norm


def lamda_unnorm(matrix_no_norm):
    matrix_norm = np.empty(matrix_no_norm.shape)

    for i in range(matrix_norm.shape[0]):
        matrix_norm[i, :] = matrix_no_norm[i] / np.sum(matrix_no_norm[i, :])

    return matrix_norm


def combine_lamdas(l1, l2):
    def new_lamda(matrix):
        return l1(l2(matrix))

    return new_lamda


def load_virtual_gate(name, real_gates, virtual_gates=None, matrix=None) -> virtual_gate_matrix:

    virtual_gates = name_virtual_gates(virtual_gates, real_gates)

    if matrix is None:
        matrix = np.eye(len(real_gates))

    return virtual_gate_matrix(name, real_gates, virtual_gates, matrix)


def name_virtual_gates(v_gate_names: List[str], real_gates: List[str]) -> List[str]:
    """Generate virtual gate names from physical gates"""
    if v_gate_names is None:
        v_gates = []
        for i in real_gates:
            v_gates += ["v" + i]
    else:
        v_gates = v_gate_names

    return v_gates

class virtual_gates_mgr():
    def __init__(self):
        self.virtual_gate_names = []

    def add(self, name: str, gates: List[str], virtual_gates: Optional[List[str]] = None, matrix=None):

        if name not in self.virtual_gate_names:
            self.virtual_gate_names += [name]

        setattr(self, name, load_virtual_gate(name, gates, virtual_gates, matrix))

    def remove(self, name: str):
        if name not in self.virtual_gate_names:
            raise Exception(f'virtual gate matrix {name} does not exist')

        self.virtual_gate_names.remove(name)
        delattr(self, name)

    def __len__(self):
        return len(self.virtual_gate_names)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return getattr(self, self.virtual_gate_names[idx])
        return getattr(self, idx)

    def __repr__(self):
        content = f'Found {len(self)} virtual gate matrix :\n'

        for vg in self:
            content += f'\tname :: {vg.name} \t(size = {vg.matrix.shape[0]}x{vg.matrix.shape[1]})'

        return content + '\n'


class hardware(qc.Instrument):

    def __init__(self, name: str = 'hardware', dac_gate_map=None):
        """ Collection of hardware related settings

        The `hardware` is effectively a singleton class, so only one instance created in each session.
        """
        super().__init__(name)

        if dac_gate_map is None:
            dac_gate_map = {}
        self._dac_gate_map: dict = dac_gate_map
        self.virtual_gates = virtual_gates_mgr()
        self.awg2dac_ratios: dict = {}  # legacy reasons
        self._boundaries: dict = {}

    @property
    def dac_gate_map(self):
        if isinstance(self._dac_gate_map, dict):
            return self._dac_gate_map
        else:
            # reference to gates object
            return self._dac_gate_map.gate_map

    @property
    def boundaries(self):
        return self._boundaries

    @boundaries.setter
    def boundaries(self, boundary_dict):
        for key, value in boundary_dict.items():
            self._boundaries[key] = value

    def snapshot_base(self, update=False, params_to_skip_update=None):
        snapshot = super().snapshot_base(update=update, params_to_skip_update=params_to_skip_update)
        vg_snap = {}
        for vg in self.virtual_gates:
            vg_snap[vg.name] = {'real_gate_names': vg.gates, 'virtual_gate_names': vg.v_gates,
                                'virtual_gate_matrix': json.dumps(np.asarray(vg.matrix).tolist())}

        snapshot.update({'dac_gate_map': self.dac_gate_map,
                         'virtual_gates': vg_snap})
        return snapshot




from typing import Dict, Union

import numpy as np
from qtt.utilities.json_serializer import JsonSerializeKey
from termcolor import colored

VectorDictionaryType = Dict[str, Union[float, np.ndarray]]

__all__ = ["VectorDictionaryType", "VectorDictionary"]


def add_vector_dictionary(
    lhs: VectorDictionaryType, rhs: VectorDictionaryType, check_identical_keys=True
) -> VectorDictionaryType:
    """Add two vector dictionaries"""

    if check_identical_keys:
        if not lhs.keys() == rhs.keys():
            raise IndexError(f"dictionaries do not have identical keys: {lhs.keys()} {rhs.keys()}")

    keys = set(list(lhs.keys()) + list(rhs.keys()))

    return {key: lhs.get(key, 0) + rhs.get(key, 0) for key in keys}


def multiply_vector_dictionary(vector_dict: VectorDictionaryType, scale_factor: float) -> VectorDictionaryType:
    """Scale elements in dictionary with a factor"""
    return {key: scale_factor * value for key, value in vector_dict.items()}


class VectorDictionary(dict):
    """Class representing vectors of floating points in sparse format"""

    __brace_left = colored("{", "cyan")
    __brace_right = colored("}", "cyan")

    def __add__(self, o):
        return VectorDictionary(add_vector_dictionary(self, o, check_identical_keys=False))

    def __radd__(self, o):
        return VectorDictionary(add_vector_dictionary(o, self, check_identical_keys=False))

    def __sub__(self, o):
        return VectorDictionary(add_vector_dictionary(self, -1 * VectorDictionary(o), check_identical_keys=False))

    def __truediv__(self, other):
        return (1 / other) * self

    def __neg__(self):
        return -1 * self

    def __pos__(self):
        return self

    def __mul__(self, o):
        return VectorDictionary(multiply_vector_dictionary(self, o))

    def __rmul__(self, o):
        return VectorDictionary(multiply_vector_dictionary(self, o))

    def __str__(self):
        s = super().__str__()

        if s.startswith("{"):
            s = self.__brace_left + s[1:]
        if s.endswith("}"):
            s = s[:-1] + self.__brace_right
        return s

class QiVirtualDAC(VirtualDAC):
    def __init__(self, name, instruments, gate_map, **kwargs):
        """VirtualDAC instrument

        The class is overloaded to connect to the GUI elements from core_tools
        """
        super().__init__(name, instruments, gate_map, **kwargs)

        self.v_gates = {}
        self.hardware = hardware(f"{name}_hardware", dac_gate_map=self)

    def close(self):
        self.hardware.close()
        super().close()

    def virtual_gate_index(self, gate: str) -> tuple[str, int] | None:
        if gate in self.gate_map:
            return None
        else:
            for vgate, gates in self.v_gates.items():
                if gate in gates:
                    idx = gates.index(gate)
                    return vgate, idx
            return None

    def is_physical_gate(self, gate: str) -> bool:
        return gate in self.gate_map

    def vector_dictionary(self, gate: str):
        vgate = self.virtual_gate_index(gate)
        if vgate:
            v = self.hardware.virtual_gates[vgate[0]]
            w = v.inv.matrix
            pos = vgate[1]
            return VectorDictionary(zip(v.gates, w[:, pos]))
        else:
            if self.is_physical_gate(gate):
                return VectorDictionary({gate: 1.0})
            else:
                raise Exception(f"gate {gate} not found")

    def add_hardware(self, hardware):
        """Add a core_tools hardware object"""
        self.hardware = hardware

        self._gv = {}

        for virt_gate_set in self.hardware.virtual_gates:
            self._add_virtual_gates(virt_gate_set)

    def add_virtual_gate_matrix(self, name: str, gates: list[str], virtual_gates: list[str] | None = None, matrix=None):
        self.hardware.virtual_gates.add(name, gates, virtual_gates, matrix=matrix)
        vgo = self.hardware.virtual_gates[name]
        self._add_virtual_gates(vgo)

    def remove_virtual_gate_matrix(self, name: str):
        vgo = self.hardware.virtual_gates[name]
        vgates = vgo.v_gates

        self.hardware.virtual_gates.remove(name)
        for p in vgates:
            self.parameters.pop(p)

    def _add_virtual_gates(self, virt_gate_set: virtual_gate_matrix):
        """Add virtual gates from a virtual gate matrix object"""
        self.v_gates[virt_gate_set.name] = []
        for i in range(len(virt_gate_set)):
            if virt_gate_set.gates[i] in self.hardware.dac_gate_map.keys():
                virtual_gate_name = virt_gate_set.v_gates[i]
                self.v_gates[virt_gate_set.name].append(virtual_gate_name)
                logging.info(f"{self.__class__} adding {virtual_gate_name}")
                self.add_parameter(
                    virtual_gate_name,
                    set_cmd=partial(self._set_voltage_virt, virtual_gate_name, virt_gate_set),
                    get_cmd=partial(self._get_voltage_virt, virtual_gate_name, virt_gate_set),
                    unit="mV",
                    max_val_age=0,
                )

    def _set_voltage_virt(self, gate_name: str, virt_gate_obj: Any, voltage: float):
        """Set a voltage to the virtual dac

        Args:
            gate_name : name of the virtual gate
            voltage: voltage to set
        """
        names = self.gate_names()

        names_in_vg_matrix = list(set(names).intersection(virt_gate_obj.gates))
        red_virt_gates_obj = virt_gate_obj.reduce(names_in_vg_matrix)
        current_voltages_formatted = np.zeros([len(red_virt_gates_obj)])

        for i in range(len(red_virt_gates_obj)):
            current_voltages_formatted[i] = self.get(red_virt_gates_obj.gates[i])

        voltage_key = red_virt_gates_obj.v_gates.index(gate_name)
        virtual_voltages = np.matmul(red_virt_gates_obj.matrix, current_voltages_formatted)
        virtual_voltages[voltage_key] = voltage
        new_voltages = np.matmul(np.linalg.inv(red_virt_gates_obj.matrix), virtual_voltages)

        values_dict = dict(zip(red_virt_gates_obj.gates, new_voltages))
        logging.info(f"{self.__class__}: {gate_name} = {voltage} -> {values_dict}")
        self.resetgates(values_dict, values_dict, verbose=0)

    def gate_names(self) -> list[str]:
        return list(self._gate_map.keys())

    def _get_voltage_virt(self, gate_name: str, virt_gate_obj) -> float:
        """Get a voltage from the virtual dac

        Args:
            gate_name : name of the gate
            virt_gate_obj: Virtual gate object
        """
        names = self.gate_names()

        names_in_vg_matrix = list(set(names).intersection(virt_gate_obj.gates))
        red_virt_gates_obj = virt_gate_obj.reduce(names_in_vg_matrix)
        current_voltages_formatted = np.zeros([len(red_virt_gates_obj)], dtype=float)

        for i in range(len(red_virt_gates_obj)):
            # current_voltages_formatted[i] = current_voltages[names.index(red_virt_gates_obj.gates[i])]
            current_voltages_formatted[i] = self.get(red_virt_gates_obj.gates[i])

        voltage_key = red_virt_gates_obj.v_gates.index(gate_name)
        virtual_voltages = np.matmul(red_virt_gates_obj.matrix, current_voltages_formatted)

        return virtual_voltages[voltage_key]

    @property
    def gv(self):
        return self.allvalues()

    @gv.setter
    def gv(self, my_gv):
        self.gates.resetgates(my_gv, my_gv)

    def restrict_boundaries_from_current(gates, delta: float = 200):
        """Update gate boundaries to within specified range of current values"""
        new_boundaries = {}
        for gate, value in gates.allvalues().items():
            new_boundaries[gate] = (value - delta, value + delta)
        gates.restrict_boundaries(new_boundaries)
#%%

import logging
import warnings
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import qcodes
import quantify_core
import xarray
from matplotlib.axes import Axes
from qtt.utilities.visualization import get_axis
from quantify_core.visualization import mpl_plotting as qpl
from quantify_core.visualization.SI_utilities import adjust_axeslabels_SI, set_cbarlabel, set_xlabel, set_ylabel
import qcodes_loop.data

QcodesDataSet = qcodes_loop.data.data_set.DataSet
FDataSet = Union[xarray.Dataset, QcodesDataSet, xarray.DataArray]
XDataSet = Union[xarray.Dataset, xarray.DataArray]




def default_data_variable(dataset: FDataSet) -> str:
    """Return default data variable"""
    if isinstance(dataset, QcodesDataSet):
        return dataset.default_parameter_name()
    else:
        return list(dataset.data_vars)[0]


def default_coords(dataset: FDataSet) -> List[str]:
    """Return the default coordinates as a list of strings"""
    if isinstance(dataset, QcodesDataSet):
        dv = default_data_variable(dataset)
        set_arrays = dataset.arrays[dv].set_arrays
        return [s.array_id for s in set_arrays]
    else:
        coords = getattr(dataset, str(list(dataset.dims)[0])).coords
        return list(coords.keys())


def plot_xarray_dataarray(da: xarray.DataArray, fig: Union[int, None, Axes] = None):
    """Plot xarray DataArray to specified axis or figure"""
    ax = get_axis(fig)

    if len(da.dims) == 2:
        # 2D dataset, transpose
        da.transpose().plot(ax=ax)
    else:
        da.plot(ax=ax)

        dset = da.to_dataset()
        ytag = da.name
        cc = default_coords(dset)
        x0 = cc[0]
        set_xlabel(ax, dset[x0].attrs.get("long_name", dset[x0].name), dset[x0].attrs.get("units", None))
        set_ylabel(ax, dset[ytag].attrs.get("long_name", dset[ytag].name), dset[ytag].attrs.get("units", None))


def plot_xarray_dataset(
    dset: XDataSet, fig: Union[int, None, Axes] = 1, set_title: bool = True, parameter_names: Optional[List[str]] = None
):
    """Plot xarray dataset
    Args
        dset: Dataset to plot
        fig: Specification of matplotlib handle to plot to
        parameter_names: Which parameter to plot. If None, select the first one
    """
    if isinstance(dset, xarray.DataArray):
        plot_xarray_dataarray(dset, fig)
        return

    def get_y_variable(dset):
        if parameter_names is None:
            ytag = default_data_variable(dset)
        else:
            ytag = parameter_names[0]
            dnames = [dset.data_vars[v].attrs["name"] for v in dset.data_vars]
            index = dnames.index(ytag)
            ytag = list(dset.data_vars)[index]
            if len(parameter_names) > 1:
                warnings.warn(f"only plotting parameter {ytag}")
        return ytag

    ax = get_axis(fig)

    def get_name(y):
        return y.attrs.get("long_name", y.name)

    if dset.attrs.get("grid_2d", False) and len(dset.dims) == 1:
        logging.info("plot_xarray_dataset: gridded array in flat format")
        gridded_dataset = quantify_core.data.handling.to_gridded_dataset(dset)
        gridded_dataset.attrs["grid_2d"] = False

        ytag = get_y_variable(dset)

        for yi, yvals in gridded_dataset.data_vars.items():
            if yi != ytag:
                continue

            # transpose is required to have x0 on the xaxis and x1 on the y-axis
            quadmesh = yvals.transpose().plot(ax=ax)
            # adjust the labels to be SI aware
            adjust_axeslabels_SI(ax)
            set_cbarlabel(quadmesh.colorbar, get_name(yvals), yvals.units)
            # autodect degrees and radians to use circular colormap.
            qpl.set_cyclic_colormap(quadmesh, shifted=yvals.min() < 0, unit=yvals.units)

            fig = ax.get_figure()
            vv = ",".join(list(yvals.coords))
            qpl.set_suptitle_from_dataset(fig, dset, f"{vv}-{yi}")
        return

    logging.info(f"plot_xarray_dataset: dimension {dset.dims}, length {len(dset.dims)}")

    ytag = get_y_variable(dset)

    if len(dset.dims) > 2:
        raise NotImplementedError("2D array not supported yet")
    elif len(dset.dims) == 2:
        logging.info(f"plot_xarray_dataset: 2d case, variable {ytag}")

        quadmesh = dset.data_vars[ytag].transpose().plot(ax=ax)
        adjust_axeslabels_SI(ax)
        logging.info("plot_xarray_dataset: 2d case, plot done")
    else:
        logging.info("plot_xarray_dataset: 1d case")
        cc = default_coords(dset)
        x0 = cc[0]

        # maybe use xarray plotting functions? e.g. dset.plot.scatter('x0', 'y0', axes=plt.gca())
        lbly = xarray.plot.utils.label_from_attrs(dset[ytag])  # lbl: dset[ytag].attrs['name']
        ax.plot(dset[x0], dset[ytag], marker="o", label=lbly)
        set_xlabel(dset[x0].attrs.get("long_name", dset[x0].name), dset[x0].attrs.get("units", None), axis=ax)
        set_ylabel(dset[ytag].attrs.get("long_name", dset[ytag].name), dset[ytag].attrs.get("units", None), axis=ax)

    if set_title:
        title = dset.attrs.get("name", None)
        tuid = dset.attrs.get("tuid", None)
        if tuid is None:
            tuid = dset.attrs.get("qcodes_location", None)
        if title is None:
            if tuid is not None:
                ax.set_title(f"tuid {tuid}")
        else:
            if title:
                if tuid:
                    ax.set_title(f"{title}\ntuid: {tuid}")
                else:
                    ax.set_title(f"tuid: {tuid}")
                
plot_dataset = plot_xarray_dataset                    
logger = logging.getLogger(__name__)


    
# %% Data for the model


def gate_boundaries(gate_map: Mapping[str, Any]) -> Mapping[str, tuple[float, float]]:
    """Return gate boundaries

    Args:
        gate_map: Map from gate names to instrument handle
    Returns:
        Dictionary with gate boundaries
    """
    gate_boundaries = {}
    for g in gate_map:
        if "bias" in g:
            gate_boundaries[g] = (-900, 900)
        elif "SD" in g:
            gate_boundaries[g] = (-1000, 1000)
        elif "O" in g:
            gate_boundaries[g] = (-4000, 4000)
        else:
            gate_boundaries[g] = (-1000, 800)

    return gate_boundaries


def generate_configuration(ndots: int):
    """Generate configuration for a standard linear dot array sample

    Args:
        ndots: number of dots
    Returns:
        number_dac_modules (int)
        gate_map (dict)
        gates (list)
        bottomgates (list)
    """
    bottomgates = []
    for ii in range(ndots + 1):
        if ii > 0:
            bottomgates += ["P%d" % ii]
        bottomgates += ["B%d" % ii]

    sdgates = []
    for ii in [1]:
        sdgates += ["SD%da" % ii, "SD%db" % ii, "SD%dc" % ii]
    gates = ["D0"] + bottomgates + sdgates
    gates += ["bias_1", "bias_2"]
    gates += ["O1", "O2", "O3", "O4", "O5"]

    number_dac_modules = int(np.ceil(len(gates) / 14))
    gate_map = {}
    for ii, g in enumerate(sorted(gates)):
        i = int(np.floor(ii / 16))
        d = ii % 16
        gate_map[g] = (i, d + 1)

    return number_dac_modules, gate_map, gates, bottomgates


# %%
class DotModel(Instrument, ReprMixin):
    """Simulation model for linear dot array

    The model is intended for testing the code and learning. It does _not_ simulate any meaningful physics.

    """

    def __init__(
        self,
        name: str,
        number_of_dots: int = 3,
        maxelectrons: int = 2,
        sdplunger: str | None = None,
        **kwargs,
    ):
        """The model is for a linear arrays of dots with a single sensing dot

        Args:
            name  name for the instrument
            number_of_dots: number of dots in the linear array
            sdplunger: Optional used for pinchoff of sensing dot plunger
            maxelectrons: maximum number of electrons in each dot
        """

        super().__init__(name, **kwargs)

        number_dac_modules, gate_map, _, bottomgates = generate_configuration(number_of_dots)

        self.nr_ivvi = number_dac_modules
        self.gate_map = gate_map

        self._sdplunger = sdplunger

        self._data: dict = {}
        self.lock = threading.Lock()

        self.sdnoise = 0.001  # noise for the sensing dot
        self.pinchoff_noise = 0.01
        self.gates = list(self.gate_map.keys())
        self.bottomgates = bottomgates
        self.gate_pinchoff = -200
        self.sd_pinchoff = -100
        self.ohmic_resistance = 4e-3

        gateset = [(i, a) for a in range(1, 17) for i in range(number_dac_modules)]
        for i, idx in gateset:
            g = f"ivvi{i+1}_dac{idx}"
            g_named = f"{self.name}_ivvi{i+1}_dac{idx}"
            logging.debug("add gate %s" % g)
            self.add_parameter(
                g_named,
                label="Gate {g}",
                get_cmd=partial(self._data_get, g),
                set_cmd=partial(self._data_set, g),
                unit="mV",
            )

        # make entries for keithleys
        for instr in ["keithley1", "keithley2", "keithley3", "keithley4"]:
            g = self.name + "_" + instr + "_amplitude"
            self.add_parameter(
                g, label=f"Amplitude {g}", get_cmd=partial(getattr(self, instr + "_get"), "amplitude"), unit="pA"
            )

        # initialize the actual dot system
        if number_of_dots == 1:
            self.ds: DotSystem = OneDot(maxelectrons=maxelectrons)
        elif number_of_dots == 2:
            self.ds = DoubleDot(maxelectrons=maxelectrons)
        elif number_of_dots == 3:
            self.ds = TripleDot(maxelectrons=maxelectrons)
        elif number_of_dots == 6 or number_of_dots == 4 or number_of_dots == 5:
            self.ds = MultiDot(name="dotmodel", number_of_dots=number_of_dots, maxelectrons=maxelectrons)
        else:
            raise Exception("number of dots %d not implemented yet..." % number_of_dots)
        self.ds.alpha = -np.eye(self.ds.number_of_dots)  # type: ignore
        self.sourcenames = bottomgates
        self.targetnames = ["chemical_potential%d" % (i) for i in range(self.ds.number_of_dots)]

        # Vmatrix is a projective transformation
        Vmatrix = np.zeros((len(self.targetnames) + 1, len(self.sourcenames) + 1))
        Vmatrix[-1, -1] = 1
        for ii in range(self.ds.number_of_dots):
            ns = len(self.sourcenames)
            dot_distance = np.arange(ns) - (1 + 2 * ii)
            effective_lever_arm = 1 / (1 + 0.08 * np.abs(dot_distance) ** 3)
            Vmatrix[ii, 0:ns] = effective_lever_arm
            # compensate for the barriers
            Vmatrix[0 : self.ds.number_of_dots, -1] = (Vmatrix[ii, [2 * ii, 2 * ii + 2]].sum()) * -self.gate_pinchoff
        Vmatrix = np.around(Vmatrix, decimals=3)
        self.gate_transform = GateTransform(Vmatrix, self.sourcenames, self.targetnames)

        self.sensingdot1_distance = self.ds.number_of_dots / (1 + np.arange(self.ds.number_of_dots))
        self.sensingdot2_distance = self.sensingdot1_distance[::-1]

    def get_idn(self):
        """Overrule because the default get_idn yields a warning"""
        IDN = {"vendor": "QuTech", "model": self.name, "serial": None, "firmware": None}
        return IDN

    def _data_get(self, param: str) -> float:
        return self._data.get(param, 0)

    def _data_set(self, param: str, value: float):
        self._data[param] = value
        return

    def _gate2ivvi_value(self, g: str) -> float:
        i, j = self.gate_map[g]
        return self._data.get(f"ivvi{i+1}_dac{j}", 0)

    def get_gate(self, g: str) -> float:
        """Return voltage on specified gate"""
        i, j = self.gate_map[g]
        return self._data.get(f"ivvi{i+1}_dac{j}", 0)

    def _calculate_pinchoff(self, gates, offset: float = -200.0, random: float = 0):
        """Calculate current due to pinchoff of specified gates"""
        resistances = np.empty(len(gates))
        for jj, g in enumerate(gates):
            current = 10 * (qtt.algorithms.functions.logistic(self.get_gate(g), x0=offset + jj * 0.5, alpha=1 / 40.0))
            resistances[jj] = 1.0 / (current + 1e-10)
        current = 1.0 / (resistances).sum()
        if random:
            current = current + (np.random.rand() - 0.5) * random
        return current

    def computeSD(self, usediag=True, verbose=0) -> float:
        """Compute value of the sensing dots"""
        logging.debug("start SD computation")

        # contribution of charge from bottom dots
        gs_occ = self.calculate_ground_state()

        sd1 = (1.0 / np.add.reduce(self.sensingdot1_distance)) * (gs_occ * self.sensingdot1_distance).sum()
        sd2 = (1.0 / np.add.reduce(self.sensingdot2_distance)) * (gs_occ * self.sensingdot2_distance).sum()

        sd1 += self.sdnoise * (np.random.rand() - 0.5)
        sd2 += self.sdnoise * (np.random.rand() - 0.5)

        if self._sdplunger:
            val = self._calculate_pinchoff([self._sdplunger], offset=self.sd_pinchoff, random=0)
            sd1 += val

        self.sd1 = sd1
        self.sd2 = sd2

        return sd1

    def calculate_ground_state(self) -> np.ndarray:
        gv = [self.get_gate(g) for g in self.sourcenames]
        tv = self.gate_transform.transformGateScan(gv)

        gatevalues = [float(tv[k]) for k in sorted(tv.keys())]
        gs_occ = self.ds.calculate_ground_state(gatevalues=gatevalues)
        return gs_occ

    def compute(self, random: float = 0.02) -> float:
        """Compute output of the model

        Returns:
            Current in bottom channel
        """
        val = self._calculate_pinchoff(self.bottomgates, offset=self.gate_pinchoff, random=self.pinchoff_noise)
        return val

    def keithley1_get(self, param: str) -> float:
        assert param == "amplitude"
        with self.lock:
            sd1 = self.computeSD()
            self._data["keithley1_amplitude"] = sd1
        return sd1

    def keithley2_get(self, param: str) -> float:
        self.keithley1_get(param)
        return self.sd2s

    def keithley4_get(self, param: str) -> float:
        with self.lock:
            current = self.ohmic_resistance * self.get_gate("O1")
            self._data["keithley4_amplitude"] = current
        return current

    def keithley3_get(self, param: str) -> float:
        assert param == "amplitude"
        with self.lock:
            val = self.compute()
            self._data["keithley3_amplitude"] = val
        return val
    
def create_virtual_system(nr_dots: int = 2, maxelectrons: int = 2, name="dotmodel") -> qcodes.Station:
    """Create virtual system with virtual DAC and measurement instruments"""
    logger = logging.getLogger("virtual_setup")
    station = qcodes.Station()

    model = DotModel(
        name=qtt.measurements.scans.instrumentName(name),
        number_of_dots=nr_dots,
        maxelectrons=maxelectrons,
        sdplunger="SD1b",
    )
    gate_map = model.gate_map
    logging.info("initialize: DotModel created")
    ivvis = []
    for ii in range(model.nr_ivvi):
        ivvis.append(VirtualIVVI(name=f"{model.name}_ivvi{ii+1}", model=model))
    gates = QiVirtualDAC(name=f"{model.name}_gates", gate_map=gate_map, instruments=ivvis)
    gates.set("D0", 101)
    for g in model.gates:
        gates.set(g, np.random.rand() - 0.5)

    logging.info("initialize: create virtual keithley instruments")
    keithley1 = VirtualMeter(f"{model.name}_keithley1", model=model)
    keithley3 = VirtualMeter((f"{model.name}_keithley3"), model=model)
    keithley4 = VirtualMeter((f"{model.name}_keithley4"), model=model)

    logging.info("initialize: create station")
    station = qcodes.Station(gates, keithley1, keithley3, keithley4, *ivvis, model, update_snapshot=False)
    station.model = model  # type: ignore
    station.gates = gates

    gates.hardware.awg2dac_ratios["P1"] = 0.5
    gates.hardware.awg2dac_ratios["P2"] = 0.5
    gates.hardware.awg2dac_ratios["SDP"] = 0.25

    logger.info("initialized virtual dot system (%d dots)" % nr_dots)
    return station

#%%
if __name__ == "__main__":
    from rich import print as rprint
    from quantify_core.measurement import MeasurementControl

    station = create_virtual_system(name="x")
    model = station.model
    gates = station.components[model.name + "_gates"]
    gates.P1()
    
    model.sdnoise=.0
    gates = station.gates
    keithley1, keithley3 = station.components[f'{model.name}_keithley1'], station.components[f'{model.name}_keithley3']
    
    M = np.array([[1, -0.1, -0.01], [0.1, 1, -0.1], [-0.03, -0.11, 1]])
    gates.add_virtual_gate_matrix("test2", ["P1", "B1", "P2"], matrix=M)
    gates.hardware.awg2dac_ratios = {gate: 80 / 1000 for gate in ["P1", "P2", "B1", "B2", "P3"]}
    
    for g in ["P1", "vP2"]:
        print(f"vector_dictionary: {g}: {gates.vector_dictionary(g)}")
    
    rprint(gates.gate_map)
    rprint(gates.hardware.virtual_gates)
    
    
    
    MC = find_or_create_instrument(MeasurementControl, "MC")
    
    MC.settables(gates.B1)
    MC.setpoints(np.linspace(-500, 0, 340))
    MC.gettables(keithley3.amplitude)
    
    with gates.restore_at_exit():
        dataset = MC.run("1D charge stability diagram", save_data=False)
    
    plot_dataset(dataset, fig=1)
    
