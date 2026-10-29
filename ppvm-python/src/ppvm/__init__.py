# SPDX-FileCopyrightText: 2026 The PPVM Authors
# SPDX-License-Identifier: Apache-2.0

from ._core import StimProgram as StimProgram
from .generalized_tableau import GeneralizedTableau as GeneralizedTableau
from .generalized_tableau import MeasurementResult as MeasurementResult
from .generalized_tableau import sample_stim as sample_stim
from .block_lindblad import BlockLindbladian as BlockLindbladian
from .lindblad import MAX_QUBITS as MAX_QUBITS
from .lindblad import STEP_FEATURES as STEP_FEATURES
from .lindblad import WORD_WIDTHS as WORD_WIDTHS
from .lindblad import Lindbladian as Lindbladian
from .lindblad import max_qubits as max_qubits
from .lindblad import sigma_minus as sigma_minus
from .lindblad import sigma_plus as sigma_plus
from .paulisum import LossyPauliSum as LossyPauliSum
from .paulisum import PauliSum as PauliSum
from .squin_interpreter.device import (
    GeneralizedTableauSimulator as GeneralizedTableauSimulator,
)
from .squin_interpreter.device import (
    GeneralizedTableauSimulatorTask as GeneralizedTableauSimulatorTask,
)

try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("ppvm")
except Exception:  # pragma: no cover - not installed as a distribution
    __version__ = "unknown"
