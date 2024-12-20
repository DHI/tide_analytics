"""The simulation module"""

import numpy as np


class Simulator:
    def __init__(self, name: str, low: float = 0.0, high: float = 1.0) -> None:
        """Create a new simulator

        Parameters
        ----------
        name: str
            name of simulation
        low: float, optional
            lower bound
        high: float, optional
            upper bound

        Examples
        --------
        >>> sim = Simulator(name="Scenario1", low=2.0, high=10.0)
        """
        self._name = name
        self._low = low
        self._high = high

    def simulate(self, n_samples: int = 1) -> np.ndarray:
        """Sample random numbers

        Parameters
        ----------
        n_samples: int
            number of samples

        Returns
        -------
        np.ndarray
        """
        return np.random.uniform(low=self._low, high=self._high, size=n_samples)
