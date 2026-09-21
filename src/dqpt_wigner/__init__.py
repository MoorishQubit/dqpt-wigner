"""Phase-space tools for dynamical quantum phase transitions.

The package separates DQPT-I, diagnosed by late-time local order, from
DQPT-II, diagnosed by nonanalytic symmetry-resolved return rates.  For odd
local dimension it further decomposes every stabilizer return branch into an
unsigned-support rate and a Wigner-sign rate.
"""

from .version import __version__

__all__ = ["__version__"]
