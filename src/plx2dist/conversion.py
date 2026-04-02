from __future__ import annotations

from astropy import units as u
from astropy.units import Quantity

def parallax_to_distance(parallax: Quantity) -> Quantity:
    """Convert parallax to distance.

    Parameters
    ----------
    parallax:
        Parallax angle with angular units (e.g. milliarcseconds).

    Returns
    -------
    astropy.units.Quantity
        Distance (parsecs).

    Notes
    -----
    This is a simple 1/p conversion (distance in parsecs for parallax in arcseconds).
    It does not apply priors or correct for measurement uncertainties.
    """
    parallax_arcsec = parallax.to(u.arcsecond)
    return (1.0 / parallax_arcsec.value) * u.parsec
