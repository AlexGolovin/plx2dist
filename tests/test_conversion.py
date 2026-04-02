import astropy.units as u

from plx2dist.conversion import parallax_to_distance


def test_parallax_to_distance_one_parsec():
    distance = parallax_to_distance(1000 * u.milliarcsecond)
    assert distance.to(u.parsec).value == 1.0
