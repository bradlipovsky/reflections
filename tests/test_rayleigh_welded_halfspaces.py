import numpy as np

from rayleigh_welded_halfspaces import poynting_x, rayleigh_mode, solve_interface_reflection


def _default_setup():
    omega = 2.0 * np.pi
    z = np.linspace(0.0, 5.0, 500)
    n_basis = 7
    n_unknown = 2 + 4 * n_basis
    z_match = np.linspace(0.0, 3.5, max(50, 2 * n_unknown))
    return omega, z, z_match, n_basis


def test_identical_media_gives_r_near_zero_t_near_one():
    medium = dict(alpha=6.0, beta=3.5, rho=2.7)
    omega, z, z_match, n_basis = _default_setup()

    sol = solve_interface_reflection(medium, medium, omega, z, z_match, n_basis=n_basis)

    assert abs(sol["r"]) < 5e-2
    assert abs(sol["t"] - 1.0) < 5e-2


def test_free_surface_traction_is_near_zero_for_each_rayleigh_mode():
    medium = dict(alpha=6.0, beta=3.5, rho=2.7)
    omega, z, _, _ = _default_setup()

    for phase_sign in (+1, -1):
        mode = rayleigh_mode(medium["alpha"], medium["beta"], medium["rho"], omega, z, phase_sign=phase_sign)

        sxz_rel = abs(mode["sigma_xz"][0]) / (np.max(np.abs(mode["sigma_xz"])) + 1e-15)
        szz_rel = abs(mode["sigma_zz"][0]) / (np.max(np.abs(mode["sigma_zz"])) + 1e-15)

        assert sxz_rel < 1e-8
        assert szz_rel < 1e-8


def test_reflected_mode_has_negative_integrated_poynting_flux():
    medium = dict(alpha=6.0, beta=3.5, rho=2.7)
    omega, z, _, _ = _default_setup()

    reflected = rayleigh_mode(medium["alpha"], medium["beta"], medium["rho"], omega, z, phase_sign=-1)
    flux_ref = np.trapz(
        poynting_x(reflected["sigma_xx"], reflected["sigma_xz"], reflected["vx"], reflected["vz"]),
        z,
    )

    assert np.real(flux_ref) < 0.0
