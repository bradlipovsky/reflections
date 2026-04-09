import numpy as np
import matplotlib.pyplot as plt


def rayleigh_speed(alpha, beta):
    """Return Rayleigh speed c_R for a homogeneous isotropic half-space."""
    kappa = beta / alpha
    coeff = [1.0, 0.0, -8.0, 0.0, 8.0 * (3.0 - 2.0 * kappa ** 2), 0.0, -16.0 * (1.0 - kappa ** 2)]
    roots = np.roots(coeff)
    candidates = [r.real for r in roots if abs(r.imag) < 1e-10 and 0.0 < r.real < 1.0]
    if not candidates:
        raise RuntimeError("No physical Rayleigh root found.")
    return max(candidates) * beta


def lame_from_vp_vs_rho(alpha, beta, rho):
    mu = rho * beta ** 2
    lam = rho * alpha ** 2 - 2.0 * mu
    return lam, mu


def poynting_x(sig_xx, sig_xz, vx, vz):
    return -0.5 * np.real(sig_xx * np.conj(vx) + sig_xz * np.conj(vz))


def rayleigh_mode(alpha, beta, rho, omega, z, phase_sign=+1, amplitude=1.0 + 0j):
    """
    Exact Rayleigh eigenfunction in a homogeneous half-space.

    phase_sign = +1 : +x propagation, exp(+i k x - i omega t)
    phase_sign = -1 : -x propagation, exp(-i k x - i omega t)
    """
    cR = rayleigh_speed(alpha, beta)
    k = omega / cR
    lam, mu = lame_from_vp_vs_rho(alpha, beta, rho)

    p = np.sqrt(k ** 2 - (omega / alpha) ** 2 + 0j)
    q = np.sqrt(k ** 2 - (omega / beta) ** 2 + 0j)

    s = phase_sign
    A = amplitude
    B = -(2j * s * k * p) / (k ** 2 + q ** 2) * A

    ep = np.exp(-p * z)
    eq = np.exp(-q * z)

    phi = A * ep
    psi = B * eq
    phi_z = -p * phi
    psi_z = -q * psi
    phi_x = 1j * s * k * phi
    psi_x = 1j * s * k * psi

    ux = phi_x - psi_z
    uz = phi_z + psi_x

    vx = -1j * omega * ux
    vz = -1j * omega * uz

    div_u = (p ** 2 - k ** 2) * phi
    sigma_xx = lam * div_u + 2.0 * mu * (1j * s * k * ux)
    sigma_xz = mu * (phi_x.conjugate() * 0.0 + (2j * s * k * phi_z - (k ** 2 + q ** 2) * psi))
    sigma_zz = lam * div_u + 2.0 * mu * phi_z * (-p) + 2.0 * mu * psi_x * (1j * s * k)

    return {
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "lambda": lam,
        "mu": mu,
        "omega": omega,
        "cR": cR,
        "k": k,
        "p": p,
        "q": q,
        "A": A,
        "B": B,
        "ux": ux,
        "uz": uz,
        "vx": vx,
        "vz": vz,
        "sigma_xx": sigma_xx,
        "sigma_xz": sigma_xz,
        "sigma_zz": sigma_zz,
    }


def bulk_basis_component(alpha, beta, rho, omega, z, kx, pol, side, normalize_surface=True):
    """Build one evanescent P or SV basis field for interface matching."""
    lam, mu = lame_from_vp_vs_rho(alpha, beta, rho)

    if pol.upper() == "P":
        kz = np.sqrt(kx ** 2 - (omega / alpha) ** 2 + 0j)
    elif pol.upper() == "SV":
        kz = np.sqrt(kx ** 2 - (omega / beta) ** 2 + 0j)
    else:
        raise ValueError("pol must be 'P' or 'SV'.")

    if np.real(kz) < 0.0:
        kz = -kz

    if side.upper() == "L":
        sx = -1.0
    elif side.upper() == "R":
        sx = +1.0
    else:
        raise ValueError("side must be 'L' or 'R'.")

    expz = np.exp(-kz * z)

    if pol.upper() == "P":
        phi = expz
        psi = 0.0 * expz
    else:
        phi = 0.0 * expz
        psi = expz

    d_dx = 1j * sx * kx
    d_dz = -kz

    ux = d_dx * phi - d_dz * psi
    uz = d_dz * phi + d_dx * psi

    div_u = (-kx ** 2 + kz ** 2) * phi
    sigma_xx = lam * div_u + 2.0 * mu * d_dx * ux
    sigma_xz = mu * (d_dz * ux + d_dx * uz)
    vx = -1j * omega * ux
    vz = -1j * omega * uz

    if normalize_surface:
        scale = np.sqrt(np.abs(ux[0]) ** 2 + np.abs(uz[0]) ** 2)
        if scale > 0:
            ux = ux / scale
            uz = uz / scale
            sigma_xx = sigma_xx / scale
            sigma_xz = sigma_xz / scale
            vx = vx / scale
            vz = vz / scale

    return {
        "pol": pol.upper(),
        "side": side.upper(),
        "kx": kx,
        "kz": kz,
        "ux": ux,
        "uz": uz,
        "sigma_xx": sigma_xx,
        "sigma_xz": sigma_xz,
        "vx": vx,
        "vz": vz,
    }


def build_bulk_basis(props, omega, z, side, n_basis=8, kx_min_factor=1.05, kx_max_factor=3.5):
    """Build a small set of evanescent P/SV components on one side."""
    alpha = props["alpha"]
    beta = props["beta"]
    rho = props["rho"]

    kR = omega / rayleigh_speed(alpha, beta)
    kmin = kx_min_factor * kR
    kmax = kx_max_factor * kR
    kxs = np.linspace(kmin, kmax, n_basis)

    basis = []
    for kx in kxs:
        basis.append(bulk_basis_component(alpha, beta, rho, omega, z, kx, "P", side))
        basis.append(bulk_basis_component(alpha, beta, rho, omega, z, kx, "SV", side))
    return basis


def solve_interface_reflection(left, right, omega, z, z_match, n_basis=8):
    """Solve for reflected/transmitted amplitudes plus evanescent basis coefficients."""
    mode_L_inc = rayleigh_mode(left["alpha"], left["beta"], left["rho"], omega, z, phase_sign=+1)
    mode_L_ref = rayleigh_mode(left["alpha"], left["beta"], left["rho"], omega, z, phase_sign=-1)
    mode_R_tra = rayleigh_mode(right["alpha"], right["beta"], right["rho"], omega, z, phase_sign=+1)

    basis_L = build_bulk_basis(left, omega, z, side="L", n_basis=n_basis)
    basis_R = build_bulk_basis(right, omega, z, side="R", n_basis=n_basis)

    idx = np.array([np.argmin(np.abs(z - zm)) for zm in z_match], dtype=int)
    zm = z[idx]

    n_unknown = 1 + len(basis_L) + 1 + len(basis_R)
    n_eq = 4 * len(idx)
    A = np.zeros((n_eq, n_unknown), dtype=complex)
    b = np.zeros(n_eq, dtype=complex)

    col_r = 0
    col_t = 1 + len(basis_L)

    row = 0
    for _, k in enumerate(idx):
        A[row, col_r] = mode_L_ref["ux"][k]
        for m, comp in enumerate(basis_L):
            A[row, 1 + m] = comp["ux"][k]
        A[row, col_t] = -mode_R_tra["ux"][k]
        for m, comp in enumerate(basis_R):
            A[row, col_t + 1 + m] = -comp["ux"][k]
        b[row] = -mode_L_inc["ux"][k]
        row += 1

        A[row, col_r] = mode_L_ref["uz"][k]
        for m, comp in enumerate(basis_L):
            A[row, 1 + m] = comp["uz"][k]
        A[row, col_t] = -mode_R_tra["uz"][k]
        for m, comp in enumerate(basis_R):
            A[row, col_t + 1 + m] = -comp["uz"][k]
        b[row] = -mode_L_inc["uz"][k]
        row += 1

        A[row, col_r] = mode_L_ref["sigma_xx"][k]
        for m, comp in enumerate(basis_L):
            A[row, 1 + m] = comp["sigma_xx"][k]
        A[row, col_t] = -mode_R_tra["sigma_xx"][k]
        for m, comp in enumerate(basis_R):
            A[row, col_t + 1 + m] = -comp["sigma_xx"][k]
        b[row] = -mode_L_inc["sigma_xx"][k]
        row += 1

        A[row, col_r] = mode_L_ref["sigma_xz"][k]
        for m, comp in enumerate(basis_L):
            A[row, 1 + m] = comp["sigma_xz"][k]
        A[row, col_t] = -mode_R_tra["sigma_xz"][k]
        for m, comp in enumerate(basis_R):
            A[row, col_t + 1 + m] = -comp["sigma_xz"][k]
        b[row] = -mode_L_inc["sigma_xz"][k]
        row += 1

    x_ls, residuals, rank, svals = np.linalg.lstsq(A, b, rcond=None)
    r = x_ls[col_r]
    t = x_ls[col_t]
    cL = x_ls[1:1 + len(basis_L)]
    cR = x_ls[col_t + 1:]

    uxL = mode_L_inc["ux"] + r * mode_L_ref["ux"]
    uzL = mode_L_inc["uz"] + r * mode_L_ref["uz"]
    sxxL = mode_L_inc["sigma_xx"] + r * mode_L_ref["sigma_xx"]
    sxzL = mode_L_inc["sigma_xz"] + r * mode_L_ref["sigma_xz"]
    for coeff, comp in zip(cL, basis_L):
        uxL += coeff * comp["ux"]
        uzL += coeff * comp["uz"]
        sxxL += coeff * comp["sigma_xx"]
        sxzL += coeff * comp["sigma_xz"]

    uxR = t * mode_R_tra["ux"]
    uzR = t * mode_R_tra["uz"]
    sxxR = t * mode_R_tra["sigma_xx"]
    sxzR = t * mode_R_tra["sigma_xz"]
    for coeff, comp in zip(cR, basis_R):
        uxR += coeff * comp["ux"]
        uzR += coeff * comp["uz"]
        sxxR += coeff * comp["sigma_xx"]
        sxzR += coeff * comp["sigma_xz"]

    misfit = {
        "ux": uxL[idx] - uxR[idx],
        "uz": uzL[idx] - uzR[idx],
        "sigma_xx": sxxL[idx] - sxxR[idx],
        "sigma_xz": sxzL[idx] - sxzR[idx],
    }

    return {
        "mode_L_inc": mode_L_inc,
        "mode_L_ref": mode_L_ref,
        "mode_R_tra": mode_R_tra,
        "basis_L": basis_L,
        "basis_R": basis_R,
        "r": r,
        "t": t,
        "cL": cL,
        "cR": cR,
        "z_match": zm,
        "idx_match": idx,
        "A": A,
        "b": b,
        "residuals": residuals,
        "rank": rank,
        "singular_values": svals,
        "uxL0": uxL,
        "uzL0": uzL,
        "sxxL0": sxxL,
        "sxzL0": sxzL,
        "uxR0": uxR,
        "uzR0": uzR,
        "sxxR0": sxxR,
        "sxzR0": sxzR,
        "misfit": misfit,
    }


def synthesize_field(x, z, sol, x_decay_clip=6.0):
    """Build a visualization field away from the interface using solved basis."""
    mode_L_inc = sol["mode_L_inc"]
    mode_L_ref = sol["mode_L_ref"]
    mode_R_tra = sol["mode_R_tra"]
    r = sol["r"]
    t = sol["t"]
    basis_L = sol["basis_L"]
    basis_R = sol["basis_R"]
    cL = sol["cL"]
    cR = sol["cR"]

    nx = len(x)
    nz = len(z)
    ux = np.zeros((nx, nz), dtype=complex)
    uz = np.zeros((nx, nz), dtype=complex)

    for i, xv in enumerate(x):
        if xv < 0.0:
            ux[i, :] += mode_L_inc["ux"] * np.exp(1j * mode_L_inc["k"] * xv)
            uz[i, :] += mode_L_inc["uz"] * np.exp(1j * mode_L_inc["k"] * xv)
            ux[i, :] += r * mode_L_ref["ux"] * np.exp(-1j * mode_L_ref["k"] * xv)
            uz[i, :] += r * mode_L_ref["uz"] * np.exp(-1j * mode_L_ref["k"] * xv)
            for coeff, comp in zip(cL, basis_L):
                factor = np.exp(comp["kx"] * xv)
                factor = np.where(np.real(comp["kx"] * xv) < -x_decay_clip, np.exp(-x_decay_clip), factor)
                ux[i, :] += coeff * comp["ux"] * factor
                uz[i, :] += coeff * comp["uz"] * factor
        else:
            ux[i, :] += t * mode_R_tra["ux"] * np.exp(1j * mode_R_tra["k"] * xv)
            uz[i, :] += t * mode_R_tra["uz"] * np.exp(1j * mode_R_tra["k"] * xv)
            for coeff, comp in zip(cR, basis_R):
                factor = np.exp(-comp["kx"] * xv)
                factor = np.where(np.real(comp["kx"] * xv) > x_decay_clip, np.exp(-x_decay_clip), factor)
                ux[i, :] += coeff * comp["ux"] * factor
                uz[i, :] += coeff * comp["uz"] * factor
    return ux, uz


def run_demo(show=True, output_prefix="mm_"):
    left = dict(alpha=6.0, beta=3.5, rho=2.7)
    right = dict(alpha=7.2, beta=4.2, rho=3.0)

    f = 1.0
    omega = 2.0 * np.pi * f
    z = np.linspace(0.0, 5.0, 500)

    n_basis = 7
    n_unknown = 2 + 4 * n_basis
    z_match = np.linspace(0.0, 3.5, max(50, 2 * n_unknown))

    sol = solve_interface_reflection(left, right, omega, z, z_match, n_basis=n_basis)

    mode_L_inc = sol["mode_L_inc"]
    mode_L_ref = sol["mode_L_ref"]
    mode_R_tra = sol["mode_R_tra"]

    flux_inc = np.trapz(poynting_x(mode_L_inc["sigma_xx"], mode_L_inc["sigma_xz"], mode_L_inc["vx"], mode_L_inc["vz"]), z)
    flux_ref_mag = np.trapz(poynting_x(mode_L_ref["sigma_xx"], mode_L_ref["sigma_xz"], mode_L_ref["vx"], mode_L_ref["vz"]), z)
    flux_tra = np.trapz(poynting_x(mode_R_tra["sigma_xx"], mode_R_tra["sigma_xz"], mode_R_tra["vx"], mode_R_tra["vz"]), z)

    Pi_I = np.real(flux_inc)
    Pi_R_mode = -np.real(flux_ref_mag)
    Pi_T_mode = np.real(flux_tra)

    r = sol["r"]
    t = sol["t"]
    R = abs(r) ** 2 * Pi_R_mode / Pi_I
    T = abs(t) ** 2 * Pi_T_mode / Pi_I

    print(f"r = {r.real:.6e} + {r.imag:.6e}j")
    print(f"t = {t.real:.6e} + {t.imag:.6e}j")
    print(f"R = {R:.6f}, T = {T:.6f}, R+T = {R+T:.6f}")

    mis = sol["misfit"]
    fig2, ax2 = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    zm = sol["z_match"]
    ax2[0, 0].plot(np.real(mis["ux"]), zm)
    ax2[0, 0].invert_yaxis()
    ax2[0, 0].set_title("Re[ux_L - ux_R] at x=0")
    ax2[0, 1].plot(np.real(mis["uz"]), zm)
    ax2[0, 1].invert_yaxis()
    ax2[0, 1].set_title("Re[uz_L - uz_R] at x=0")
    ax2[1, 0].plot(np.real(mis["sigma_xx"]), zm)
    ax2[1, 0].invert_yaxis()
    ax2[1, 0].set_title("Re[sxx_L - sxx_R] at x=0")
    ax2[1, 1].plot(np.real(mis["sigma_xz"]), zm)
    ax2[1, 1].invert_yaxis()
    ax2[1, 1].set_title("Re[sxz_L - sxz_R] at x=0")
    fig2.savefig(f"{output_prefix}interface_misfit.png", dpi=150)

    lamR_left = mode_L_inc["cR"] / f
    x = np.linspace(-3.0 * lamR_left, 3.0 * lamR_left, 600)
    ux_field, uz_field = synthesize_field(x, z, sol)
    amp = np.sqrt(np.abs(ux_field) ** 2 + np.abs(uz_field) ** 2)
    X, Z = np.meshgrid(x, z, indexing="ij")
    fig3, ax3 = plt.subplots(figsize=(10.5, 4.2), constrained_layout=True)
    im = ax3.pcolormesh(X, Z, amp, shading="auto")
    ax3.axvline(0.0, color="w", ls="--", lw=1.2)
    ax3.invert_yaxis()
    ax3.set_xlabel("x")
    ax3.set_ylabel("z")
    ax3.set_title("Synthesized displacement amplitude |u(x,z)|")
    plt.colorbar(im, ax=ax3, label="|u|")
    fig3.savefig(f"{output_prefix}field_amplitude.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close("all")

    return sol
