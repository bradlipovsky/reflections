import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Semi-analytical Rayleigh reflection at a welded vertical
# interface between two isotropic elastic half-spaces.
#
# This version avoids the full x-z finite-difference solve.
# Instead it solves a depth-matched interface problem at x=0.
#
# Model:
#   x < 0 : homogeneous elastic half-space (left)
#   x > 0 : homogeneous elastic half-space (right)
#   z = 0 : traction-free free surface
#   time dependence exp(-i omega t)
#
# Unknowns:
#   - reflected Rayleigh amplitude r on the left
#   - transmitted Rayleigh amplitude t on the right
#   - coefficients of additional evanescent P/SV terms on each side
#
# Interface matching:
#   continuity of ux, uz, sigma_xx, sigma_xz at x=0 on a set of
#   collocation depths z_m.
#
# Notes:
#   - For welded half-spaces there is only one discrete Rayleigh mode on
#     each side, so the near-interface mismatch must be absorbed by extra
#     evanescent P/SV content.
#   - The basis and assembly are written so the same strategy generalizes
#     naturally to a layer-over-half-space model: replace the vertical
#     basis on each side by the corresponding layered-half-space modal/
#     evanescent basis and keep the same interface matching machinery.
# ============================================================


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
    """
    Build one evanescent P or SV basis field for interface matching.

    side = 'L' or 'R'
      L: field decays away from x=0 into x<0, so x dependence is exp(-i kx x)
      R: field decays away from x=0 into x>0, so x dependence is exp(+i kx x)

    At x=0 only the sign of d/dx matters, handled below.

    kx can be complex. We choose kz with Re(kz)>0 so the basis decays with z.
    """
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
    """
    Build a small set of evanescent P/SV components on one side.

    We sample complex horizontal wavenumbers with positive real part larger
    than the propagating thresholds so the fields are evanescent both in x
    and z. This is not a complete continuum representation, but it works well
    as a compact semi-analytical approximation and generalizes nicely.
    """
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
    """
    Solve for reflected/transmitted Rayleigh amplitudes plus evanescent basis
    coefficients by least squares collocation in depth at x=0.
    """
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

    # Unknown ordering: [r, cL..., t, cR...]
    col_r = 0
    col_t = 1 + len(basis_L)

    row = 0
    for j, k in enumerate(idx):
        # ux continuity: uL_inc + r uL_ref + sum cL uL = t uR + sum cR uR
        A[row, col_r] = mode_L_ref["ux"][k]
        for m, comp in enumerate(basis_L):
            A[row, 1 + m] = comp["ux"][k]
        A[row, col_t] = -mode_R_tra["ux"][k]
        for m, comp in enumerate(basis_R):
            A[row, col_t + 1 + m] = -comp["ux"][k]
        b[row] = -mode_L_inc["ux"][k]
        row += 1

        # uz continuity
        A[row, col_r] = mode_L_ref["uz"][k]
        for m, comp in enumerate(basis_L):
            A[row, 1 + m] = comp["uz"][k]
        A[row, col_t] = -mode_R_tra["uz"][k]
        for m, comp in enumerate(basis_R):
            A[row, col_t + 1 + m] = -comp["uz"][k]
        b[row] = -mode_L_inc["uz"][k]
        row += 1

        # sigma_xx continuity
        A[row, col_r] = mode_L_ref["sigma_xx"][k]
        for m, comp in enumerate(basis_L):
            A[row, 1 + m] = comp["sigma_xx"][k]
        A[row, col_t] = -mode_R_tra["sigma_xx"][k]
        for m, comp in enumerate(basis_R):
            A[row, col_t + 1 + m] = -comp["sigma_xx"][k]
        b[row] = -mode_L_inc["sigma_xx"][k]
        row += 1

        # sigma_xz continuity
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

    # Reconstruct full depth profiles at x=0.
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


def synthesize_field(x, z, sol, side, x_decay_clip=6.0):
    """
    Build a visualization field away from the interface using the solved basis.

    This is mainly for diagnostics: it shows the incident/reflected/transmitted
    Rayleigh content plus the evanescent near-interface correction.
    """
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
                # Left-side evanescent fields decay as x -> -inf, so use exp(+Im? actually exp(kx x) if kx real >0 and x<0)
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


def main():
    # ------------------------------------------------------------
    # User-adjustable parameters
    # ------------------------------------------------------------
    left = dict(alpha=6.0, beta=3.5, rho=2.7)
    right = dict(alpha=7.2, beta=4.2, rho=3.0)

    f = 1.0
    omega = 2.0 * np.pi * f

    # Depth grid used for modal fields and collocation.
    zmax = 5.0
    nz = 500
    z = np.linspace(0.0, zmax, nz)

    # Interface collocation depths. More points than unknowns gives a
    # least-squares solve and helps stabilize the approximation.
    n_basis = 7
    n_unknown = 2 + 4 * n_basis
    n_match = max(50, 2 * n_unknown)
    z_match = np.linspace(0.0, 3.5, n_match)

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

    mis = sol["misfit"]
    rms_ux = np.sqrt(np.mean(np.abs(mis["ux"]) ** 2))
    rms_uz = np.sqrt(np.mean(np.abs(mis["uz"]) ** 2))
    rms_sxx = np.sqrt(np.mean(np.abs(mis["sigma_xx"]) ** 2))
    rms_sxz = np.sqrt(np.mean(np.abs(mis["sigma_xz"]) ** 2))

    print("Left medium:")
    print(f"  alpha = {left['alpha']:.4f}, beta = {left['beta']:.4f}, rho = {left['rho']:.4f}")
    print(f"  c_R(left) = {mode_L_inc['cR']:.6f}")
    print(f"  k_R(left) = {mode_L_inc['k']:.6f}")
    print()
    print("Right medium:")
    print(f"  alpha = {right['alpha']:.4f}, beta = {right['beta']:.4f}, rho = {right['rho']:.4f}")
    print(f"  c_R(right) = {mode_R_tra['cR']:.6f}")
    print(f"  k_R(right) = {mode_R_tra['k']:.6f}")
    print()

    print("Power normalization checks:")
    print(f"  Incident Rayleigh power      = {Pi_I:.6e}")
    print(f"  Reflected unit-mode power    = {Pi_R_mode:.6e}")
    print(f"  Transmitted unit-mode power  = {Pi_T_mode:.6e}")
    print()

    print("Scattering results:")
    print(f"  r = {r.real:.6e} + {r.imag:.6e}j")
    print(f"  t = {t.real:.6e} + {t.imag:.6e}j")
    print(f"  R = {R:.6f}")
    print(f"  T = {T:.6f}")
    print(f"  R + T = {R + T:.6f}")
    print("  Note: R+T need not equal 1 exactly here because the interface")
    print("  matching includes extra evanescent P/SV fields and the truncated")
    print("  basis is only an approximation to the continuous near-field content.")
    print()

    print("Interface continuity RMS misfit at x=0:")
    print(f"  ux        : {rms_ux:.6e}")
    print(f"  uz        : {rms_uz:.6e}")
    print(f"  sigma_xx  : {rms_sxx:.6e}")
    print(f"  sigma_xz  : {rms_sxz:.6e}")
    print(f"  LS rank   : {sol['rank']}")
    print(f"  cond(A)   : {sol['singular_values'][0] / sol['singular_values'][-1]:.6e}")
    print()

    # ------------------------------------------------------------
    # Diagnostic figures
    # ------------------------------------------------------------
    fig1, ax = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    ax[0].plot(np.real(mode_L_inc["ux"]), z, label="Re ux")
    ax[0].plot(np.real(mode_L_inc["uz"]), z, label="Re uz")
    ax[0].invert_yaxis()
    ax[0].set_title("Incident Rayleigh mode (left)")
    ax[0].set_xlabel("displacement")
    ax[0].set_ylabel("z")
    ax[0].legend()

    ax[1].plot(np.real(mode_R_tra["ux"]), z, label="Re ux")
    ax[1].plot(np.real(mode_R_tra["uz"]), z, label="Re uz")
    ax[1].invert_yaxis()
    ax[1].set_title("Transmitted Rayleigh mode (right)")
    ax[1].set_xlabel("displacement")
    ax[1].legend()

    ampL = np.sqrt(np.abs(mode_L_inc["ux"]) ** 2 + np.abs(mode_L_inc["uz"]) ** 2)
    ampR = np.sqrt(np.abs(mode_R_tra["ux"]) ** 2 + np.abs(mode_R_tra["uz"]) ** 2)
    ax[2].plot(ampL / np.max(ampL), z, label="left mode")
    ax[2].plot(ampR / np.max(ampR), z, label="right mode")
    ax[2].invert_yaxis()
    ax[2].set_title("Rayleigh depth decay")
    ax[2].set_xlabel("normalized |u|")
    ax[2].legend()
    fig1.savefig("mm_rayleigh_modes.png", dpi=150)

    fig2, ax2 = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    zm = sol["z_match"]
    ax2[0, 0].plot(np.real(mis["ux"]), zm)
    ax2[0, 0].invert_yaxis()
    ax2[0, 0].set_title("Re[ux_L - ux_R] at x=0")
    ax2[0, 0].set_ylabel("z")

    ax2[0, 1].plot(np.real(mis["uz"]), zm)
    ax2[0, 1].invert_yaxis()
    ax2[0, 1].set_title("Re[uz_L - uz_R] at x=0")

    ax2[1, 0].plot(np.real(mis["sigma_xx"]), zm)
    ax2[1, 0].invert_yaxis()
    ax2[1, 0].set_title("Re[sxx_L - sxx_R] at x=0")
    ax2[1, 0].set_xlabel("misfit")
    ax2[1, 0].set_ylabel("z")

    ax2[1, 1].plot(np.real(mis["sigma_xz"]), zm)
    ax2[1, 1].invert_yaxis()
    ax2[1, 1].set_title("Re[sxz_L - sxz_R] at x=0")
    ax2[1, 1].set_xlabel("misfit")
    fig2.savefig("mm_interface_misfit.png", dpi=150)

    # Visualize synthesized total field.
    lamR_left = mode_L_inc["cR"] / f
    x = np.linspace(-3.0 * lamR_left, 3.0 * lamR_left, 600)
    ux_field, uz_field = synthesize_field(x, z, sol, side=None)
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
    fig3.savefig("mm_field_amplitude.png", dpi=150)

    # Surface field and decomposition.
    iz0 = 0
    ux_surf = ux_field[:, iz0]
    incident_surf = np.zeros_like(x, dtype=complex)
    reflected_surf = np.zeros_like(x, dtype=complex)
    transmitted_surf = np.zeros_like(x, dtype=complex)
    for i, xv in enumerate(x):
        if xv < 0.0:
            incident_surf[i] = mode_L_inc["ux"][0] * np.exp(1j * mode_L_inc["k"] * xv)
            reflected_surf[i] = r * mode_L_ref["ux"][0] * np.exp(-1j * mode_L_ref["k"] * xv)
        else:
            transmitted_surf[i] = t * mode_R_tra["ux"][0] * np.exp(1j * mode_R_tra["k"] * xv)

    fig4, ax4 = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    ax4[0].plot(x, np.real(ux_surf), label="Re total surface ux")
    ax4[0].plot(x, np.real(incident_surf), "--", label="incident")
    ax4[0].plot(x, np.real(reflected_surf), "--", label="reflected")
    ax4[0].plot(x, np.real(transmitted_surf), "--", label="transmitted")
    ax4[0].axvline(0.0, color="k", ls=":")
    ax4[0].legend(ncol=2)
    ax4[0].set_ylabel("surface ux")

    ax4[1].plot(x, np.sqrt(np.abs(ux_surf) ** 2 + np.abs(uz_field[:, 0]) ** 2))
    ax4[1].axvline(0.0, color="k", ls=":")
    ax4[1].set_xlabel("x")
    ax4[1].set_ylabel("surface |u|")
    ax4[1].set_title("Surface field after interface matching")
    fig4.savefig("mm_surface_field.png", dpi=150)

    plt.show()


if __name__ == "__main__":
    main()
