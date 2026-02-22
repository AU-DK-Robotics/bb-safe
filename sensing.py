import numpy as np
import time
import cv2
import math
from gripper_control import gripper

# Sensor full scales + saturation ranges (same concept as your reference)
FS = {
    "dist": 348.0,      "dist_clip": (2.0, 350.0),     # cm
    "arms": 5.0,        "arms_clip": (0.0, 5.0),       # units
    "force": 60.0,      "force_clip": (-30.0, 30.0),   # N
    "torque": 20.0,     "torque_clip": (-10.0, 10.0),  # Nm
}

def read(ur,ard=None,std=0.0,N=1,dt=1):
    dist_arr = np.zeros((N))
    arms_arr = np.zeros((2,N))
    wrench_arr = np.zeros((6,N))

    # Start keeping track of time
    t_start = time.perf_counter()

    # i from 0 to N-1
    for i in range(0,N):

        if ard:
            # Read the gripper sensors (ultrasonic range and arm forces)
            T_base_mat = getPoseMatrix(ur)
            _ , dist_arr[i], arms_arr[:,i] = gripper.getGripperSensors(ard,ur,T_base_mat)

        # Read the UR's 6-axis F-T sensor
        wrench_arr[:,i] = ur.getActualTCPForce()


        # Try to take measurements according to the schedule defined
        # by N and dt
        if N > i+1:
            while (time.perf_counter() - t_start) < (dt*(i+1)):
                pass

    dist_arr_orig  = dist_arr.copy()
    arms_arr_orig  = arms_arr.copy()
    wrench_arr_orig = wrench_arr.copy()

    if std>0:
        # add noise equivalent to the normalized base camera noise rate
        # by multiplying by each sensor's full scale, then clipping

        std_dist = std * FS["dist"]
        std_arms = std * FS["arms"]
        std_force = std * FS["force"]
        std_torque = std * FS["torque"]

        rand = np.random.default_rng()
        noise = rand.normal(loc=0,scale=std,size=(9,N))
        dist_arr = np.clip(dist_arr + noise[0,:]*FS["dist"],min=FS["dist_clip"][0],max=FS["dist_clip"][1])
        arms_arr = np.clip(arms_arr + noise[1:3,:]*FS["arms"],min=FS["arms_clip"][0],max=FS["arms_clip"][1])
        wrench_arr[0:3,:] = np.clip(wrench_arr[0:3,:] + noise[3:6,:]*FS["force"],min=FS["force_clip"][0],max=FS["force_clip"][1])
        wrench_arr[3:6,:] = np.clip(wrench_arr[3:6,:] + noise[6:9,:]*FS["torque"],min=FS["torque_clip"][0],max=FS["torque_clip"][1])

        # Predicted drift reports from the *clipped noisy samples*
        # sigma per channel = std_norm * full_scale
        predicted_drift_dist, _ = drift_report_from_clipped_samples(dist_arr, sigma=std_dist, L=FS["dist_clip"][0], U=FS["dist_clip"][1], N_nominal=N)
        predicted_drift_arm1, _ = drift_report_from_clipped_samples(arms_arr[0, :], sigma=std_arms, L=FS["arms_clip"][0], U=FS["arms_clip"][1], N_nominal=N)
        predicted_drift_arm2, _ = drift_report_from_clipped_samples(arms_arr[1, :], sigma=std_arms, L=FS["arms_clip"][0], U=FS["arms_clip"][1], N_nominal=N)
        predicted_drift_wrench_force_x, _ = drift_report_from_clipped_samples(wrench_arr[1, :], sigma=std_force, L=FS["force_clip"][0], U=FS["force_clip"][1], N_nominal=N)
        predicted_drift_wrench_force_y, _ = drift_report_from_clipped_samples(wrench_arr[2, :], sigma=std_force, L=FS["force_clip"][0], U=FS["force_clip"][1], N_nominal=N)
        predicted_drift_wrench_force_z, _ = drift_report_from_clipped_samples(wrench_arr[3, :], sigma=std_force, L=FS["force_clip"][0], U=FS["force_clip"][1], N_nominal=N)
        predicted_drift_wrench_torque_x, _ = drift_report_from_clipped_samples(wrench_arr[4, :], sigma=std_torque, L=FS["torque_clip"][0], U=FS["torque_clip"][1], N_nominal=N)
        predicted_drift_wrench_torque_y, _ = drift_report_from_clipped_samples(wrench_arr[5, :], sigma=std_torque, L=FS["torque_clip"][0], U=FS["torque_clip"][1], N_nominal=N)
        predicted_drift_wrench_torque_z, _ = drift_report_from_clipped_samples(wrench_arr[6, :], sigma=std_torque, L=FS["torque_clip"][0], U=FS["torque_clip"][1], N_nominal=N)
    else:
        predicted_drift_dist = 0.0
        predicted_drift_arm1 = 0.0
        predicted_drift_arm2 = 0.0
        predicted_drift_wrench_force_x = 0.0
        predicted_drift_wrench_force_y = 0.0
        predicted_drift_wrench_force_z = 0.0
        predicted_drift_wrench_torque_x = 0.0
        predicted_drift_wrench_torque_y = 0.0
        predicted_drift_wrench_torque_z = 0.0

    predicted_drifts = [
        predicted_drift_dist,
        predicted_drift_arm1,
        predicted_drift_arm2,
        predicted_drift_wrench_force_x,
        predicted_drift_wrench_force_y,
        predicted_drift_wrench_force_z,
        predicted_drift_wrench_torque_x,
        predicted_drift_wrench_torque_y,
        predicted_drift_wrench_torque_z
    ]

    if N > 1:
        dist_mean_corr = np.mean(dist_arr,axis=1)
        arms_mean_corr = np.mean(arms_arr,axis=1)
        wrench_mean_corr = np.mean(wrench_arr,axis=1)
    else:
        dist_mean_corr = dist_arr.flatten()
        arms_mean_corr = arms_arr.flatten()
        wrench_mean_corr = wrench_arr.flatten()

    dist_mean_corr[0] = dist_mean_corr[0] - predicted_drift_dist
    arms_mean_corr[0] = arms_mean_corr[0] - predicted_drift_arm1
    arms_mean_corr[1] = arms_mean_corr[1] - predicted_drift_arm2
    wrench_mean_corr[0] = wrench_mean_corr[0] - predicted_drift_wrench_force_x
    wrench_mean_corr[1] = wrench_mean_corr[1] - predicted_drift_wrench_force_y
    wrench_mean_corr[2] = wrench_mean_corr[2] - predicted_drift_wrench_force_z
    wrench_mean_corr[3] = wrench_mean_corr[3] - predicted_drift_wrench_torque_x
    wrench_mean_corr[4] = wrench_mean_corr[4] - predicted_drift_wrench_torque_y
    wrench_mean_corr[5] = wrench_mean_corr[5] - predicted_drift_wrench_torque_z

    return wrench_mean_corr, dist_mean_corr, arms_mean_corr, wrench_arr_orig, dist_arr_orig, arms_arr_orig, predicted_drifts

def getPoseMatrix(rtde_r):
    """
    Return ^base T_ee as a 4x4 homogeneous transformation matrix.
    UR 'getActualTCPPose()' returns [x, y, z, Rx, Ry, Rz] in meters / axis-angle.
    """
    tcp_pose = rtde_r.getActualTCPPose()
    x, y, z, rx, ry, rz = tcp_pose

    # Convert axis-angle (rotation vector) to rotation matrix
    rotation_vector = np.array([rx, ry, rz], dtype=float)
    R, _ = cv2.Rodrigues(rotation_vector)  # Convert to 3x3 rotation matrix

    # Build the 4x4 homogeneous transformation matrix
    pose_mat = np.eye(4)
    pose_mat[:3, :3] = R
    pose_mat[:3, 3] = [x, y, z]

    return pose_mat


# ============================================================
# --------- Tobit (censored normal) + profile CI + drift CI ---
# ============================================================

def _norm_pdf(x: float) -> float:
    return (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x * x)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _clipped_normal_moments(mu: float, sigma: float, L: float, U: float):
    """
    Y = clip(Z, L, U), Z ~ N(mu, sigma^2)
    Returns (E[Y], Var[Y]).
    """
    if sigma <= 0:
        Ey = float(np.clip(mu, L, U))
        return Ey, 0.0

    a = (L - mu) / sigma
    b = (U - mu) / sigma
    Phi_a, Phi_b = _norm_cdf(a), _norm_cdf(b)
    phi_a, phi_b = _norm_pdf(a), _norm_pdf(b)

    P_L = Phi_a
    P_U = 1.0 - Phi_b
    P_M = Phi_b - Phi_a

    # E[Z 1(L<Z<U)]
    EZ_mid = mu * P_M + sigma * (phi_a - phi_b)
    Ey = L * P_L + U * P_U + EZ_mid

    # E[Z^2 1(L<Z<U)]
    EZ2_mid = (mu * mu + sigma * sigma) * P_M + sigma * (mu + L) * phi_a - sigma * (mu + U) * phi_b
    Ey2 = (L * L) * P_L + (U * U) * P_U + EZ2_mid

    Vy = max(0.0, Ey2 - Ey * Ey)
    return Ey, Vy

def _summarize_censored(y: np.ndarray, L: float, U: float, eps: float = 1e-12):
    """
    Summarize clipped samples into censoring statistics.
    Uses only mid-sample sum/sumsq for speed/stability.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None

    low = y <= (L + eps)
    high = y >= (U - eps)
    mid = (~low) & (~high)

    y_mid = y[mid]
    return {
        "N_eff": int(y.size),
        "ybar": float(np.mean(y)),
        "n_low": int(np.sum(low)),
        "n_high": int(np.sum(high)),
        "n_mid": int(np.sum(mid)),
        "sum_mid": float(np.sum(y_mid)) if y_mid.size else 0.0,
        "sumsq_mid": float(np.sum(y_mid * y_mid)) if y_mid.size else 0.0,
        "low_frac": float(np.mean(low)),
        "high_frac": float(np.mean(high)),
    }

def _tobit_loglike_from_stats(stats: dict, mu: float, sigma: float, L: float, U: float, eps: float = 1e-12):
    """
    Censored-normal log-likelihood using only censoring counts + mid sums.
    """
    if sigma <= 0:
        return -np.inf

    nL = stats["n_low"]
    nU = stats["n_high"]
    nM = stats["n_mid"]
    S1 = stats["sum_mid"]
    S2 = stats["sumsq_mid"]

    a = (L - mu) / sigma
    b = (U - mu) / sigma
    Phi_a = _norm_cdf(a)
    Phi_b = _norm_cdf(b)

    ll = 0.0
    if nL > 0:
        ll += nL * math.log(max(Phi_a, eps))
    if nU > 0:
        ll += nU * math.log(max(1.0 - Phi_b, eps))

    if nM > 0:
        # sum((y - mu)^2) = S2 - 2 mu S1 + nM mu^2
        quad = (S2 - 2.0 * mu * S1 + nM * mu * mu)
        ll += -0.5 * quad / (sigma * sigma) - nM * math.log(sigma) - 0.5 * nM * math.log(2.0 * math.pi)

    return ll

def _tobit_mle_and_profile_ci_from_stats(
    stats: dict,
    sigma: float,
    L: float,
    U: float,
    grid_pad_sig: float = 8.0,
    grid_n: int = 4001,
):
    """
    Grid MLE for mu + Wilks profile-likelihood CI (approx).
    95% CI threshold: 2*(ll_hat - ll(mu)) <= 3.84  -> ll(mu) >= ll_hat - 1.92
    Returns: (mu_hat, (mu_lo, mu_hi), ll_hat)
    """
    if stats is None or stats["N_eff"] == 0 or sigma <= 0:
        return None, (None, None), None
    if stats["n_mid"] == 0:
        return None, (None, None), None  # fully censored, mu not identifiable

    ybar = stats["ybar"]
    lo = min(L, ybar - grid_pad_sig * sigma)
    hi = max(U, ybar + grid_pad_sig * sigma)

    mus = np.linspace(lo, hi, grid_n)
    lls = np.array([_tobit_loglike_from_stats(stats, float(m), sigma, L, U) for m in mus], dtype=np.float64)

    i_hat = int(np.argmax(lls))
    mu_hat = float(mus[i_hat])
    ll_hat = float(lls[i_hat])

    ll_cut = ll_hat - 0.5 * 3.841458820694124  # 1.9207...
    ok = lls >= ll_cut
    if not np.any(ok):
        return mu_hat, (None, None), ll_hat

    idx = np.where(ok)[0]
    mu_lo = float(mus[idx[0]])
    mu_hi = float(mus[idx[-1]])
    return mu_hat, (mu_lo, mu_hi), ll_hat

def drift_report_from_clipped_samples(y: np.ndarray, *, sigma: float, L: float, U: float, N_nominal: int = 50):
    """
    Prior-free 'rigorous' report:
      - Tobit MLE mu_hat
      - 95% profile-likelihood CI for mu
      - drift_hat = E[clip]-mu_hat
      - drift CI induced by mu CI
      - fit residual: ybar - E_clip(mu_hat)

    Uses only noisy clipped samples + known sigma + bounds.
    """

    drift_hat = None

    stats = _summarize_censored(y, L, U)
    if stats is None:
        return drift_hat, {"status": "no_data", "sigma": float(sigma), "L": float(L), "U": float(U)}

    if sigma <= 0:
        return drift_hat, {
            "status": "sigma_zero",
            "sigma": float(sigma),
            "L": float(L),
            "U": float(U),
            "N_nominal": int(N_nominal),
            "stats": stats,
            "mu_hat": None,
            "mu_ci95": [None, None],
            "E_clip_mu_hat": None,
            "predicted_drift_mean": 0.0,
            "drift_ci95": [0.0, 0.0],
            "fit_residual": None,
            "SE_sample_mean": 0.0,
            "ll_hat": None,
            "note": "sigma<=0: no stochastic noise -> no noise+clipping drift to predict.",
        }

    mu_hat, mu_ci, ll_hat = _tobit_mle_and_profile_ci_from_stats(stats, sigma, L, U)
    out = {
        "status": "ok" if mu_hat is not None else "fully_censored",
        "sigma": float(sigma),
        "L": float(L),
        "U": float(U),
        "N_nominal": int(N_nominal),
        "stats": stats,
        "mu_hat": mu_hat,
        "mu_ci95": [mu_ci[0], mu_ci[1]],
        "ll_hat": ll_hat,
    }

    if mu_hat is None:
        return drift_hat, out

    Ey, Vy = _clipped_normal_moments(mu_hat, sigma, L, U)
    N_eff = max(1, stats["N_eff"])
    drift_hat = float(Ey - mu_hat)
    fit_residual = float(stats["ybar"] - Ey)

    # Drift CI induced by mu CI (scan)
    drift_ci = (None, None)
    mu_lo, mu_hi = mu_ci
    if (mu_lo is not None) and (mu_hi is not None) and (mu_hi > mu_lo):
        mus = np.linspace(mu_lo, mu_hi, 2001)
        dr = np.array([_clipped_normal_moments(float(m), sigma, L, U)[0] - float(m) for m in mus], dtype=np.float64)
        drift_ci = (float(np.min(dr)), float(np.max(dr)))

    out.update({
        "E_clip_mu_hat": float(Ey),
        "Var_clip_mu_hat": float(Vy),
        "predicted_drift_mean": drift_hat,
        "drift_ci95": [drift_ci[0], drift_ci[1]],
        "fit_residual": fit_residual,
        "SE_sample_mean": float(math.sqrt(Vy / N_eff)),
    })
    return drift_hat, out
