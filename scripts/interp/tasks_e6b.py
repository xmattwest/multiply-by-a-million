"""tasks_e6b.py (POD): BOS/format-orthogonalized, layer-covariance-matched random controls for E6 Part B.

Registered automatically by tasks.py._load_plugins(). Must be run after e6_build and e6_dirs.
Inserts 'rand_cov' and 'rand_orth' into tasks_e6._DIR[L] so tasks_e6.e6_steer can steer with them directly.
"""
import torch

import tasks


def _unit(v, size):
    return v / v.norm().clamp_min(1e-12) * size


def e6b_dirs(model, tok, items, settings, model_name, model_io, layers=(42,), role="seller", k_fmt=8, seed=20261122, **kw):
    """Build layer-covariance-matched, format-orthogonalized random directions at `layers`."""
    e6_g = tasks.TASKS["e6_dirs"].__globals__
    E5 = e6_g["E5"]
    E4 = e6_g["E4"]
    E6_DIR = e6_g["_DIR"]
    out = []
    with E5._mode(role):
        st_, sg = E4._STORE, E4._STORE["segs"]
        if not st_["d"] or sg is None:
            raise ValueError("run e6_build and e6_dirs first")
        i0, i1 = sg["reply"]
        for L in sorted(int(x) for x in layers):
            if L not in E6_DIR:
                raise ValueError(f"layer {L} missing from E6._DIR; run e6_dirs first")
            vR = E6_DIR[L]["vR"]
            size = vR.norm()
            dev = vR.device

            # Collect residual-stream vectors at layer L across all fixtures & prompt/reply tokens
            idx = st_["keep"].index(L)
            all_rows = []
            fmt_rows = []
            for fid, rec in st_["d"].items():
                h = rec[idx].float().cpu()  # (T, d_model)
                all_rows.append(h)
                fmt_rows.append(h[i0:i1])
            H = torch.cat(all_rows, dim=0)
            H_c = H - H.mean(0, keepdim=True)
            F = torch.cat(fmt_rows, dim=0)
            F_c = F - F.mean(0, keepdim=True)
            # Top-k_fmt right singular vectors of reply/format header tokens (+ mean format vector)
            _, _, Vh = torch.linalg.svd(F_c, full_matrices=False)
            k = min(int(k_fmt), Vh.shape[0])
            U_fmt = Vh[:k].T  # (d_model, k)
            # Also include the normalized mean format vector so projection removes format DC shift
            f_mean = F.mean(0)
            f_mean = f_mean - U_fmt @ (U_fmt.T @ f_mean)
            if f_mean.norm() > 1e-8:
                U_fmt = torch.cat([U_fmt, (f_mean / f_mean.norm())[:, None]], dim=1)

            # 1. rand_cov: sampled in empirical residual covariance span, orthogonalized to U_fmt
            g1 = torch.Generator(device="cpu").manual_seed(int(seed) + L)
            eps = torch.randn(H_c.shape[0], generator=g1, dtype=torch.float32)
            v_cov = H_c.T @ eps
            v_cov = v_cov - U_fmt @ (U_fmt.T @ v_cov)
            v_cov = _unit(v_cov.to(dev), size)

            # 2. rand_orth: isotropic Gaussian orthogonalized to U_fmt
            g2 = torch.Generator(device="cpu").manual_seed(int(seed) + 1000 + L)
            v_orth = torch.randn(vR.shape, generator=g2, dtype=torch.float32)
            v_orth = v_orth - U_fmt @ (U_fmt.T @ v_orth)
            v_orth = _unit(v_orth.to(dev), size)

            E6_DIR[L]["rand_cov"] = v_cov
            E6_DIR[L]["rand_orth"] = v_orth
            cos = lambda a, b: round(float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-12)), 4)  # noqa: E731
            out.append({
                "item_id": f"e6b-dirs-L{L}",
                "layer": L,
                "n_rows": int(H.shape[0]),
                "k_fmt_dims": int(U_fmt.shape[1]),
                "norm": round(float(size), 4),
                "cos_vR_rand_cov": cos(vR, v_cov),
                "cos_vR_rand_orth": cos(vR, v_orth),
            })
    return out


TASKS = {"e6b_dirs": e6b_dirs}
