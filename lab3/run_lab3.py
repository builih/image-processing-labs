"""ECE 60141 Lab 3: experiment driver.

Imports lab3.py, runs every check and experiment required by the handout, and
writes the figures into figures/ and the numbers into results.txt and
results.tex.

Usage:  python run_lab3.py
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import lab3
from lab3 import (a_identity, adjoint, adjoint_autograd, cost, cost_grad,
                  forward, gradient_descent, load_kernel, load_pgm,
                  prior_grad, step_size)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 0                      # the random seed, reported in the write-up
DATA = "data"
FIGS = "figures"
NUM_ITERS = 50
SIGMA_X_LIST = [0.01, 0.05, 0.25]
NRMSE_THRESHOLD = 0.002       # the 0.2% stopping rule

os.makedirs(FIGS, exist_ok=True)
torch.manual_seed(SEED)

R = {}      # every number that ends up in the report
log_lines = []


def log(line=""):
    print(line)
    log_lines.append(str(line))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def show(ax, img, title):
    """Display a (H, W) tensor with the fixed gray scale over [0, 1]."""
    ax.imshow(img.detach().numpy(), cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def rmse(x_hat, x):
    """Root mean squared error in [0, 1] units."""
    return torch.sqrt(((x_hat - x) ** 2).mean()).item()


def descend_tracked(y, a, sigma_w, sigma_x, num_iters, alpha=None):
    """Same iteration as lab3.gradient_descent, but also records

        NRMSE_n = ||x^(n) - x^(n-1)|| / ||x^(n-1)||

    returns (x_hat, cost_history, nrmse_history) where nrmse_history has
    length num_iters and nrmse_history[0] corresponds to n = 1.
    """
    if alpha is None:
        alpha = step_size(sigma_w, sigma_x)
    x = y.clone()
    cost_history = [cost(x, y, a, sigma_w, sigma_x).item()]
    nrmse_history = []
    for _ in range(num_iters):
        x_prev = x
        x = x_prev - alpha * cost_grad(x_prev, y, a, sigma_w, sigma_x)
        nrmse_history.append(
            (torch.linalg.norm(x - x_prev) / torch.linalg.norm(x_prev)).item())
        cost_history.append(cost(x, y, a, sigma_w, sigma_x).item())
    return x, cost_history, nrmse_history


def first_below(nrmse_history, threshold=NRMSE_THRESHOLD):
    """Return the first iteration n (1 based) with NRMSE_n < threshold."""
    for n, value in enumerate(nrmse_history, start=1):
        if value < threshold:
            return n
    return None


def convergence_figure(cost_history, nrmse_history, title, path):
    """The two panel convergence figure required in Steps 7 and 8."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(range(len(cost_history)), cost_history, lw=1.6)
    axes[0].set_xlabel("iteration $n$")
    axes[0].set_ylabel("cost $c(x^{(n)})$")
    axes[0].set_title("(a) cost versus iteration", fontsize=10)
    axes[0].grid(alpha=0.3)

    n = range(1, len(nrmse_history) + 1)
    axes[1].semilogy(n, nrmse_history, lw=1.6)
    axes[1].axhline(NRMSE_THRESHOLD, color="crimson", ls="--", lw=1.2,
                    label="0.2% threshold")
    axes[1].set_xlabel("iteration $n$")
    axes[1].set_ylabel(r"NRMSE$_n$")
    axes[1].set_title("(b) per-iteration image change", fontsize=10)
    axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def experiment(tag, x, y, a, sigma_w, label):
    """Run one restoration experiment at all three sigma_x values.

    Produces the image panel, the convergence figure, the error table numbers
    and the timing.  Returns a dict of results.
    """
    out = {"sigma_w": sigma_w, "rmse_y": rmse(y, x), "rows": []}

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), layout="constrained")
    show(axes[0, 0], y, f"$y$   (RMSE {out['rmse_y']:.4f})")

    for i, sigma_x in enumerate(SIGMA_X_LIST):
        alpha = step_size(sigma_w, sigma_x)
        x_hat, cost_history, nrmse_history = descend_tracked(
            y, a, sigma_w, sigma_x, NUM_ITERS)
        e = rmse(x_hat, x)
        n_stop = first_below(nrmse_history)
        x_stop, _, _ = descend_tracked(y, a, sigma_w, sigma_x, n_stop)
        out["rows"].append({"sigma_x": sigma_x, "alpha": alpha, "rmse": e,
                            "n_stop": n_stop, "rmse_at_stop": rmse(x_stop, x),
                            "cost_final": cost_history[-1]})
        ax = axes.flat[i + 1]
        show(ax, x_hat.clamp(0, 1),
             rf"$\hat{{x}}$, $\sigma_x={sigma_x}$   (RMSE {e:.4f})")

        if sigma_x == 0.05:
            convergence_figure(
                cost_history, nrmse_history,
                f"{label}, $\\sigma_x = 0.05$",
                f"{FIGS}/fig_{tag}_convergence.png")
            # confirm the tracked loop reproduces gradient_descent exactly
            x_ref, cost_ref = gradient_descent(y, a, sigma_w, sigma_x,
                                               NUM_ITERS)
            out["gd_agreement"] = (x_hat - x_ref).abs().max().item()

    fig.suptitle(label, fontsize=12)
    fig.savefig(f"{FIGS}/fig_{tag}_images.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # timing of one 50 iteration run at the good sigma_x
    t0 = time.perf_counter()
    gradient_descent(y, a, sigma_w, 0.05, NUM_ITERS)
    out["seconds"] = time.perf_counter() - t0

    best = min(out["rows"], key=lambda r: r["rmse"])
    out["best_sigma_x"] = best["sigma_x"]
    return out


# ---------------------------------------------------------------------------
# Load the data
# ---------------------------------------------------------------------------

x = load_pgm(f"{DATA}/kodim23.pgm")
a = load_kernel(f"{DATA}/levin09_kernel1.txt")
H, W = x.shape
N = H * W

log("=" * 70)
log(f"Lab 3.  image {tuple(x.shape)}, seed {SEED}")
log("=" * 70)
R["seed"] = SEED
R["H"], R["W"], R["N"] = H, W, N


# ---------------------------------------------------------------------------
# Task 3a: look at A and A^t
# ---------------------------------------------------------------------------

Ax = forward(x, a)
Atx = adjoint(x, a)
diff = Ax - Atx

fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), layout="constrained")
show(axes[0, 0], x, "$x$")
show(axes[0, 1], Ax, "$Ax$")
show(axes[1, 0], Atx, "$A^t x$")
show(axes[1, 1], (diff + 0.5).clamp(0, 1), "$Ax - A^t x$  (+0.5, clipped)")
fig.suptitle("Task 3a: the blur operator and its transpose, kernel 1",
             fontsize=12)
fig.savefig(f"{FIGS}/fig_d1_operators.png", dpi=150, bbox_inches="tight")
plt.close(fig)

R["kernel_shape"] = list(a.shape)
R["kernel_sum"] = a.sum().item()
R["kernel_nnz"] = int((a != 0).sum().item())
R["diff_absmax"] = diff.abs().max().item()

log("\n--- D1, Task 3a ---")
log(f"kernel shape         {tuple(a.shape)}")
log(f"kernel sum           {R['kernel_sum']:.12f}")
log(f"kernel nonzeros      {R['kernel_nnz']}")
log(f"max |Ax - A^t x|     {R['diff_absmax']:.6f}")


# ---------------------------------------------------------------------------
# Task 3b: the three adjoint checks
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
u = torch.randn(512, 768, dtype=torch.float64)
v = torch.randn(512, 768, dtype=torch.float64)

check1 = (adjoint(v, a) - adjoint_autograd(v, a)).abs().max()

lhs = (forward(u, a) * v).sum()
rhs = (u * adjoint(v, a)).sum()
check2 = (lhs - rhs).abs() / lhs.abs()

check3 = (forward(u, a) - adjoint(u, a)).abs().max()
fwd_absmax = forward(u, a).abs().max()

R["check1"] = check1.item()
R["check2"] = check2.item()
R["check3"] = check3.item()
R["fwd_absmax"] = fwd_absmax.item()

log("\n--- D2, Task 3b ---")
log(f"check1  max |adjoint - adjoint_autograd|   {R['check1']:.3e}")
log(f"check2  relative transpose-property error {R['check2']:.3e}")
log(f"check3  max |forward(u) - adjoint(u)|     {R['check3']:.6f}")
log(f"        max |forward(u)| for reference    {R['fwd_absmax']:.6f}")


# ---------------------------------------------------------------------------
# Task 3c: how big A would be as a matrix
# ---------------------------------------------------------------------------

dense_bytes = N * N * 8
sparse_bytes = N * R["kernel_nnz"] * (8 + 4)
kernel_bytes = a.numel() * 8

R["dense_bytes"] = dense_bytes
R["sparse_bytes"] = sparse_bytes
R["kernel_bytes"] = kernel_bytes

log("\n--- D3, Task 3c ---")
log(f"N = {H} x {W} = {N}")
log(f"dense   {N}^2 x 8 B  = {dense_bytes:.6e} B = {dense_bytes/1e12:.3f} TB")
log(f"sparse  {N} x {R['kernel_nnz']} x 12 B = {sparse_bytes:.6e} B "
    f"= {sparse_bytes/1e9:.3f} GB")
log(f"kernel  19 x 19 x 8 B = {kernel_bytes} B")


# ---------------------------------------------------------------------------
# Task 4a: the noncausal prediction error of the clean image
# ---------------------------------------------------------------------------

e = x - lab3._neighbor_sum(x)
interior = e[1:-1, 1:-1]
R["sigma_x_estimate"] = interior.std().item()

fig, ax = plt.subplots(figsize=(7, 5))
show(ax, (e + 0.5).clamp(0, 1),
     rf"noncausal prediction error $e$  (+0.5, clipped),  "
     rf"std $= {R['sigma_x_estimate']:.4f}$")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_d4_prediction_error.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)

log("\n--- D4, Task 4a ---")
log(f"std of interior prediction error  {R['sigma_x_estimate']:.6f}")


# ---------------------------------------------------------------------------
# Task 6a: the two remaining code checks
# ---------------------------------------------------------------------------

sigma_x_chk = 0.05
sigma_w_chk = 0.05
check4 = ((u * prior_grad(v, sigma_x_chk)).sum()
          - (v * prior_grad(u, sigma_x_chk)).sum()).abs()

y_small = torch.randn(32, 32, dtype=torch.float64)
z = torch.randn(32, 32, dtype=torch.float64, requires_grad=True)
auto = torch.autograd.grad(
    cost(z, y_small, a, sigma_w_chk, sigma_x_chk), z)[0]
check5 = (auto - cost_grad(z.detach(), y_small, a, sigma_w_chk,
                           sigma_x_chk)).abs().max()

bilinear_scale = (u * prior_grad(v, sigma_x_chk)).sum().abs()
R["check4"] = check4.item()
R["check4_rel"] = (check4 / bilinear_scale).item()
R["bilinear_scale"] = bilinear_scale.item()
R["check5"] = check5.item()

log("\n--- D6, Task 6a ---")
log(f"check4  |u^t B v - v^t B u|                 {R['check4']:.3e}  "
    f"(relative to |u^t B v| = {R['bilinear_scale']:.4e}: {R['check4_rel']:.3e})")
log(f"check5  max |autograd grad - cost_grad|     {R['check5']:.3e}")


# ---------------------------------------------------------------------------
# Step 7: denoising
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
sigma_w_dn = 0.05
y_dn = x + sigma_w_dn * torch.randn(H, W, dtype=torch.float64)
R["denoise"] = experiment("d8", x, y_dn, a_identity, sigma_w_dn,
                          r"Step 7: denoising, $A = I$, $\sigma_w = 0.05$")

log("\n--- D8, Step 7 denoising ---")
log(f"RMSE of y                {R['denoise']['rmse_y']:.6f}")
for row in R["denoise"]["rows"]:
    log(f"sigma_x {row['sigma_x']:<6} alpha {row['alpha']:.6e}  "
        f"RMSE {row['rmse']:.6f}  stop at n = {row['n_stop']}  "
        f"RMSE@stop {row['rmse_at_stop']:.6f}")
log(f"best sigma_x             {R['denoise']['best_sigma_x']}")
log(f"wall clock, 50 iters     {R['denoise']['seconds']:.2f} s")
log(f"tracked vs gradient_descent agreement  "
    f"{R['denoise']['gd_agreement']:.3e}")


# ---------------------------------------------------------------------------
# Step 8: deconvolution
# ---------------------------------------------------------------------------

torch.manual_seed(SEED)
sigma_w_dc = 0.02
y_dc = forward(x, a) + sigma_w_dc * torch.randn(H, W, dtype=torch.float64)
R["deconv"] = experiment(
    "d9", x, y_dc, a, sigma_w_dc,
    r"Step 8: deconvolution, camera shake kernel 1, $\sigma_w = 0.02$")

log("\n--- D9, Step 8 deconvolution ---")
log(f"RMSE of y                {R['deconv']['rmse_y']:.6f}")
for row in R["deconv"]["rows"]:
    log(f"sigma_x {row['sigma_x']:<6} alpha {row['alpha']:.6e}  "
        f"RMSE {row['rmse']:.6f}  stop at n = {row['n_stop']}  "
        f"RMSE@stop {row['rmse_at_stop']:.6f}")
log(f"best sigma_x             {R['deconv']['best_sigma_x']}")
log(f"wall clock, 50 iters     {R['deconv']['seconds']:.2f} s")
log(f"tracked vs gradient_descent agreement  "
    f"{R['deconv']['gd_agreement']:.3e}")


# ---------------------------------------------------------------------------
# Step 9 question 4: denoising with four times the stable step size
# ---------------------------------------------------------------------------

alpha_ok = step_size(sigma_w_dn, 0.05)
_, cost_ok = gradient_descent(y_dn, a_identity, sigma_w_dn, 0.05, 20)
_, cost_big = gradient_descent(y_dn, a_identity, sigma_w_dn, 0.05, 20,
                               alpha=4 * alpha_ok)

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.semilogy(range(len(cost_ok)), cost_ok, lw=1.6,
            label=f"alpha = {alpha_ok:.4e}  (eq. 11)")
ax.semilogy(range(len(cost_big)), [abs(c) for c in cost_big], lw=1.6,
            ls="--", color="crimson",
            label=f"alpha = {4*alpha_ok:.4e}  (4x eq. 11)")
ax.set_xlabel("iteration $n$")
ax.set_ylabel("$|c(x^{(n)})|$")
ax.set_title(r"Denoising, $\sigma_x = 0.05$: stable and unstable step sizes",
             fontsize=10)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_q4_stepsize.png", dpi=150, bbox_inches="tight")
plt.close(fig)

R["alpha_ok"] = alpha_ok
R["cost_ok_final"] = cost_ok[-1]
R["cost_big_final"] = cost_big[-1]
R["cost_big_history"] = cost_big[:8]

log("\n--- D10 question 4 ---")
log(f"alpha (eq 11)            {alpha_ok:.6e}")
log(f"c after 20 iters, alpha  {cost_ok[-1]:.6e}")
log(f"c after 20 iters, 4alpha {cost_big[-1]:.6e}")


# ---------------------------------------------------------------------------
# Write out
# ---------------------------------------------------------------------------

with open("results.txt", "w") as f:
    f.write("\n".join(log_lines) + "\n")
with open("results.json", "w") as f:
    json.dump(R, f, indent=2)

log("\nwrote results.txt, results.json, and figures/")
