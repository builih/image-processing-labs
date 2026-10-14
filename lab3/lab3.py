"""ECE 60141 Lab 3: linear forward models and MAP reconstruction.

Functions only.  Run experiments from run_lab3.py, which imports this module.
All arithmetic is float64.  No loop over pixels anywhere.
"""

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Readers (provided by the lab handout)
# ---------------------------------------------------------------------------

def load_pgm(path):
    """Read a binary PGM image file.

    path     path to a P5 PGM file
    returns  (H, W) float64 tensor with values in [0, 1]
    """
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5"
        width, height = map(int, f.readline().split())
        assert int(f.readline()) == 255
        raw = np.frombuffer(f.read(), dtype=np.uint8).reshape(height, width)
    return torch.tensor(raw / 255.0, dtype=torch.float64)


def load_kernel(path):
    """Read a blur kernel from a text file.

    path     path to a kernel file, one row of the kernel per line
    returns  (Ka, Kb) float64 tensor, nonnegative and summing to 1
    """
    return torch.tensor(np.loadtxt(path), dtype=torch.float64)


# ---------------------------------------------------------------------------
# Module 1: the forward model and its adjoint
# ---------------------------------------------------------------------------

# The identity operator, written as a 1 by 1 convolution kernel.
a_identity = torch.ones((1, 1), dtype=torch.float64)


def _circular_conv2d(x, w):
    """Correlate (H, W) tensor x with (Ka, Kb) tensor w, circular boundaries.

    Pads x circularly by the kernel half width, then calls conv2d, which
    computes correlation.  Output has the same shape as the input.

    Requires each image dimension to be at least the corresponding kernel
    half width, which F.pad's circular mode enforces.  The lab images are
    512 by 768 and the kernel is 19 by 19, so this is never binding.
    """
    ka, kb = w.shape
    pa, pb = ka // 2, kb // 2
    xp = F.pad(x[None, None], (pb, pb, pa, pa), mode="circular")
    return F.conv2d(xp, w[None, None])[0, 0]


def forward(x, a):
    """Apply the blur operator A of equation (2).

    x        (H, W) float64 tensor, the image
    a        (Ka, Kb) float64 tensor, the blur kernel, odd sized in both axes
    returns  (H, W) float64 tensor, A x
    """
    # conv2d correlates, so flip the kernel in both axes to get convolution.
    return _circular_conv2d(x, torch.flip(a, dims=(0, 1)))


def adjoint(v, a):
    """Apply the transpose operator A^t of equation (4).

    v        (H, W) float64 tensor
    a        (Ka, Kb) float64 tensor, the same kernel passed to forward
    returns  (H, W) float64 tensor, A^t v
    """
    # Equation (4) is correlation with a, which is what conv2d computes, so
    # the kernel is used unflipped.
    return _circular_conv2d(v, a)


def adjoint_autograd(v, a):
    """Apply A^t, obtained from forward() by automatic differentiation.

    v        (H, W) float64 tensor
    a        (Ka, Kb) float64 tensor, the same kernel passed to forward
    returns  (H, W) float64 tensor, A^t v
    """
    x = torch.zeros_like(v, requires_grad=True)
    return torch.autograd.grad((forward(x, a) * v).sum(), x)[0]


# ---------------------------------------------------------------------------
# Module 2: the GMRF prior
# ---------------------------------------------------------------------------

# The 3 by 3 array g of noncausal prediction coefficients, 8 point
# neighborhood, normalized so that the entries sum to one.
g_coeff = torch.tensor(
    [[1.0 / 12, 1.0 / 6, 1.0 / 12],
     [1.0 / 6, 0.0, 1.0 / 6],
     [1.0 / 12, 1.0 / 6, 1.0 / 12]],
    dtype=torch.float64,
)


def _neighbor_sum(x):
    """Return the (H, W) array whose entry s is sum_{r in ds} g_{s-r} x_r.

    One 3 by 3 convolution with zero padding.  g is symmetric, so correlation
    and convolution agree and no flip is needed.  Zero padding implements the
    convention that a pair {s, r} contributes only when both pixels lie inside
    the image.
    """
    return F.conv2d(x[None, None], g_coeff[None, None], padding=1)[0, 0]


def _neighbor_weight(x):
    """Return the (H, W) array c, where c_s = sum_{r in ds} g_{s-r}.

    The same 3 by 3 convolution applied to an all ones mask.  c_s is 1 in the
    interior and less along an edge or at a corner.
    """
    return _neighbor_sum(torch.ones_like(x))


def prior_cost(x, sigma_x):
    """Return the prior term (1/2) x^t B x of equation (7).

    x        (H, W) float64 tensor
    sigma_x  float
    returns  float64 scalar tensor
    """
    # (1/2) x^t B x = (1/2) sum_s x_s [B x]_s.
    return 0.5 * (x * prior_grad(x, sigma_x)).sum()


def prior_grad(x, sigma_x):
    """Return B x, the gradient of prior_cost, from equation (7).

    x        (H, W) float64 tensor
    sigma_x  float
    returns  (H, W) float64 tensor
    """
    return (_neighbor_weight(x) * x - _neighbor_sum(x)) / (sigma_x ** 2)


# ---------------------------------------------------------------------------
# Module 2: the MAP cost and gradient descent
# ---------------------------------------------------------------------------

def cost(x, y, a, sigma_w, sigma_x):
    """Return the MAP cost c(x) of equation (8).

    x, y     (H, W) float64 tensors
    a        (Ka, Kb) float64 tensor, the blur kernel
    returns  float64 scalar tensor
    """
    residual = y - forward(x, a)
    return (residual ** 2).sum() / (2 * sigma_w ** 2) + prior_cost(x, sigma_x)


def cost_grad(x, y, a, sigma_w, sigma_x):
    """Return the gradient of the MAP cost, equation (9).

    returns  (H, W) float64 tensor
    """
    residual = y - forward(x, a)
    return -adjoint(residual, a) / (sigma_w ** 2) + prior_grad(x, sigma_x)


def step_size(sigma_w, sigma_x):
    """Return the step size alpha of equation (11).

    returns  float
    """
    return 1.0 / (1.0 / sigma_w ** 2 + 2.0 / sigma_x ** 2)


def gradient_descent(y, a, sigma_w, sigma_x, num_iters, alpha=None):
    """Minimize the MAP cost by gradient descent, equation (10).

    Starts at x = y.  Uses the step size of equation (11) when alpha is None.

    returns  (x_hat, cost_history), an (H, W) tensor and a list of length
             num_iters + 1 whose first entry is c(y)
    """
    if alpha is None:
        alpha = step_size(sigma_w, sigma_x)
    x = y.clone()
    cost_history = [cost(x, y, a, sigma_w, sigma_x).item()]
    for _ in range(num_iters):
        x = x - alpha * cost_grad(x, y, a, sigma_w, sigma_x)
        cost_history.append(cost(x, y, a, sigma_w, sigma_x).item())
    return x, cost_history
