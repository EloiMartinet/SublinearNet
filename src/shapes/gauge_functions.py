import torch
from torch import nn


class LSEGauge(nn.Module):
    """
    Log-Sum-Exp (LSE) smoothed gauge function.

    This module represents a smooth approximation of the gauge
    (Minkowski functional) of a convex polygon or polytope.

    Given a convex set :math:`\\Omega \\subset \\mathbb{R}^d` containing the origin, its gauge is

    .. math::

        \\gamma_\\Omega(x) = \\inf \\{ t > 0 : x \\in t \\Omega \\}

    This implementation approximates the gauge using a
    log-sum-exp (soft maximum) of linear functionals, yielding
    a smooth, positively 1-homogeneous, convex function.

    The functional form is

    .. math::

        \\gamma(x) = \\|x\\| \\cdot \\frac{1}{\\beta} \\log \\sum_k \\exp\\Big(\\langle w_k, x / \\|x\\| \\rangle\\Big)

    where
    - :math:`\\{ w_k \\}` are learned directions
    - :math:`\\beta > 0` controls the sharpness of the approximation

    Parameters
    ----------
    input_size : int, optional
        Ambient dimension :math:`d`.
    n_unit : int, optional
        Number of linear directions (facets).
    beta : float, optional
        Initial inverse temperature for the log-sum-exp smoothing.
        Larger values give sharper (less smooth) approximations.
    """

    def __init__(self, input_size=2, n_unit=50, beta=10):
        super().__init__()

        # Linear map defining supporting hyperplanes
        # Bias is disabled to preserve 1-homogeneity
        self.input_layer = nn.Linear(input_size, n_unit, bias=False)

        # Initialize weights with relatively large variance
        # to encourage diverse facet orientations
        self.input_layer.weight = torch.nn.init.normal_(
            self.input_layer.weight,
            std=5.0,
        )

        # Learnable inverse temperature controlling smoothness
        self.beta = nn.Parameter(beta * torch.rand(1))

        # Smooth approximation of ReLU used for numerical stability
        self.softplus = nn.Softplus()

    def forward(self, x):
        """
        Evaluate the smoothed gauge function.

        Parameters
        ----------
        x : torch.Tensor, shape (B, d)
            Input vectors.

        Returns
        -------
        torch.Tensor, shape (B, 1)
        """
        # Euclidean norm ||x||
        x_norm = torch.norm(x, dim=-1, keepdim=True)

        # Normalize direction (avoid division by zero)
        direction = x / (x_norm + 1e-12)

        # Evaluate supporting hyperplanes ⟨w_k, x / ||x||⟩
        out = self.input_layer(direction)

        # Log-sum-exp aggregation (smooth max)
        # Softplus ensures beta stays positive
        beta_eff = self.softplus(self.beta)
        out = (1.0 / beta_eff) * torch.logsumexp(out, dim=-1, keepdim=True)

        # Restore 1-homogeneity
        out = out * x_norm

        return out
