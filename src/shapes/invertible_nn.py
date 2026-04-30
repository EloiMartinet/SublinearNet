import torch
from torch import nn
import math

import shapes.gauge_functions as gf
from fiblat import sphere_lattice


class ConvexDiffeo(nn.Module):
    """
    Neural parametrization of a convex diffeomorphism.

    This class represents a smooth diffeomorphism

    .. math::

        \\varphi : B(0, 1) \\subset \\mathbb{R}^d \\longrightarrow \\Omega \\subset \\mathbb{R}^d

    mapping the unit ball onto a convex domain :math:`\\Omega`.
    The map is defined via a learned convex gauge function :math:`G`:

    .. math::

        \\varphi(x) = \\frac{x \\, \\|x\\|}{G(x)}

    Optionally, symmetry constraints can be enforced by averaging
    the gauge over a finite symmetry group.

    Parameters
    ----------
    mesh_length : optional
        Reserved for future discretization / meshing purposes.
    input_size : int
        Ambient dimension :math:`d`.
    n_unit : int
        Number of hidden units in the convex gauge network.
    symmetries : iterable of callables, optional
        Group actions :math:`g(x)` enforcing symmetry through averaging.
    """

    def __init__(self, input_size=2, n_unit=50, mode='gauge', symmetries=None, gauge_function='LSE'):
        super().__init__()

        self.dim = input_size
        self.n_unit = n_unit
        self.symmetries = symmetries
        self.mode = mode

        # Learned convex gauge function G(x)
        if gauge_function == 'LSE':
            self.sublinear_nn = gf.LSEGauge(
                input_size=input_size,
                n_unit=n_unit
            )
        elif gauge_function == 'cube':
            self.sublinear_nn = gf.CubeGauge()
        elif gauge_function == 'ball':
            self.sublinear_nn = gf.BallGauge()
        elif gauge_function == 'octahedron':
            self.sublinear_nn = gf.OctahedronGauge()
        else:
            print(f'Unsupported gauge function \"{gauge_function}\"')
        
    # ------------------------------------------------------------------
    # Core map
    # ------------------------------------------------------------------

    def forward_gauge(self, x, eps=1e-12):
        """
        Evaluate the diffeomorphism :math:`\\varphi(x)` given by the gauge function.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, d)``. Points in the unit ball.
        eps : float, optional
            Small stabilization constant to avoid division by zero.

        Returns
        -------
        torch.Tensor
            Shape ``(B, d)``. Points in the target convex domain :math:`\\Omega`.
        """
        norm = torch.norm(x, dim=-1, keepdim=True)

        if self.symmetries is None:
            gauge = self.sublinear_nn(x)
        else:
            # Group-averaged gauge to enforce symmetry
            gauge = 0.0
            for g in self.symmetries:
                gauge += self.sublinear_nn(g(x))

        return x * (norm + eps) / (gauge + eps)

    def forward_support(self, x):
        """
        Evaluate the diffeomorphism :math:`\\varphi(x)` given by the gauge function

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, d)``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(B, d)`` containing the support-gradient evaluated at
            :math:`x`, scaled by :math:`\\|x\\|`.
        """
        norm = torch.norm(x, dim=-1, keepdim=True)

        def support_single(xi):
            return self.sublinear_nn(xi.unsqueeze(0)).squeeze(0)

        grad_fn = torch.func.vmap(torch.func.jacrev(support_single))

        if self.symmetries is None:
            grad_support = grad_fn(x).squeeze()
            return grad_support * norm

        summed = 0.0

        for g in self.symmetries:
            xg = g(x)
            grad = grad_fn(xg).squeeze()

            # IMPORTANT: no autograd, no jacobian, no requires_grad
            # just pull back symmetry

            R = g.keywords["R"]   # works if you used functools.partial
            grad = grad @ R       # correct pullback for rotation

            summed = summed + grad

        return summed * norm

    def forward(self, x, eps=1e-12):
        """
        Dispatch forward computation based on the selected mode.

        This method routes the input to either the gauge or support computation,
        depending on the value of ``self.mode``.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, d)``.
        eps : float, optional
            Small numerical constant used for stability in gauge computations.
            Default is ``1e-12``. Ignored when ``mode='support'``.

        Returns
        -------
        torch.Tensor
            Output of the selected forward computation. Shape depends on the mode:
            
            - ``'gauge'``: typically ``(B,)`` or ``(B, 1)``, depending on implementation.
            - ``'support'``: ``(B, d)``.
        """
        if self.mode == 'gauge':
            return self.forward_gauge(x, eps)
        elif self.mode == 'support':
            return self.forward_support(x)
        else:
            print('Unknown mode')
    
    # ------------------------------------------------------------------
    # Inverse map
    # ------------------------------------------------------------------
    def inverse_gauge(self, x, eps=1e-12):
        """
        Evaluate the inverse of the diffeomorphism :math:`\\varphi(x)` when :math:`\\varphi(x)` is given by the gauge function.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, d)``. Points in the unit ball.
        eps : float, optional
            Small stabilization constant to avoid division by zero.

        Returns
        -------
        torch.Tensor
            Shape ``(B, d)``. Points in the target convex domain :math:`\\Omega`.
        """

        norm = torch.norm(x, dim=-1, keepdim=True)

        if self.symmetries is None:
            gauge = self.sublinear_nn(x)
        else:
            # Group-averaged gauge to enforce symmetry
            gauge = 0.0
            for g in self.symmetries:
                gauge += self.sublinear_nn(g(x))

        return x * (gauge + eps) / (norm + eps)

    def inverse(self, x, eps=1e-12):
        """
        Dispatch inverse computation based on the selected mode.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, d)``.
        eps : float, optional
            Small numerical constant used for stability in gauge-based inversion.
            Default is ``1e-12``.

        Returns
        -------
        torch.Tensor or None
            Output of the inverse computation:

            - ``'gauge'``: Tensor of shape ``(B, d)`` returned by
            :meth:`inverse_gauge`.
            - ``'support'``: Not implemented; returns ``None``.
            - otherwise: Returns ``None``.
        """

        if self.mode == 'gauge':
            return self.inverse_gauge(x, eps)
        elif self.mode == 'support':
            print('Inverse support mode is not supported')
        else:
            print('Unknown mode')

    # ------------------------------------------------------------------
    # Alternative representations
    # ------------------------------------------------------------------
    def level_set(self, x):
        """
        Return a level set function representing the model.
        Warning: this is not the signed distance function.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, d)``. Points in the input space.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 1)``.
        """
        return self.sublinear_nn(x) - 1

    # ------------------------------------------------------------------
    # Sampling utilities
    # ------------------------------------------------------------------

    def sample_ball(self, n_points=50000, requires_grad=False, random=False):
        """
        Sample points uniformly inside the unit ball B(0,1).

        Sampling is performed by rejection from the hypercube [-1,1]^d.

        Parameters
        ----------
        n_points : int
            Approximate number of accepted samples.
        requires_grad : bool
            Whether returned points require gradients.
        random : bool
            If True, sample randomly; otherwise use a Cartesian grid.

        Returns
        -------
        torch.Tensor, shape (N, d)
            Points inside the unit ball.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Volume of unit ball in dimension d
        omega = math.pi ** (self.dim / 2) / math.gamma(self.dim / 2 + 1)

        # Oversample to compensate for rejection
        n_total = int(math.floor((n_points * 4 / omega)))

        if random:
            x = 2 * torch.rand((n_total, self.dim),
                               device=device, dtype=dtype) - 1
        else:
            n_side = int(n_total ** (1 / self.dim))
            lin = torch.linspace(-1, 1, n_side, device=device, dtype=dtype)
            grids = torch.meshgrid(*([lin] * self.dim), indexing="ij")
            x = torch.stack([g.flatten() for g in grids], dim=-1)

        mask = torch.linalg.norm(x, dim=1) <= 1
        x = x[mask].detach()

        if requires_grad:
            x.requires_grad_(True)

        return x

    def sample_sphere(self, n_points=50000, seed=1073, random=False, requires_grad=False):
        """
        Sample points uniformly on the unit sphere S^{d-1}.

        Sampling strategy depends on the dimension:
        - 2D: exact angular parametrization
        - 3D: Fibonacci lattice
        - d>3: projection from the unit ball

        Parameters
        ----------
        n_points : int
            Number of sampled points.
        requires_grad : bool
            Whether returned points require gradients.

        Returns
        -------
        torch.Tensor, shape (N, d)
            Points on the unit sphere.
        """
        device = next(self.buffers(), torch.empty(0)).device
        dtype = next(self.buffers(), torch.tensor(0.0)).dtype

        if random:
            x = torch.randn(n_points, self.dim)
            x /= torch.norm(x, dim=1, keepdim=True)
        else:
            if self.dim == 2:
                theta = torch.linspace(
                    0, 2 * math.pi, n_points + 1,
                    device=device, dtype=dtype
                )[:-1]
                x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)

            elif self.dim == 3:
                sphere = sphere_lattice(3, n_points)
                x = torch.tensor(sphere, device=device, dtype=dtype)

            else:
                # Sample deterministically in order to not mess with bfgs
                g = torch.Generator()
                g.manual_seed(42)
                x = torch.randn(n_points, self.dim, generator=g)
                x /= torch.norm(x, dim=1, keepdim=True)

        x = x.detach()
        if requires_grad:
            x.requires_grad_(True)

        return x

    # ------------------------------------------------------------------
    # Differential geometry
    # ------------------------------------------------------------------

    def jacobian(self, x):
        """
        Compute the Jacobian matrix :math:`D\\varphi(x)`.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(B, d)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, d, d)``. Jacobian matrices evaluated at :math:`x`.
        """
        def f_single(xi):
            return self.forward(xi.unsqueeze(0)).squeeze(0)

        return torch.func.vmap(torch.func.jacrev(f_single))(x)

    def jacobian_cofactor_matrix(self, x):
        """
        Compute the cofactor (adjugate) matrix of the Jacobian.

        This matrix naturally appears in surface measure
        transformations under :math:`\\varphi`.

        Returns
        -------
        torch.Tensor, shape (B, d, d)
        """
        jac = self.jacobian(x)
        det = torch.linalg.det(jac)[:, None, None]
        inv_jac = torch.linalg.inv(jac)

        return det * inv_jac.transpose(1, 2)

    def jacobian_regularizer(self, n_points=500):
        """
        Regularizer preventing Jacobian degeneracy.

        Returns the maximum condition number of :math:`D\\varphi(x)`
        over sampled points on the sphere.

        Returns
        -------
        torch.Tensor (scalar)
        """
        x = self.sample_sphere(n_points)
        jac = self.jacobian(x)
        return torch.linalg.cond(jac).max()

    # ------------------------------------------------------------------
    # Integral geometric quantities
    # ------------------------------------------------------------------

    def volume(self, n_points=50000):
        """
        Estimate of the volume :math:`|\\Omega|`.

        Uses the change-of-variables formula:

        .. math::

            |\\Omega| = \\int_{B} \\left| \\det D\\varphi(x) \\right| \\, \\mathrm{d}x

        Returns
        -------
        torch.Tensor
            Shape ``(1,)``.
        """
        x = self.sample_ball(n_points)
        det = torch.abs(torch.linalg.det(self.jacobian(x)))

        omega = math.pi ** (self.dim / 2) / math.gamma(self.dim / 2 + 1)
        return (omega * det.mean()).unsqueeze(0)

    def perimeter(self, n_points=50000):
        """
        Estimate of the surface area :math:`|\\partial \\Omega|`.

        Returns
        -------
        torch.Tensor, shape (1,)
        """
        x = self.sample_sphere(n_points, requires_grad=True)
        normals = x.detach().requires_grad_(True)

        comatrix = self.jacobian_cofactor_matrix(x)
        product = torch.matmul(comatrix, normals.unsqueeze(2))

        sigma = self.dim * math.pi ** (self.dim / 2) / math.gamma(self.dim / 2 + 1)
        return (torch.norm(product, dim=1).mean() * sigma).unsqueeze(0)

    def center_of_gravity(self, n_points=50000):
        """
        Compute the center of gravity of :math:`\\Omega`.

        Returns
        -------
        torch.Tensor, shape (1, d)
        """
        x = self.sample_ball(n_points)
        y = self.forward(x)

        det = torch.abs(torch.linalg.det(self.jacobian(x))).unsqueeze(-1)
        cog = torch.mean(y * det, dim=0, keepdim=True)
        cog /= torch.mean(det, dim=0, keepdim=True)

        return cog

    def moment_of_inertia(self, n_points=50000):
        """
        Compute the scalar moment of inertia of :math:`\\Omega`
        with respect to its center of gravity.

        Returns
        -------
        torch.Tensor, shape (1,)
        """
        x = self.sample_ball(n_points)
        cog = self.center_of_gravity(n_points)
        y = self.forward(x)

        integrand = torch.norm(y - cog, dim=-1) ** 2
        det = torch.abs(torch.linalg.det(self.jacobian(x)))

        omega = math.pi ** (self.dim / 2) / math.gamma(self.dim / 2 + 1)
        return (omega * torch.mean(integrand * det)).unsqueeze(0)

    def integrate_interior(self, f, n_points=50000):
        """
        Integrate :math:`f` on :math:`\\Omega` 

        Uses the change-of-variables formula:

        .. math::

            \\int_{\\Omega} f \\, \\mathrm{d}x = \\int_{B} (f \\circ \\varphi)(x) \\left| \\det D\\varphi(x) \\right| \\, \\mathrm{d}x

        Returns
        -------
        torch.Tensor
            Shape ``(1,)``.
        """
        x = self.sample_ball(n_points)
        y = self.forward(x)
        det = torch.abs(torch.linalg.det(self.jacobian(x)))

        omega = math.pi ** (self.dim / 2) / math.gamma(self.dim / 2 + 1)
        integrand = f(y) * det

        return (omega * integrand.mean()).unsqueeze(0)
    
    def mahler_volume(self, n_points=50000):
        """
        Estimate the Mahler volume of the convex body.

        The Mahler volume is defined as the product of the volume of a convex
        body :math:`\\Omega` and the volume of its polar body :math:`\\Omega^\\circ`.

        Parameters
        ----------
        n_points : int, optional
            Number of sample points used for Monte Carlo volume estimation.
            Default is ``50000``.

        Returns
        -------
        float
            Estimated Mahler volume :math:`\\mathrm{vol}(\\Omega) \\cdot \\mathrm{vol}(\\Omega^\\circ)`.
        """
        # Save the default mode
        mode = self.mode 

        # Compute the volumes of the convex and its polar body
        self.mode = 'gauge'
        vol = self.volume(n_points)
        self.mode = 'support'
        vol_polar = self.volume(n_points)

        # Reset the mode
        self.mode = mode

        return vol*vol_polar

    # ------------------------------------------------------------------
    # Surface differential geometry
    # ------------------------------------------------------------------

    def normal(self, x):
        """
        Compute the outward unit normal at :math:`\\varphi(x)`,
        where :math:`x` lies on the unit sphere.

        Parameters
        ----------
        x : torch.Tensor, shape (B, d)

        Returns
        -------
        torch.Tensor, shape (B, d)
        """
        n_ref = x / torch.norm(x, dim=1, keepdim=True)
        jac = self.jacobian(x)

        inv_jac_T = torch.linalg.inv(jac).transpose(1, 2)
        n = torch.einsum("bij,bj->bi", inv_jac_T, n_ref)
        n = n/torch.norm(n, dim=1, keepdim=True)

        return n

    def mean_curvature(self, x):
        """
        Compute the mean curvature :math:`H` at :math:`\\varphi(x)`.

        Parameters
        ----------
        x : torch.Tensor, shape (B, d)
            Points on the unit sphere.

        Returns
        -------
        torch.Tensor, shape (B,)
        """
        B, d = x.shape

        n = self.normal(x)

        def n_single(x_single):
            return self.normal(x_single.unsqueeze(0)).squeeze(0)

        dndx = torch.func.vmap(torch.func.jacrev(n_single))(x)

        jac = self.jacobian(x)
        inv_jac = torch.linalg.inv(jac)
        dndy = torch.einsum('bij,bjk->bik', dndx, inv_jac)

        eye = torch.eye(d, dtype=x.dtype, device=x.device).expand(B, d, d)
        P = eye - n.unsqueeze(2) * n.unsqueeze(1)

        Hd = torch.einsum('bij,bji->b', P, dndy)/(self.dim-1)

        return Hd

    def gaussian_curvature(self, x):
        """
        Gaussian curvature via Householder-based tangent restriction.
        Fully vectorized, no QR, no eigenvalues.

        Parameters
        ----------
        x : torch.Tensor, shape (B, d)

        Returns
        -------
        torch.Tensor, shape (B,)
        """
        B, d = x.shape
        device, dtype = x.device, x.dtype

        n = self.normal(x)  # (B, d)

        def n_single(x_single):
            return self.normal(x_single.unsqueeze(0)).squeeze(0)

        # dn/dx
        dndx = torch.func.vmap(torch.func.jacrev(n_single))(x)  # (B, d, d)

        # dn/dy
        jac = self.jacobian(x)                                  # (B, d, d)
        inv_jac = torch.linalg.inv(jac)
        S = torch.einsum('bij,bjk->bik', dndx, inv_jac)         # (B, d, d)

        # --- Householder construction (vectorized) ---

        # target basis vector e_d
        e = torch.zeros(B, d, device=device, dtype=dtype)
        e[:, -1] = 1.0

        # v = n - e (Householder direction)
        v = n - e                                               # (B, d)

        # handle edge case: n ≈ e → avoid division by 0
        v_norm = torch.norm(v, dim=1, keepdim=True)             # (B, 1)
        v = v / (v_norm + 1e-12)

        # Householder matrices: H = I - 2 v v^T
        eye = torch.eye(d, device=device, dtype=dtype).expand(B, d, d)
        H = eye - 2.0 * v.unsqueeze(2) * v.unsqueeze(1)         # (B, d, d)

        # Tangent basis = first (d-1) columns
        T = H[:, :, :-1]                                        # (B, d, d-1)

        # Restrict shape operator
        S_tan = torch.einsum('bik,bkl,blj->bij',
                            T.transpose(1, 2), S, T)           # (B, d-1, d-1)

        # Determinant
        K = torch.linalg.det(S_tan)                             # (B,)

        return K
        
    def integral_mean_curvature(self, n_points=50000):
        """
        Compute :math:`\\int_{\\partial \\Omega} H dS` via a change of variable

        Returns
        -------
        torch.Tensor
            Shape ``(1,)``.
        """
        B = self.dim
        x = self.sample_sphere(n_points, requires_grad=True)

        H = self.mean_curvature(x)
        comatrix = self.jacobian_cofactor_matrix(x)

        n_unit = x / torch.norm(x, dim=1, keepdim=True)
        dS = torch.norm(torch.matmul(comatrix, n_unit.unsqueeze(2)).squeeze(2), dim=1)

        sigma = B * math.pi ** (B / 2) / math.gamma(B / 2 + 1)
        return ((H * dS).mean() * sigma).unsqueeze(0)

    def willmore_energy(self, n_points=50000):
        """
        Compute the Willmore energy :math:`\\int_{\\partial \\Omega} H^2 dS`

        Returns
        -------
        torch.Tensor, shape (1,)
        """
        B = self.dim
        x = self.sample_sphere(n_points, requires_grad=True)

        H = self.mean_curvature(x)
        comatrix = self.jacobian_cofactor_matrix(x)

        n_unit = x / torch.norm(x, dim=1, keepdim=True)
        dS = torch.norm(torch.matmul(comatrix, n_unit.unsqueeze(2)).squeeze(2), dim=1)

        sigma = B * math.pi ** (B / 2) / math.gamma(B / 2 + 1)
        return ((H ** 2 * dS).mean() * sigma).unsqueeze(0)


    # ------------------------------------------------------------------
    # Potential theory: Green functions
    # ------------------------------------------------------------------

    def psi(self, x, y):
        """
        Fundamental solution of the Laplacian.

        Constructs the matrix

        .. math::

            \\left[ \\varphi(x_i - y_j) \\right]_{i,j}

        where :math:`\\varphi` is the fundamental solution of the Laplacian in :math:`\\mathbb{R}^d`:

        .. math::

            \\varphi(r) =
            \\begin{cases}
            -\\log |r|, & d = 2 \\\\
            |r|^{2-d}, & d \\ge 3
            \\end{cases}

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N_x, d)``. Evaluation points.
        y : torch.Tensor
            Shape ``(N_y, d)``. Source points.

        Returns
        -------
        torch.Tensor
            Shape ``(N_x, N_y)``. Pairwise evaluations of :math:`\\varphi(x_i - y_j)`.
        """
        dim = x.shape[1]

        x_expanded = x.unsqueeze(1)   # (N_x, 1, d)
        y_expanded = y.unsqueeze(0)   # (1, N_y, d)
        diff = x_expanded - y_expanded

        r = torch.linalg.norm(diff, dim=2)

        if dim == 2:
            return -torch.log(r)
        else:
            return 1.0 / r ** (dim - 2)

    def grad_psi(self, x, y):
        """
        Gradient of the fundamental solution with respect to :math:`x`.

        Computes

        .. math::

            \\nabla_x \\varphi(x_i - y_j)

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(N_x, d)``.
        y : torch.Tensor
            Shape ``(N_y, d)``.

        Returns
        -------
        torch.Tensor
            Shape ``(N_x, N_y, d)``.
        """
        dim = x.shape[1]

        diff = x.unsqueeze(1) - y.unsqueeze(0)
        r = torch.linalg.norm(diff, dim=2, keepdim=True)

        if dim == 2:
            grad_phi = -diff / (r ** 2)
        else:
            grad_phi = -(dim - 2) * diff / (r ** dim)

        return grad_phi

    # ------------------------------------------------------------------
    # Linear algebra utilities
    # ------------------------------------------------------------------

    def gelsd_like_lstsq(self, A, b, rcond=1e-15):
        """
        Solve a least-squares problem using SVD (GELSD-like).

        This function mimics the behavior of LAPACK's GELSD
        while remaining compatible with CUDA.

        Solves the problem

        .. math::

            \\min \\|A x - b\\|_2

        and returns the minimum-norm solution.

        Parameters
        ----------
        A : torch.Tensor
            Shape ``(M, N)``.
        b : torch.Tensor
            Shape ``(M,)`` or ``(M, K)``.
        rcond : float
            Relative cutoff for small singular values.

        Returns
        -------
        torch.Tensor
            Shape ``(N,)`` or ``(N, K)``.
        """
        if b.ndim == 1:
            b = b[:, None]

        U, S, Vh = torch.linalg.svd(A, full_matrices=False)

        tol = rcond * S.max()
        mask = S > tol

        S_inv = torch.zeros_like(S)
        S_inv[mask] = 1.0 / S[mask]

        x = Vh.mT @ (S_inv[:, None] * (U.mT @ b))
        return x.squeeze()

    # ------------------------------------------------------------------
    # Torsional rigidity (Poisson problem)
    # ------------------------------------------------------------------

    def torsional_rigidity(
        self,
        n_points=10_000,
        tol=1e-4,
        n_sources_grid=None,
        offset_grid=None,
    ):
        """
        Compute torsional rigidity using adaptive parameter search.

        A grid search is performed over:
        - number of sources
        - offset distance outside the domain

        until the relative boundary error is below `tol`.

        Parameters
        ----------
        n_points : int
            Number of quadrature points.
        tol : float
            Target relative boundary error.
        n_sources_grid : list[int], optional
            Candidate numbers of sources.
        offset_grid : list[float], optional
            Candidate source offsets.

        Returns
        -------
        torch.Tensor, shape (1,)
            Estimated torsional rigidity.
        """
        if n_sources_grid is None:
            n_sources_grid = [100, 200, 400, 800, 1600]
        if offset_grid is None:
            offset_grid = [0.5, 1.0]

        best_val = None
        best_err = float("inf")

        for offset in offset_grid:
            for n_sources in n_sources_grid:
                torsion, rel_err, _, _, _ = self.torsional_rigidity_(
                    n_points, n_sources, offset
                )
                if rel_err < best_err:
                    best_err = rel_err
                    best_val = torsion

                if rel_err <= tol:
                    return torsion

        return best_val

    def torsional_rigidity_(self, n_points=2000, n_sources=300, offset=0.1):
        """
        Compute torsional rigidity for fixed parameters.

        Solves the Poisson problem

        .. math::

            \\begin{cases}
            -\\Delta u = 1 & \\text{in } \\Omega, \\\\
            u = 0 & \\text{on } \\partial\\Omega
            \\end{cases}

        using a boundary collocation method with Green functions.

        Returns
        -------
        torsion : torch.Tensor
            Shape ``(1,)``.
        rel_err : torch.Tensor
            Relative boundary error.
        """
        # Source points outside Ω
        sphere = self.sample_sphere(n_sources)
        sources = self.forward(sphere * (1 + offset))

        # Collocation points on ∂Ω
        sphere = self.sample_sphere(5 * n_sources)
        colloc = self.forward(sphere)

        M = self.psi(colloc, sources)
        b = 0.5 * colloc[:, 0] ** 2

        coefs = self.gelsd_like_lstsq(M, b, rcond=1e-12)

        # Boundary error estimation
        with torch.no_grad():
            sphere = self.sample_sphere(10 * n_sources)
            colloc_test = self.forward(sphere)

            M_test = self.psi(colloc_test, sources)
            phi_test = M_test @ coefs
            b_test = 0.5 * colloc_test[:, 0] ** 2

            rel_err = (
                torch.max(torch.abs(phi_test - b_test))
                / torch.max(torch.abs(b_test))
            )

        # Volume integration
        x = self.sample_ball(n_points)
        y = self.forward(x)

        u = self.psi(y, sources) @ coefs - 0.5 * y[:, 0] ** 2
        det = torch.abs(torch.linalg.det(self.jacobian(x)))

        omega = math.pi ** (self.dim / 2) / math.gamma(self.dim / 2 + 1)
        torsion = omega * torch.mean(u * det)

        return torsion.unsqueeze(0), rel_err, coefs, colloc, sources
    
    def grad_norm_infinity_boundary(self,
        n_points=20000,
        tol=2e-4,
        n_sources_grid=None,
        offset_grid=None,
    ):
        """
        This method solves the torsion boundary problem, reconstructs the solution
        :math:`u`, evaluates its gradient analytically at a boundary point, and
        returns the maximum gradient magnitude.

        Parameters
        ----------
        n_points : int, optional
            Number of sample points used in the torsion solver. Default is ``20000``.
        tol : float, optional
            Target relative error tolerance for the torsion solver. Default is ``2e-4``.
        n_sources_grid : list of int, optional
            Grid of source counts used in the line search. If ``None``, defaults to
            ``[200, 400, 800, 1600]``.
        offset_grid : list of float, optional
            Grid of source offsets used in the line search. If ``None``, defaults to
            ``[0.5, 1.0]``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(1,)`` containing the estimated value of
            :math:`\\|\\nabla u\\|_{\\infty, \\partial \\Omega}`.

        Notes
        -----
        - A grid search over ``n_sources_grid`` and ``offset_grid`` is performed to
        identify coefficients minimizing the relative error of the torsion solver.
        """

        #
        # --- Step 1: Compute torsion coefficients (line search for a given precision)
        #

        if n_sources_grid is None:
            n_sources_grid = [200, 400, 800, 1600]
        if offset_grid is None:
            offset_grid = [0.5, 1.0]

        best_coefs = None
        best_err = float("inf")

        for offset in offset_grid:
            for n_sources in n_sources_grid:
                _, rel_err, coefs, colloc, sources = self.torsional_rigidity_(
                    n_points, n_sources, offset
                )
                
                if rel_err < best_err:
                    best_err = rel_err
                    best_coefs = coefs

                if rel_err <= tol:
                    break

            if rel_err <= tol:
                break

        if best_coefs is None:
            best_coefs = coefs
        
        #
        # --- Step 2: Compute ∇u analytically at the boundary
        #

        # Evaluate at a single boundary point
        sphere_eval = torch.zeros(1, self.dim, device=colloc.device, dtype=colloc.dtype)
        sphere_eval[0,-1] = 1
        colloc_eval = self.forward(sphere_eval)

        # Compute analytic gradient of the kernel
        grad_phi = self.grad_psi(colloc_eval, sources)  # shape (n_eval, n_sources, dim)

        # Contract with coefficients to get ∇u
        grad_u = torch.einsum("ij,ijd->id", best_coefs.unsqueeze(0).expand(colloc_eval.size(0), -1), grad_phi)

        # Subtract ∇(½ x₁²) = (x₁, 0, ...)
        grad_u[:, 0] -= colloc_eval[:, 0]

        # Compute ‖∇u‖ and its max
        grad_norm = torch.linalg.norm(grad_u, dim=1)
        grad_inf = torch.max(grad_norm)

        return grad_inf.unsqueeze(0)

    
    # ------------------------------------------------------------------
    # Radial basis functions (RBF)
    # ------------------------------------------------------------------

    def _poly_basis(self, x):
        """
        Linear polynomial basis for thin-plate splines in 2D.

        Parameters
        ----------
        x : torch.Tensor, shape (N, 2)

        Returns
        -------
        torch.Tensor, shape (N, 3)
            [1, x, y]
        """
        return torch.stack(
            [torch.ones(x.shape[0], device=x.device), x[:, 0], x[:, 1],  x[:, 0]**2, x[:, 0]*x[:, 1], x[:, 1]**2],
            dim=1,
        )

    def rbf(self, x, y, eps=1.0, rbf="thin_plate"):
        """
        Evaluate radial basis functions :math:`\\varphi(|x - y|)`.

        Supports several kernels:
        - Gaussian
        - Multiquadric
        - Inverse multiquadric
        - Thin-plate spline
        - Wendland C²

        Returns
        -------
        torch.Tensor
            RBF matrix.
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        r = torch.linalg.norm(diff, dim=2)

        if rbf == "gaussian":
            Phi = torch.exp(-(r ** 2) / eps ** 2)
        elif rbf == "multiquadric":
            Phi = torch.sqrt(r ** 2 + eps ** 2)
        elif rbf == "inverse_multiquadric":
            Phi = 1.0 / torch.sqrt(r ** 2 + eps ** 2)
        elif rbf == "thin_plate":
            r_safe = r + 1e-12
            Phi = r ** 2 * torch.log(r_safe)
            P = self._poly_basis(x)
            Phi = torch.cat([Phi, P], dim=1)
        elif rbf == "wendland_c2":
            Phi = torch.zeros_like(r)
            mask = r <= eps
            s = r[mask] / eps
            Phi[mask] = (1 - s) ** 4 * (4 * s + 1)
        else:
            raise NotImplementedError

        return Phi

    def grad_rbf_x(self, x, y, eps=1.0, rbf="thin_plate"):
        """
        Gradient of RBF with respect to x.

        Returns
        -------
        torch.Tensor, shape (N_eval, N_src(+3), d)
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        r2 = torch.sum(diff ** 2, dim=2, keepdim=True)
        r = torch.sqrt(r2 + 1e-12)

        if rbf == "gaussian":
            phi = torch.exp(-r2 / eps ** 2)
            grad = -2 * diff * phi / eps ** 2
        elif rbf == "multiquadric":
            grad = diff / torch.sqrt(r2 + eps ** 2)
        elif rbf == "inverse_multiquadric":
            grad = -diff / (r2 + eps ** 2) ** (3 / 2)
        elif rbf == "thin_plate":
            grad = (2 * torch.log(r) + 1) * diff
            P_grad = torch.zeros((x.shape[0], 6, x.shape[1]), device=x.device)
            
            # Derivatives of the linear terms
            P_grad[:, 1, 0] = 1
            P_grad[:, 2, 1] = 1

            # Derivatives of the quadratic terms
            P_grad[:, 3, 0] = 2*x[:,0]
            P_grad[:, 3, 1] = 0
            P_grad[:, 4, 0] = x[:,1]
            P_grad[:, 4, 1] = x[:,0]
            P_grad[:, 5, 0] = 0
            P_grad[:, 5, 1] = 2*x[:,1]


            grad = torch.cat([grad, P_grad], dim=1)
        elif rbf == "wendland_c2":
            grad = torch.zeros_like(diff)
            s = r / eps
            mask = s <= 1
            factor = torch.zeros_like(r)
            factor[mask] = (-20 * s + 16 * s ** 2)[mask] / eps
            grad = factor * diff / (r + 1e-12)
        else:
            raise NotImplementedError

        return grad

    # ------------------------------------------------------------------
    # Spectral problems
    # ------------------------------------------------------------------

    def generalized_sym_eig(self, A, M, k=None, eps_jitter=1e-12, return_eigvecs=True):
        """
        Solve the generalized symmetric eigenproblem :math:`Ax = \\lambda M x` using Cholesky reduction.

        Eigenvectors are normalized in the M-inner product.

        Returns
        -------
        eigvals : torch.Tensor
        eigvecs : torch.Tensor
        """
        N = A.shape[0]

        jitter = eps_jitter * max(1.0, torch.trace(M).item())
        M = M + jitter * torch.eye(N, device=M.device, dtype=M.dtype)

        L = torch.linalg.cholesky(M)

        X = torch.linalg.solve_triangular(L, A, upper=False)
        C = torch.linalg.solve_triangular(L, X.T, upper=False).T
        C = 0.5 * (C + C.T)

        w, v = torch.linalg.eigh(C)

        if k is not None:
            w = w[:k]
            v = v[:, :k]

        if not return_eigvecs:
            return w

        c = torch.linalg.solve_triangular(L.T, v, upper=True)
        Mc = M @ c
        norms = torch.sqrt(torch.sum(c * Mc, dim=0)).clamp_min(1e-30)
        c = c / norms

        return w, c

    def evaluate_rbf_function(self, x_eval, sources, coeffs, eps=1.0, rbf="thin_plate"):
        """
        Evaluate an RBF expansion

        .. math::

            u(x) = \\sum_j c_j \\varphi(x - y_j)

        Returns
        -------
        torch.Tensor
            Shape ``(N_eval,)``.
        """
        Phi = self.rbf(x_eval, sources, eps=eps, rbf=rbf)
        return Phi @ coeffs

    def neumann_eigenvalues(
        self,
        n_ev=10,
        n_sources=400,
        n_quad_points=100000,
        eps=1.0,
        rbf="thin_plate",
        normalize=False,
    ):
        """
        Compute Neumann Laplacian eigenvalues on :math:`\\Omega`.

        Returns
        -------
        eig_vals : torch.Tensor
        eig_vecs : torch.Tensor
        sources : torch.Tensor
        """
        # Sample the sources
        sources = self.sample_ball(n_points=n_sources)

        # Get the quadrature points
        quad_all = self.sample_ball(n_points=n_quad_points)
        Nq = quad_all.shape[0]

        # Assemble the mass and stiffness matrices
        jac = self.jacobian(quad_all)
        rho = torch.linalg.det(jac)

        inv_jac = torch.linalg.inv(jac)
        A = rho[:, None, None] * (
            inv_jac @ inv_jac.transpose(1, 2)
        )

        phi = self.rbf(quad_all, sources, eps=eps, rbf=rbf)
        grad_phi = self.grad_rbf_x(quad_all, sources, eps=eps, rbf=rbf)

        M = torch.einsum("bi,bj,b->ij", phi, phi, rho)

        tmp = torch.einsum("bak,bik->bia", A, grad_phi)
        K = torch.einsum("bia,bja->ij", tmp, grad_phi)

        omega = torch.pi**(self.dim/2) / math.gamma(self.dim/2 + 1)
        scale = omega / Nq
        M *= scale
        K *= scale

        # Solve eigenproblem
        eig_vals, eig_vecs = self.generalized_sym_eig(K, M, k=n_ev)

        # Volume-normalize if requested
        if normalize:
            vol = omega * torch.mean(torch.abs(torch.linalg.det(self.jacobian(quad_all))))
            eig_vals = vol ** (2 / self.dim) * eig_vals

        return eig_vals, eig_vecs, sources


    def dirichlet_eigenvalues(
        self,
        n_ev=10,
        n_sources=200,
        n_quad_points=100000,
        n_quad_points_bd=200,
        eps=1.0,
        rbf='thin_plate',
        normalize=False,
        bd_penalization=1e5
    ):
        """
        Compute Dirichlet Laplacian eigenvalues on :math:`\\Omega`.

        Returns
        -------
        eig_vals : torch.Tensor
        eig_vecs : torch.Tensor
        sources : torch.Tensor
        """
        # Sample the sources
        sources = self.sample_ball(n_points=n_sources)

        # Get the quadrature points
        quad_all = self.sample_ball(n_points=n_quad_points)
        Nq = quad_all.shape[0]

        # Assemble the mass and stiffness matrices
        jac = self.jacobian(quad_all)
        rho = torch.linalg.det(jac)

        inv_jac = torch.linalg.inv(jac)
        A = rho[:, None, None] * (
            inv_jac @ inv_jac.transpose(1, 2)
        )

        phi = self.rbf(quad_all, sources, eps=eps, rbf=rbf)
        grad_phi = self.grad_rbf_x(quad_all, sources, eps=eps, rbf=rbf)

        M = torch.einsum("bi,bj,b->ij", phi, phi, rho)

        tmp = torch.einsum("bak,bik->bia", A, grad_phi)
        K = torch.einsum("bia,bja->ij", tmp, grad_phi)

        omega = torch.pi**(self.dim/2) / math.gamma(self.dim/2 + 1)
        scale = omega / Nq
        M *= scale
        K *= scale

        # Add boundary penalization
        quad_bd_all = self.sample_sphere(n_points=n_quad_points_bd) 
        Nq_bd = quad_bd_all.shape[0]

        # Careful: here we do not impose the penalization 
        # on the target shape, but on the reference sphere.
        phi = self.rbf(quad_bd_all, sources, eps=eps, rbf=rbf)
        P = torch.einsum("bi,bj->ij", phi, phi)

        sigma = self.dim*torch.pi**(self.dim/2)/math.gamma(self.dim/2+1)
        scale = sigma / Nq_bd
        P *= scale

        K = K + bd_penalization*P

        # Solve eigenproblem
        eig_vals, eig_vecs = self.generalized_sym_eig(K, M, k=n_ev)

        if normalize:
            vol = omega * torch.mean(torch.abs(torch.linalg.det(self.jacobian(quad_all))))
            eig_vals = vol ** (2 / self.dim) * eig_vals

        return eig_vals, eig_vecs, sources


    def poisson_problem(
        self,
        f,
        n_sources=500,
        n_quad_points=20000,
        n_quad_points_bd=200,
        eps=1.0,
        rbf='thin_plate',
        bd_penalization=1e5
    ):
        """
        Solve a Poisson problem with homogeneous Dirichlet boundary conditions on :math:`\\Omega`.

        This method approximates the solution of
        :math:`-\\Delta u = f` in :math:`\\Omega` with :math:`u = 0` on
        :math:`\\partial \\Omega`, using a radial basis function (RBF) Galerkin-type
        discretization combined with Monte Carlo quadrature.

        Parameters
        ----------
        f : callable
            Right-hand side function. Takes a tensor of shape ``(B, d)`` and returns
            a tensor of shape ``(B,)``.
        n_sources : int, optional
            Number of RBF centers (sources). Default is ``500``.
        n_quad_points : int, optional
            Number of quadrature points in the domain. Default is ``20000``.
        n_quad_points_bd : int, optional
            Number of quadrature points on the boundary (reference sphere).
            Default is ``200``.
        eps : float, optional
            Shape parameter for the RBF kernel. Default is ``1.0``.
        rbf : str, optional
            Type of radial basis function to use. Default is ``'thin_plate'``.
        bd_penalization : float, optional
            Penalization coefficient enforcing homogeneous Dirichlet boundary
            conditions. Default is ``1e5``.

        Returns
        -------
        torch.Tensor
            Tensor of shape ``(n_sources,)`` containing the coefficients of the RBF
            expansion of the solution.
        torch.Tensor
            Tensor of shape ``(n_sources, d)`` containing the source locations.
        """
        # Sample the sources
        sources = self.sample_ball(n_points=n_sources)

        # Get the quadrature points
        quad_all = self.sample_ball(n_points=n_quad_points)
        Nq = quad_all.shape[0]

        # Compute the change of var tensor and weight
        jac = self.jacobian(quad_all)
        rho = torch.linalg.det(jac)

        inv_jac = torch.linalg.inv(jac)
        A = rho[:, None, None] * (
            inv_jac @ inv_jac.transpose(1, 2)
        )

        # The volume and perimeter of the ball
        omega = torch.pi**(self.dim/2) / math.gamma(self.dim/2 + 1)
        sigma = self.dim*torch.pi**(self.dim/2)/math.gamma(self.dim/2+1)

        # Compute the basis functions and their gradient
        phi = self.rbf(quad_all, sources, eps=eps, rbf=rbf)
        grad_phi = self.grad_rbf_x(quad_all, sources, eps=eps, rbf=rbf)
        
        # Compute the right hand side
        f_transported = f(self.forward(quad_all))
        rhs = omega / Nq * torch.einsum('b, b, bj -> j', rho, f_transported, phi)

        # Compute the stifness matrix
        A_grad_phi = torch.einsum("bak,bik->bia", A, grad_phi)
        K = omega / Nq * torch.einsum("bia,bja->ij", A_grad_phi, grad_phi)

        # Add boundary penalization
        quad_bd_all = self.sample_sphere(n_points=n_quad_points_bd) 
        Nq_bd = quad_bd_all.shape[0]

        # Careful: here we do not impose the penalization 
        # on the target shape, but on the reference sphere.
        phi = self.rbf(quad_bd_all, sources, eps=eps, rbf=rbf)
        P = sigma / Nq_bd * torch.einsum("bi,bj->ij", phi, phi)

        K = K + bd_penalization * P

        # Solve problem
        u = torch.linalg.solve(K, rhs)

        return u, sources