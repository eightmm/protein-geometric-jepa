"""Real cuEquivariance tensor-product backend, optional and never silently substituted.

API reference: NVIDIA cuequivariance_torch FullyConnectedTensorProduct and
SphericalHarmonics documentation. This module requires installed CuEq packages.
The Cartesian/irrep bridge is calibrated from that exact installed SH basis,
so it does not assume an e3nn convention or reshape a rank-2 tensor incorrectly.
"""
from functools import lru_cache

import torch
from torch import nn
from .fibers import Fiber, FiberDims
from ..geometry.primitives import outer_stf, normalize


def stf_basis():
    b = torch.zeros(5, 3, 3, dtype=torch.float64)
    b[0] = torch.diag(torch.tensor([1., -1., 0.], dtype=b.dtype))/2**0.5
    b[1] = torch.diag(torch.tensor([-1., -1., 2.], dtype=b.dtype))/6**0.5
    for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)), start=2):
        b[k, i, j] = b[k, j, i] = 1/2**0.5
    return b


class CuEqMessage(nn.Module):
    def __init__(self, dims: FiberDims, backend: str):
        super().__init__()
        try:
            import cuequivariance as cue
            import cuequivariance_torch as cuet
        except ImportError as error:
            raise ImportError("CuEq requested. Install pip install '.[cueq]'; no fallback is implicit.") from error
        if backend == "cueq-cuda" and not torch.cuda.is_available():
            raise RuntimeError("cueq-cuda requires a CUDA GPU and matching CuEq ops wheel.")
        self.dims = dims
        irreps = cue.Irreps(cue.SO3, f"{dims.scalar}x0 + {dims.vector}x1 + {dims.tensor}x2")
        harmonics = cue.Irreps(cue.SO3, "1x0 + 1x1 + 1x2")
        self.tp = cuet.FullyConnectedTensorProduct(
            irreps, harmonics, irreps, layout=cue.mul_ir,
            shared_weights=True, internal_weights=True,
            method="naive" if backend == "cueq-naive" else "fused_tp",
        )
        self.sh = cuet.SphericalHarmonics([0, 1, 2], normalize=True,
                                         method="naive" if backend == "cueq-naive" else "uniform_1d")
        # Calibration is coordinate-independent, cached on CPU; each module
        # owns cloned buffers so moving one model cannot mutate the cache.
        for name, value in calibrated_basis().items():
            self.register_buffer(name, value.clone())

    def pack(self, h: Fiber):
        v = h.v @ self.m1
        t = torch.einsum('ncij,kij->nck', h.t, self.basis) @ self.m2
        return torch.cat((h.s, v.flatten(1), t.flatten(1)), -1)

    def unpack(self, x):
        s, v, t = x.split((self.dims.scalar, 3*self.dims.vector, 5*self.dims.tensor), -1)
        v = v.reshape(-1, self.dims.vector, 3) @ self.m1_inv
        t = t.reshape(-1, self.dims.tensor, 5) @ self.m2_inv
        return Fiber(s, v, torch.einsum('nck,kij->ncij', t, self.basis))

    def forward(self, h: Fiber, unit):
        # A coincident edge has no direction. Keep Y_0, set Y_{l>0}=0 rather
        # than feeding 0/0 into spherical harmonics or selecting a world axis.
        valid = unit.square().sum(-1) > 1e-12
        safe = torch.where(valid[:, None], unit, unit.new_tensor([1., 0., 0.]))
        y = self.sh(safe)
        y = torch.cat((y[:, :1], y[:, 1:]*valid[:, None]), -1)
        return self.unpack(self.tp(self.pack(h), y))


@lru_cache(maxsize=1)
def calibrated_basis():
    """Verify both l=1/l=2 conventions with out-of-calibration directions."""
    import cuequivariance_torch as cuet
    sh = cuet.SphericalHarmonics([0, 1, 2], normalize=True, method="naive")
    u, _ = normalize(torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                                  [1., 1., 0.], [1., 0., 1.], [0., 1., 1.],
                                  [1., -1., 1.], [1., 1., -1.]]))
    with torch.no_grad():
        y = sh(u).double()
        basis = stf_basis()
        q = torch.einsum('nij,kij->nk', outer_stf(u).double(), basis)
        m1 = torch.linalg.lstsq(u.double(), y[:, 1:4], driver='gelsd').solution
        m2 = torch.linalg.lstsq(q, y[:, 4:9], driver='gelsd').solution
        holdout, _ = normalize(torch.tensor([[.23, -.91, .72], [-.36, .84, .51],
                                            [.11, .83, -.67], [-.51, -.17, .82]]))
        target = sh(holdout).double()
        quad = torch.einsum('nij,kij->nk', outer_stf(holdout).double(), basis)
        for actual, expected in ((holdout.double() @ m1, target[:, 1:4]),
                                 (quad @ m2, target[:, 4:9])):
            if not torch.allclose(actual, expected, atol=1e-5, rtol=1e-5):
                raise RuntimeError("Unsupported CuEq spherical-harmonic basis.")
    return {name: value.float() for name, value in {
        "basis": basis, "m1": m1, "m2": m2,
        "m1_inv": torch.linalg.inv(m1), "m2_inv": torch.linalg.inv(m2)}.items()}
