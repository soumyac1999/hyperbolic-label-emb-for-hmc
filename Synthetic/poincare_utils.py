import torch as th
from torch.autograd import Function

class Acosh(Function):
    @staticmethod
    def forward(ctx, x, eps=1e-5):
        z = th.sqrt(x * x - 1)
        ctx.save_for_backward(z)
        ctx.eps = eps
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        z, = ctx.saved_tensors
        z = th.clamp(z, min=ctx.eps)
        z = g / z
        return z, None


acosh = Acosh.apply

class PoincareDistance(Function):
    @staticmethod
    def grad(x, v, sqnormx, sqnormv, sqdist, eps):
        alpha = (1 - sqnormx)
        beta = (1 - sqnormv)
        z = 1 + 2 * sqdist / (alpha * beta)
        a = ((sqnormv - 2 * th.sum(x * v, dim=-1) + 1) / th.pow(alpha, 2))\
            .unsqueeze(-1).expand_as(x)
        a = a * x - v / alpha.unsqueeze(-1).expand_as(v)
        z = th.sqrt(th.pow(z, 2) - 1)
        z = th.clamp(z * beta, min=eps).unsqueeze(-1)
        return 4 * a / z.expand_as(x)

    @staticmethod
    def forward(ctx, u, v, eps=1e-5):
        squnorm = th.clamp(th.sum(u * u, dim=-1), 0, 1 - eps)
        sqvnorm = th.clamp(th.sum(v * v, dim=-1), 0, 1 - eps)
        sqdist = th.sum(th.pow(u - v, 2), dim=-1)
        ctx.eps = eps
        ctx.save_for_backward(u, v, squnorm, sqvnorm, sqdist)
        x = sqdist / ((1 - squnorm) * (1 - sqvnorm)) * 2 + 1
        # arcosh
        z = th.sqrt(th.pow(x, 2) - 1)
        return th.log(x + z)

    @staticmethod
    def backward(ctx, g):
        u, v, squnorm, sqvnorm, sqdist = ctx.saved_tensors
        g = g.unsqueeze(-1)
        gu = PoincareDistance.grad(u, v, squnorm, sqvnorm, sqdist, ctx.eps)
        gv = PoincareDistance.grad(v, u, sqvnorm, squnorm, sqdist, ctx.eps)
        return g.expand_as(gu) * gu, g.expand_as(gv) * gv, None
