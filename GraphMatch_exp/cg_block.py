import torch


def cg_block(A, b, n: torch.Tensor, M=None, x0=None, eps=1e-3, maxiter=None, verbose=False):
    assert n.dim() == 1
    assert b.dim() == 2

    if M is None:
        M = lambda x: x
    if x0 is None:
        # x0 = M(b)
        x0 = torch.zeros_like(b)
    if maxiter is None:
        maxiter = 100 * torch.max(n)

    dtype = b.dtype
    device = b.device
    n_b = n.shape[0]
    n_index = torch.repeat_interleave(n).unsqueeze(1).expand(-1, b.shape[1])

    xk = x0
    rk1 = b - A(xk)
    zk1 = M(rk1)

    pk = zk1
    w = A(pk)
    denominator = torch.zeros((n_b, b.shape[1]), dtype=dtype, device=device).scatter_add_(
        0, n_index, pk * w
    )
    is_zero = torch.lt(denominator, 1e-10)
    denominator.masked_fill_(is_zero, 1.)
    alphak = (torch.zeros((n_b, b.shape[1]), dtype=dtype, device=device).scatter_add_(
        0, n_index, rk1 * zk1
    )) / denominator
    alphak.masked_fill_(is_zero, 0.)
    gathered_alphak = torch.gather(alphak, 0, n_index)
    xk = xk + gathered_alphak * pk
    rk = rk1 - gathered_alphak * w

    if verbose:
        print("%03s | %010s" % ("it", "dist"))

    for it in range(1, maxiter + 1):
        residual = torch.sqrt(torch.zeros((n_b, b.shape[1]), dtype=dtype, device=device).scatter_add_(
            0, n_index, rk * rk
        ))
        if verbose:
            print("%03d | %8.4e" % (it, torch.max(residual).item()))
        if torch.all(residual <= eps):
            break

        zk = M(rk)

        denominator = torch.zeros((n_b, b.shape[1]), dtype=dtype, device=device).scatter_add_(
            0, n_index, rk1 * zk1
        )
        is_zero = torch.lt(denominator, 1e-10)
        denominator.masked_fill_(is_zero, 1.)
        rkzk = torch.zeros((n_b, b.shape[1]), dtype=dtype, device=device).scatter_add_(
            0, n_index, rk * zk
        )
        betak = rkzk / denominator
        betak.masked_fill_(is_zero, 0.)

        pk = zk + torch.gather(betak, 0, n_index) * pk

        w = A(pk)

        denominator = torch.zeros((n_b, b.shape[1]), dtype=dtype, device=device).scatter_add_(
            0, n_index, pk * w
        )
        is_zero = torch.lt(denominator, 1e-10)
        denominator.masked_fill_(is_zero, 1.)
        alphak = rkzk / denominator
        alphak.masked_fill_(is_zero, 0.)

        gathered_alphak = torch.gather(alphak, 0, n_index)
        xk = xk + gathered_alphak * pk
        rk1 = rk
        rk = rk1 - gathered_alphak * w
        zk1 = zk

    return xk


if __name__ == "__main__":
    A = torch.block_diag(*[
        torch.randn(3, 3),
        torch.randn(5, 5)
    ])
    A = torch.matmul(A, A.t()) + torch.eye(8) * 10
    b = torch.randn(8, 1)
    print(b.shape)
    n = torch.tensor([3, 5], dtype=torch.int64)

    print(A)

    x = cg_block(A.matmul, b, n)
    print(torch.matmul(A, x) - b)

