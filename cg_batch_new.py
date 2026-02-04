import torch


def cg_batch(A, b, M=None, x0=None, eps=1e-3, maxiter=None, verbose=False):
    assert b.dim() == 3

    if M is None:
        M = lambda x: x
    if x0 is None:
        # x0 = M(b)
        x0 = torch.zeros_like(b)
    if maxiter is None:
        maxiter = 100 * b.shape[1]

    xk = x0
    rk1 = b - A(xk)
    zk1 = M(rk1)

    pk = zk1
    w = A(pk)
    denominator = torch.sum(pk * w, dim=1, keepdim=True)
    is_zero = torch.lt(denominator, 1e-10)
    denominator.masked_fill_(is_zero, 1.)
    alphak = torch.sum(rk1 * zk1, dim=1, keepdim=True) / denominator
    alphak.masked_fill_(is_zero, 0.)
    xk = xk + alphak * pk
    rk = rk1 - alphak * w

    if verbose:
        print("%03s | %010s" % ("it", "dist"))

    for it in range(1, maxiter + 1):
        residual = torch.linalg.vector_norm(rk, ord=2, dim=1)
        if verbose:
            print("%03d | %8.4e" % (it, torch.max(residual).item()))
        if torch.all(residual <= eps):
            break

        zk = M(rk)

        denominator = torch.sum(rk1 * zk1, dim=1, keepdim=True)
        is_zero = torch.lt(denominator, 1e-10)
        denominator.masked_fill_(is_zero, 1.)
        rkzk = torch.sum(rk * zk, dim=1, keepdim=True)
        betak = rkzk / denominator
        betak.masked_fill_(is_zero, 0.)

        pk = zk + betak * pk

        w = A(pk)

        denominator = torch.sum(pk * w, dim=1, keepdim=True)
        is_zero = torch.lt(denominator, 1e-10)
        denominator.masked_fill_(is_zero, 1.)
        alphak = rkzk / denominator
        alphak.masked_fill_(is_zero, 0.)

        xk = xk + alphak * pk
        rk1 = rk
        rk = rk1 - alphak * w
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
    # n = torch.tensor([3, 5], dtype=torch.int64)

    print(A)

    x = cg_batch(A[None, :, :].matmul, b[None, :, :], verbose=True)
    print(torch.matmul(A, x) - b)

