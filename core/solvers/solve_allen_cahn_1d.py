import numpy as np

def solve_allen_cahn_1d(u0: np.ndarray, x: np.ndarray, t_max: float = 1.0, nt: int = 100,
                        diffusion: float = 0.0001, reaction: float = 5.0, nt_sub: int = 2000):
    # Allen-Cahn: u_t = diffusion * u_xx + reaction * (u - u^3), 주기 경계
    nx = len(x)
    dx = x[1] - x[0]

    u_history = np.zeros((nt + 1, nx))
    u_history[0] = u0

    nt_sub = max(nt_sub, nt)
    dt = t_max / nt_sub
    sub_ratio = nt_sub // nt

    u_curr = u0.copy()

    def rhs_fdm(u_vec):
        u_shifted_right = np.roll(u_vec, -1)
        u_shifted_left = np.roll(u_vec, 1)

        # 확산항 2차 중앙 차분
        d2u_dx2 = (u_shifted_right - 2 * u_vec + u_shifted_left) / (dx ** 2)

        # 반응항: u 가 0 근처면 불안정해서 +1 또는 -1 로 끌려감
        react = reaction * (u_vec - u_vec ** 3)

        return diffusion * d2u_dx2 + react

    save_idx = 1
    for step in range(1, nt_sub + 1):
        k1 = rhs_fdm(u_curr)
        k2 = rhs_fdm(u_curr + 0.5 * dt * k1)
        k3 = rhs_fdm(u_curr + 0.5 * dt * k2)
        k4 = rhs_fdm(u_curr + dt * k3)
        u_curr = u_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if step % sub_ratio == 0 and save_idx <= nt:
            u_history[save_idx] = u_curr
            save_idx += 1

    return u_history