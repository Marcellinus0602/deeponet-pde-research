import numpy as np

def solve_burgers_1d_conservative(u0: np.ndarray, x: np.ndarray, t_max: float = 1.0, nt: int = 100,
                                  nu: float = 0.01 / np.pi, nt_sub: int = 2000):
    # 보존형(flux form) Burgers: u_t + (u²/2)_x = nu * u_xx, 주기 경계
    # 셀 경계를 지나는 플럭스의 차이로 갱신 → 충격파 이동 속도가 정확히 맞음
    # 입출력 형식은 solve_burgers_1d 와 동일: (nt + 1, nx)
    nx = len(x)
    dx = x[1] - x[0]

    u_history = np.zeros((nt + 1, nx))
    u_history[0] = u0

    # 안정 조건(CFL, 확산수)을 만족하도록 서브스텝 수를 정하고, nt 의 배수로 맞춰 저장 시각을 t_grid 와 일치시킴
    u_max = max(np.abs(u0).max(), 1e-8)
    dt_stable = 0.5 * min(dx / u_max, dx ** 2 / nu)
    nt_sub = max(nt_sub, nt, int(np.ceil(t_max / dt_stable)))
    nt_sub = int(np.ceil(nt_sub / nt)) * nt
    dt = t_max / nt_sub
    sub_ratio = nt_sub // nt

    u_curr = u0.copy()

    def godunov_flux(u_left, u_right):
        # 계면 i+1/2 의 Godunov 플럭스 (f(u) = u²/2)
        f_left, f_right = 0.5 * u_left ** 2, 0.5 * u_right ** 2
        rarefaction = np.where(u_left > 0, f_left, np.where(u_right < 0, f_right, 0.0))  # 퍼지는 파
        shock = np.maximum(f_left, f_right)                                               # 충격파
        return np.where(u_left <= u_right, rarefaction, shock)

    def rhs_fvm(u_vec):
        u_shifted_right = np.roll(u_vec, -1)
        u_shifted_left = np.roll(u_vec, 1)

        # 대류항: 셀로 들어오고 나가는 플럭스 차이
        flux_right = godunov_flux(u_vec, u_shifted_right)   # F_{i+1/2}
        flux_left = np.roll(flux_right, 1)                  # F_{i-1/2}

        # 확산항 2차 중앙 차분
        d2u_dx2 = (u_shifted_right - 2 * u_vec + u_shifted_left) / (dx ** 2)

        return (nu * d2u_dx2) - (flux_right - flux_left) / dx

    for step in range(1, nt_sub + 1):
        k1 = rhs_fvm(u_curr)
        k2 = rhs_fvm(u_curr + 0.5 * dt * k1)
        k3 = rhs_fvm(u_curr + 0.5 * dt * k2)
        k4 = rhs_fvm(u_curr + dt * k3)
        u_curr = u_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if step % sub_ratio == 0:
            u_history[step // sub_ratio] = u_curr

    return u_history
