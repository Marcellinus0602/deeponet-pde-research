import numpy as np

def solve_burgers_1d_conservative(u0: np.ndarray, x: np.ndarray, t_max: float = 1.0, nt: int = 100, nu: float = 0.01 / np.pi):
    # [보존형 Burgers 솔버]
    # solve_burgers_1d (기존, 비보존형) 와 구조·입출력이 완전히 같고, 대류항 계산 방식만 다름
    #   기존(비보존형): u * du/dx 를 칸마다 따로 계산
    #   보존형      : d(u²/2)/dx 를 "칸 경계를 지나가는 양(플럭스)의 차이"로 계산
    #   → 한 칸에서 나간 양 = 옆 칸에 들어온 양 이라 전체 양이 보존되고, 충격파가 움직이는 속도가 정확함
    nx = len(x)
    dx = x[1] - x[0]

    u_history = np.zeros((nt + 1, nx))
    u_history[0] = u0

    nt_sub = max(2000, nt)
    dt = t_max / nt_sub
    sub_ratio = nt_sub // nt

    u_curr = u0.copy()

    def rhs_fvm(u_vec):
        u_shifted_right = np.roll(u_vec, -1)
        u_shifted_left = np.roll(u_vec, 1)

        # 칸 i 와 칸 i+1 사이 경계를 지나가는 양 (Godunov 플럭스, f(u) = u²/2)
        u_left, u_right = u_vec, u_shifted_right
        f_left, f_right = 0.5 * u_left ** 2, 0.5 * u_right ** 2
        flux_shock = np.maximum(f_left, f_right)                                          # u_left > u_right: 충격파
        flux_spread = np.where(u_left > 0, f_left, np.where(u_right < 0, f_right, 0.0))   # u_left <= u_right: 퍼지는 파
        flux_right = np.where(u_left > u_right, flux_shock, flux_spread)                  # 오른쪽 경계 (i+1/2)
        flux_left = np.roll(flux_right, 1)                                                # 왼쪽 경계 (i-1/2) = 옆 칸의 오른쪽 경계

        # 확산항 2차 중앙 차분 (기존과 동일)
        d2u_dx2 = (u_shifted_right - 2 * u_vec + u_shifted_left) / (dx ** 2)

        return (nu * d2u_dx2) - (flux_right - flux_left) / dx

    save_idx = 1
    for step in range(1, nt_sub + 1):
        k1 = rhs_fvm(u_curr)
        k2 = rhs_fvm(u_curr + 0.5 * dt * k1)
        k3 = rhs_fvm(u_curr + 0.5 * dt * k2)
        k4 = rhs_fvm(u_curr + dt * k3)
        u_curr = u_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if step % sub_ratio == 0 and save_idx <= nt:
            u_history[save_idx] = u_curr
            save_idx += 1

    return u_history
