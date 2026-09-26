import numpy as np

# 1D 주기 경계 의사 스펙트럴(pseudo-spectral) 풀이기
#   u_t = L u + N(u)
#   L: 푸리에 공간에서 대각인 선형항 (예: 확산 nu * u_xx -> -nu * kappa^2)
#   N: 비선형항 (물리 공간에서 계산 후 다시 푸리에 변환)
# 시간 적분: ETDRK4 (Cox & Matthews 2002, Kassam & Trefethen 2005)


def wavenumbers(n: int, length: float) -> np.ndarray:
    # np.fft.rfft 순서에 맞춘 각파수 kappa = 2*pi*m / length (m = 0, 1, ..., n/2)
    return 2 * np.pi * np.fft.rfftfreq(n, d=length / n)


def dealias_mask(n: int) -> np.ndarray:
    # 2/3 규칙: |m| >= n/3 인 모드를 제거해 비선형항의 앨리어싱 오차를 막음
    return np.fft.rfftfreq(n, d=1.0 / n) < n / 3


def etdrk4_coefficients(linear: np.ndarray, dt: float, n_contour: int = 32):
    # dt*L 이 0에 가까울 때 생기는 소거 오차를 복소 평면 원주 평균으로 피함 (Kassam & Trefethen)
    hL = dt * linear
    r = np.exp(1j * np.pi * (np.arange(1, n_contour + 1) - 0.5) / n_contour)
    z = hL[:, None] + r[None, :]
    ez = np.exp(z)

    E = np.exp(hL)
    E2 = np.exp(hL / 2)
    Q = dt * np.real(np.mean((np.exp(z / 2) - 1) / z, axis=1))
    f1 = dt * np.real(np.mean((-4 - z + ez * (4 - 3 * z + z**2)) / z**3, axis=1))
    f2 = dt * np.real(np.mean((2 + z + ez * (z - 2)) / z**3, axis=1))
    f3 = dt * np.real(np.mean((-4 - 3 * z - z**2 + ez * (4 - z)) / z**3, axis=1))
    return E, E2, Q, f1, f2, f3


def etdrk4_solve(u0: np.ndarray, linear: np.ndarray, nonlinear, t_max: float, n_save: int, dt: float) -> np.ndarray:
    # u0: (n,) 또는 여러 초기조건을 묶은 (batch, n)
    # 반환: t = 0, t_max/n_save, ..., t_max 에서의 해. (n_save+1, n) 또는 (batch, n_save+1, n)
    u0 = np.asarray(u0, dtype=np.float64)
    single = u0.ndim == 1
    u0 = np.atleast_2d(u0)
    n = u0.shape[-1]

    # 저장 시각에 정확히 도달하도록 dt 를 살짝 줄임
    steps_per_save = max(1, int(np.ceil(t_max / (n_save * dt) - 1e-9)))
    dt = t_max / (n_save * steps_per_save)
    E, E2, Q, f1, f2, f3 = etdrk4_coefficients(linear, dt)

    u_history = np.empty((u0.shape[0], n_save + 1, n))
    u_history[:, 0] = u0
    v = np.fft.rfft(u0, axis=-1)

    for save_idx in range(1, n_save + 1):
        for _ in range(steps_per_save):
            Nv = nonlinear(v)
            a = E2 * v + Q * Nv
            Na = nonlinear(a)
            b = E2 * v + Q * Na
            Nb = nonlinear(b)
            c = E2 * a + Q * (2 * Nb - Nv)
            Nc = nonlinear(c)
            v = E * v + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * Nc
        u_history[:, save_idx] = np.fft.irfft(v, n=n, axis=-1)

    return u_history[0] if single else u_history
