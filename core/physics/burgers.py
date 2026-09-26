import numpy as np

from core.solvers.spectral_1d import dealias_mask, etdrk4_solve, wavenumbers

# 점성 Burgers 방정식: u_t + u u_x = nu u_xx  (주기 경계)


def burgers_operators(n: int, length: float, nu: float):
    # 선형항 nu*u_xx 와 비선형항 -(u^2/2)_x (보존형) 의 푸리에 표현
    kappa = wavenumbers(n, length)
    mask = dealias_mask(n)
    linear = -nu * kappa**2
    grad = -0.5j * kappa * mask

    def nonlinear(v_hat: np.ndarray) -> np.ndarray:
        u = np.fft.irfft(v_hat * mask, n=n, axis=-1)
        return grad * np.fft.rfft(u * u, axis=-1)

    return linear, nonlinear


def solve_burgers_spectral(u0: np.ndarray, nu: float = 0.01 / np.pi, t_max: float = 1.0, nt: int = 100,
                           length: float = 2.0, cfl: float = 0.2) -> np.ndarray:
    # u0: [-length/2, length/2) 위 균일 격자(endpoint=False)에서의 값, (n,) 또는 (batch, n)
    # 충격파 폭을 해상하려면 촘촘한 격자(예: nu=0.01/pi 에서 n=1600)로 풀고 필요한 만큼 추려 쓸 것
    n = np.shape(u0)[-1]
    linear, nonlinear = burgers_operators(n, length, nu)

    # Burgers 해의 최댓값은 커지지 않으므로(최대 원리) 초기 최댓값으로 시간 간격을 정함
    u_max = max(float(np.max(np.abs(u0))), 1e-8)
    dt = cfl * (length / n) / u_max
    return etdrk4_solve(u0, linear, nonlinear, t_max=t_max, n_save=nt, dt=dt)


def burgers_exact_sine(x: np.ndarray, t: float, nu: float, amplitude: float = 1.0, k: int = 1,
                       ds: float = 1e-4, chunk: int = 64) -> np.ndarray:
    # u0 = -A sin(k pi x) 에 대한 정확해 (Cole-Hopf 변환)
    #   u(x,t) = ∫ (s/t) phi0(x-s) exp(-s^2/(4 nu t)) ds / ∫ phi0(x-s) exp(-s^2/(4 nu t)) ds
    #   phi0(y) = exp(-c cos(k pi y)),  c = A / (2 k pi nu)
    # 지수가 e^{±c} 로 매우 커지므로 log-sum-exp 로 계산
    x = np.asarray(x, dtype=np.float64)
    if t == 0:
        return -amplitude * np.sin(k * np.pi * x)

    c = amplitude / (2 * k * np.pi * nu)
    s_max = np.sqrt(4 * nu * t * (2 * c + 40.0))  # 가우시안 꼬리가 e^{-40} 이하가 되는 범위
    s = np.arange(-s_max, s_max + ds, ds)

    x_flat = x.ravel()
    out = np.empty_like(x_flat)
    for i in range(0, x_flat.size, chunk):
        xi = x_flat[i:i + chunk, None]
        log_w = -c * np.cos(k * np.pi * (xi - s)) - s**2 / (4 * nu * t)
        w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
        out[i:i + chunk] = (w @ s) / (t * w.sum(axis=1))
    return out.reshape(x.shape)
