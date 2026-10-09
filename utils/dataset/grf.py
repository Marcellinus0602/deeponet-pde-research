import numpy as np


def sample_grf_periodic(n_funcs: int, x: np.ndarray, length_scale: float = 0.2, wave_sigma: float = 0.7,
                        mean_sigma: float = 0.3, n_modes: int = 64, seed: int = 0) -> np.ndarray:
    # 주기 구간 [-1, 1) 위의 GRF 초기조건을 n_funcs 개 생성
    #   u0(x) = c + Σ_{k=1..n_modes} [a_k cos(kπx) + b_k sin(kπx)]
    #   c ~ N(0, mean_sigma²)       : 상수항(평균값). 전체를 평균 속도로 옮김, ℓ과 무관하게 고정
    #   a_k, b_k ~ N(0, λ_k)        : 파동 부분. 주파수가 높을수록 작게 섞음
    #   λ_k ∝ exp(-(kπℓ)²/2), Σλ_k = wave_sigma²  → 파동 크기가 ℓ과 무관하게 wave_sigma
    rng = np.random.default_rng(seed)                       # 시드 고정 → 매번 같은 곡선
    k = np.arange(1, n_modes + 1)
    weights = np.exp(-0.5 * (k * np.pi * length_scale) ** 2)
    lam = wave_sigma**2 * weights / weights.sum()
    a = rng.normal(size=(n_funcs, n_modes)) * np.sqrt(lam)
    b = rng.normal(size=(n_funcs, n_modes)) * np.sqrt(lam)
    c = rng.normal(size=(n_funcs, 1)) * mean_sigma
    kx = np.pi * k[:, None] * x[None, :]                    # (n_modes, nx)
    return c + a @ np.cos(kx) + b @ np.sin(kx)              # (n_funcs, nx)