from pathlib import Path

import numpy as np
import torch

from core.solvers.solve_burgers_1d import solve_burgers_1d
from core.solvers.solve_burgers_1d_conservative import solve_burgers_1d_conservative
from utils.dataset.grf import sample_grf_periodic

SOLVERS = {
    "upwind": solve_burgers_1d,                     # 기존 솔버 (비보존형 풍상 차분)
    "conservative": solve_burgers_1d_conservative,  # 보존형 솔버 (Godunov 플럭스)
}
GRF_DEFAULTS = {"length_scale": 0.2, "wave_sigma": 0.7, "mean_sigma": 0.3, "n_modes": 64}


def sample_sin_ic(n_funcs: int, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # 기존 Burgers_1D_dataset 과 같은 분포: u0 = -A sin(kπx), A ~ U(0.5, 1.5), k ∈ {1, 2, 3, 4}
    A = rng.uniform(0.5, 1.5, size=(n_funcs, 1))
    k = rng.choice([1, 2, 3, 4], size=(n_funcs, 1))
    return -A * np.sin(k * np.pi * x[None, :])


def dataset_tag(ic_type: str, solver: str, num_funcs: int, pts_per_func: int = 20, m_sensors: int = 200,
                nx: int = 200, nt: int = 100, nu: float = 0.01 / np.pi, seed: int = 0, grf_kwargs: dict = None) -> str:
    # 데이터 생성 조건을 모두 이름에 담아서, 조건이 바뀌면 다른 캐시 파일을 쓰도록 함
    if ic_type == "grf":
        g = {**GRF_DEFAULTS, **(grf_kwargs or {})}
        ic = f"grf-ls{g['length_scale']}-ws{g['wave_sigma']}-ms{g['mean_sigma']}-nm{g['n_modes']}"
    else:
        ic = "sin-A0.5to1.5-k1to4"
    return f"{ic}_{solver}_nx{nx}_nt{nt}_m{m_sensors}_nu{nu:.5g}_f{num_funcs}x{pts_per_func}_seed{seed}"


def Burgers_1D_operator_dataset(num_funcs: int, ic_type: str = "grf", solver: str = "conservative",
                                pts_per_func: int = 20, m_sensors: int = 200, nx: int = 200, nt: int = 100,
                                nu: float = 0.01 / np.pi, seed: int = 0, grf_kwargs: dict = None) -> dict:
    if ic_type not in ("sin", "grf"):
        raise ValueError(f"ic_type 은 'sin' 또는 'grf': {ic_type}")
    if solver not in SOLVERS:
        raise ValueError(f"solver 는 {list(SOLVERS)} 중 하나: {solver}")

    x_grid = np.linspace(-1, 1, nx, endpoint=False)
    t_grid = np.linspace(0, 1, nt + 1)
    sensor_indices = np.linspace(0, nx - 1, m_sensors, dtype=int)

    # 초기조건용 / 좌표 샘플링용 난수를 분리 → 같은 seed 면 솔버가 달라도 초기조건·좌표가 완전히 동일
    ss_ic, ss_pts = np.random.SeedSequence(seed).spawn(2)

    # 1) 초기조건 생성: (num_funcs, nx)
    if ic_type == "sin":
        u0 = sample_sin_ic(num_funcs, x_grid, np.random.default_rng(ss_ic))
    else:
        g = {**GRF_DEFAULTS, **(grf_kwargs or {})}
        u0 = sample_grf_periodic(num_funcs, x_grid, seed=int(ss_ic.generate_state(1)[0]), **g)

    # 2) 함수마다 전체 수치해 계산: (num_funcs, nt + 1, nx)
    solve = SOLVERS[solver]
    fields = np.zeros((num_funcs, nt + 1, nx))
    for i in range(num_funcs):
        fields[i] = solve(u0[i], x_grid, t_max=1.0, nt=nt, nu=nu)
        if (i + 1) % 100 == 0 or i + 1 == num_funcs:
            print(f"    [{ic_type}/{solver}] 수치해 {i + 1}/{num_funcs}")
    if not np.isfinite(fields).all():
        raise RuntimeError("수치해에 NaN/Inf 발생: 솔버의 dt 안정 조건을 확인하세요")

    # 3) 함수마다 랜덤 좌표 pts_per_func 개 → 기존과 같은 (branch 입력, trunk 좌표, 정답) 짝 데이터
    rng_pts = np.random.default_rng(ss_pts)
    func_idx = np.repeat(np.arange(num_funcs), pts_per_func)
    t_idx = rng_pts.integers(0, nt + 1, size=func_idx.size)
    x_idx = rng_pts.integers(0, nx, size=func_idx.size)

    u_sensors = u0[:, sensor_indices]
    return {
        # 학습용 짝 데이터 (기존 Burgers_1D_dataset 과 같은 형태)
        "u": torch.tensor(u_sensors[func_idx], dtype=torch.float32),                                # (N*P, m)
        "y": torch.tensor(np.stack([x_grid[x_idx], t_grid[t_idx]], axis=1), dtype=torch.float32),  # (N*P, 2)
        "target": torch.tensor(fields[func_idx, t_idx, x_idx][:, None], dtype=torch.float32),      # (N*P, 1)
        # 함수 단위 전체 정보 (전체 격자 평가, 이후 비교 실험용)
        "u_sensors": torch.tensor(u_sensors, dtype=torch.float32),   # (N, m)
        "fields": torch.tensor(fields, dtype=torch.float32),         # (N, nt + 1, nx)
        "x_grid": torch.tensor(x_grid),
        "t_grid": torch.tensor(t_grid),
        "sensor_indices": torch.tensor(sensor_indices),
    }


def load_or_create_dataset(cache_dir: Path, **kwargs) -> dict:
    # 캐시가 있으면 불러오고, 없으면 생성 후 저장
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_tag(**kwargs)}.pt"

    if cache_path.exists():
        print(f"[SSD Cache Load] {cache_path.name}")
        return torch.load(cache_path, weights_only=True)

    print(f"[Compute] 데이터 신규 생성: {cache_path.name}")
    data = Burgers_1D_operator_dataset(**kwargs)
    torch.save(data, cache_path)
    print(f"[SSD Cache Save] {cache_path.name}")
    return data
