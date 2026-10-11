import numpy as np
import torch
from core.solvers.solve_burgers_1d import solve_burgers_1d
from core.solvers.solve_burgers_1d_conservative import solve_burgers_1d_conservative
from utils.dataset.grf import sample_grf_periodic

def Burgers_1D_dataset(num_samples: int, m_sensors: int, nx: int = 200, nt: int = 100, pts_per_func: int = 20, nu: float = 0.01 / np.pi,
                       ic_type: str = "sin", solver: str = "upwind", seed: int = 0):
    # [DeepONet 학습용 데이터 생성]
    #   ic_type : 초기조건 종류  - "sin": -A sin(kπx) (기존 데이터셋), "grf": GRF 랜덤 곡선
    #   solver  : 정답 계산 솔버 - "upwind": 기존 비보존형, "conservative": 보존형
    #   seed    : 난수 시드      - 같은 seed 면 솔버가 달라도 초기조건·좌표가 같음 (솔버만 다른 공정한 비교)
    # 반환: (branch 입력, trunk 좌표, 정답, x 격자, t 격자) - 기존과 같은 형태
    if ic_type not in ("sin", "grf"):
        raise ValueError(f"ic_type 은 'sin' 또는 'grf': {ic_type}")
    if solver not in ("upwind", "conservative"):
        raise ValueError(f"solver 는 'upwind' 또는 'conservative': {solver}")

    x_grid = np.linspace(-1, 1, nx, endpoint=False)
    t_grid = np.linspace(0, 1, nt + 1)
    sensor_indices = np.linspace(0, nx - 1, m_sensors, dtype=int)

    # 이 함수 안의 랜덤(A, k, 좌표)은 모두 이 rng 에서 뽑음 → seed 가 같으면 항상 같은 데이터
    rng = np.random.default_rng(seed)

    u_branch_list = []
    trunk_coords_list = []
    target_sol_list = []

    num_funcs = num_samples // pts_per_func

    # GRF 초기조건은 미리 num_funcs 개를 한 번에 생성 (좌표용 rng 와 겹치지 않도록 seed + 1000 사용)
    if ic_type == "grf":
        u0_grf = sample_grf_periodic(num_funcs, x_grid, seed=seed + 1000)

    for i in range(num_funcs):
        # 1) 초기조건 u0(x) 만들기
        if ic_type == "sin":
            A = rng.uniform(0.5, 1.5)
            k = rng.choice([1, 2, 3, 4])
            u0 = -A * np.sin(k * np.pi * x_grid)
        else:
            u0 = u0_grf[i]

        # 2) 선택한 솔버로 시간별 해 전체 계산: (nt + 1, nx) 표
        if solver == "upwind":
            u_field = solve_burgers_1d(u0, x_grid, t_max=1.0, nt=nt, nu=nu)
        else:
            u_field = solve_burgers_1d_conservative(u0, x_grid, t_max=1.0, nt=nt, nu=nu)
        u_sensor_val = u0[sensor_indices]

        # 3) 해 표에서 랜덤 좌표 pts_per_func 개를 뽑아 (branch 입력, trunk 좌표, 정답) 짝으로 저장
        for _ in range(pts_per_func):
            t_idx = rng.integers(0, nt + 1)
            x_idx = rng.integers(0, nx)

            u_branch_list.append(u_sensor_val)
            trunk_coords_list.append([x_grid[x_idx], t_grid[t_idx]])
            target_sol_list.append([u_field[t_idx, x_idx]])

        # 진행 상황 출력 (함수 100개마다)
        if (i + 1) % 100 == 0:
            print(f"    [{ic_type}/{solver}] 수치해 계산 {i + 1}/{num_funcs}")

    u_data = torch.tensor(np.array(u_branch_list), dtype=torch.float32)
    y_data = torch.tensor(np.array(trunk_coords_list), dtype=torch.float32)
    target_data = torch.tensor(np.array(target_sol_list), dtype=torch.float32)

    return u_data, y_data, target_data, x_grid, t_grid
