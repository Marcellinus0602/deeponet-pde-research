import numpy as np
import torch
from core.solvers.solve_burgers_1d import solve_burgers_1d

def Burgers_1D_dataset(num_samples: int, m_sensors: int, nx: int = 200, nt: int = 100, pts_per_func: int = 20, nu: float = 0.01 / np.pi):
    x_grid = np.linspace(-1, 1, nx, endpoint=False)
    t_grid = np.linspace(0, 1, nt + 1)
    sensor_indices = np.linspace(0, nx - 1, m_sensors, dtype=int)

    u_branch_list = []
    trunk_coords_list = []
    target_sol_list = []

    num_funcs = num_samples // pts_per_func

    for _ in range(num_funcs):
        A = np.random.uniform(0.5, 1.5)
        k = np.random.choice([1, 2, 3, 4])
        u0 = -A * np.sin(k * np.pi * x_grid)

        u_field = solve_burgers_1d(u0, x_grid, t_max=1.0, nt=nt, nu=nu)
        u_sensor_val = u0[sensor_indices]

        for _ in range(pts_per_func):
            t_idx = np.random.randint(0, nt + 1)
            x_idx = np.random.randint(0, nx)

            u_branch_list.append(u_sensor_val)
            trunk_coords_list.append([x_grid[x_idx], t_grid[t_idx]])
            target_sol_list.append([u_field[t_idx, x_idx]])

    u_data = torch.tensor(np.array(u_branch_list), dtype=torch.float32)
    y_data = torch.tensor(np.array(trunk_coords_list), dtype=torch.float32)
    target_data = torch.tensor(np.array(target_sol_list), dtype=torch.float32)

    return u_data, y_data, target_data, x_grid, t_grid