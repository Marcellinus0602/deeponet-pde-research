# 두 Burgers 솔버(기존 upwind / 보존형)를 Cole-Hopf 정확해와 비교 (torch 불필요)
#
# 사용법 (프로젝트 루트에서):
#   python experiments/verify_burgers_solver.py
import sys
from pathlib import Path

# 상위 폴더를 모듈 검색 경로에 안전하게 등록
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np

from core.solvers.solve_burgers_1d import solve_burgers_1d
from core.solvers.solve_burgers_1d_conservative import solve_burgers_1d_conservative

NU = 0.01 / np.pi


def exact_burgers_sin(x: np.ndarray, t: float, c: float = 0.0, s: float = 1.0, nu: float = NU) -> np.ndarray:
    # u0 = c + s·sin(πx) 의 Cole-Hopf 정확해 (주기 경계, |s| <= 1, t <= 1 기준)
    #   평균 c 는 갈릴레이 변환 u(x, t) = c + w(x - ct, t) 로 분리, w0 = s·sin(πx)
    #   w(x, t) = ∫ (η/t)·φ0(x-η)·exp(-η²/4νt) dη / ∫ φ0(x-η)·exp(-η²/4νt) dη
    #   φ0(y) = exp(-∫_0^y w0 / 2ν) = exp(-s(1 - cos πy) / (2πν))
    eta = np.linspace(-1.5, 1.5, 300001)
    out = np.empty(len(x))
    for i, xi in enumerate(x - c * t):
        log_w = -s * (1 - np.cos(np.pi * (xi - eta))) / (2 * np.pi * nu) - eta ** 2 / (4 * nu * t)
        w = np.exp(log_w - log_w.max())  # 지수 오버플로 방지
        out[i] = np.sum(eta / t * w) / np.sum(w)
    return c + out


def shock_position(x: np.ndarray, u: np.ndarray) -> float:
    # 가장 급하게 떨어지는 지점 = 충격파 위치
    return x[np.argmin(np.roll(u, -1) - u)]


def main():
    nx, nt = 200, 100
    x = np.linspace(-1, 1, nx, endpoint=False)
    t_grid = np.linspace(0, 1, nt + 1)
    cases = [
        ("-sin(πx)        [대칭: 충격파 정지]", 0.0, -1.0),
        ("0.5 + sin(πx)   [평균≠0: 충격파 이동]", 0.5, 1.0),
        ("-0.3 - sin(πx)  [평균≠0: 충격파 이동]", -0.3, -1.0),
    ]
    check_times = [0.25, 0.5, 1.0]

    print(f"nx = {nx}, nu = 0.01/π, 정확해 = Cole-Hopf\n")
    print(f"{'초기조건':38s} {'t':>5s} | {'upwind relL2':>12s} {'conservative relL2':>18s} | "
          f"{'충격파 위치 exact / upwind / conservative':>40s}")
    for name, c, s in cases:
        u0 = c + s * np.sin(np.pi * x)
        sol_up = solve_burgers_1d(u0, x, t_max=1.0, nt=nt, nu=NU)
        sol_con = solve_burgers_1d_conservative(u0, x, t_max=1.0, nt=nt, nu=NU)
        for t in check_times:
            t_idx = int(np.argmin(np.abs(t_grid - t)))
            exact = exact_burgers_sin(x, t_grid[t_idx], c=c, s=s)
            err = lambda u: np.linalg.norm(u - exact) / np.linalg.norm(exact)
            pos = (f"{shock_position(x, exact):+.3f} / {shock_position(x, sol_up[t_idx]):+.3f} / "
                   f"{shock_position(x, sol_con[t_idx]):+.3f}")
            print(f"{name:38s} {t:5.2f} | {err(sol_up[t_idx]):12.4f} {err(sol_con[t_idx]):18.4f} | {pos:>40s}")
        print()


if __name__ == "__main__":
    main()
