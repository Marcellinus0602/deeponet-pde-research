import sys
from pathlib import Path

# 프로젝트 루트를 모듈 검색 경로에 등록 (pytest / 직접 실행 모두 지원)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np

from core.physics.burgers import burgers_exact_sine, solve_burgers_spectral

# 실행: python -m pytest tests   또는   python tests/test_burgers_solver.py
NU = 0.01 / np.pi
T_CHECK = (0.25, 0.5, 0.8, 1.0)  # 0.25: 충격파 생성(t = 1/pi) 전, 나머지: 생성 후


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref))


def check_against_exact(amplitude: float, k: int, n_fine: int = 1600, n_out: int = 200, nt: int = 100):
    # 촘촘한 격자로 풀고 학습용 격자(200칸)로 추린 뒤 Cole-Hopf 정확해와 비교
    x_fine = np.linspace(-1, 1, n_fine, endpoint=False)
    stride = n_fine // n_out
    u = solve_burgers_spectral(-amplitude * np.sin(k * np.pi * x_fine), nu=NU, t_max=1.0, nt=nt)
    errors = {}
    for t in T_CHECK:
        exact = burgers_exact_sine(x_fine[::stride], t, NU, amplitude, k)
        errors[t] = rel_l2(u[round(t * nt), ::stride], exact)
    return errors


def test_benchmark_sine_matches_cole_hopf():
    # 대표 벤치마크 u0 = -sin(pi x)
    errors = check_against_exact(amplitude=1.0, k=1)
    assert max(errors.values()) < 1e-6, errors


def test_hardest_training_ic_matches_cole_hopf():
    # 기존 학습 데이터 계열(A: 0.5~1.5, k: 1~4)에서 충격파가 가장 강하고 많은 경우
    errors = check_against_exact(amplitude=1.5, k=4)
    assert max(errors.values()) < 1e-6, errors


def test_batch_solve_matches_single_solves():
    # 여러 초기조건을 한 번에 풀어도 하나씩 푼 결과와 같아야 함 (최댓값이 같아 dt 도 같음)
    x = np.linspace(-1, 1, 400, endpoint=False)
    u0 = np.stack([-np.sin(np.pi * x), -np.sin(2 * np.pi * x)])
    batch = solve_burgers_spectral(u0, nu=NU, nt=20)
    assert batch.shape == (2, 21, 400)
    for i in range(2):
        single = solve_burgers_spectral(u0[i], nu=NU, nt=20)
        assert np.array_equal(batch[i, 0], u0[i])
        assert np.abs(batch[i] - single).max() < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"[PASS] {name}")
    for amp, k in [(1.0, 1), (1.5, 4)]:
        errs = check_against_exact(amp, k)
        print(f"u0 = -{amp} sin({k} pi x): " + ", ".join(f"t={t}: {e:.1e}" for t, e in errs.items()))
