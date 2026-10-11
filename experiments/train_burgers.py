import sys
from pathlib import Path
from datetime import datetime

# 상위 폴더(C:\workspace)를 모듈 검색 경로에 안전하게 등록
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 모듈 임포트
from models.deeponet import DeepONet
from core.solvers.solve_burgers_1d import solve_burgers_1d
from core.solvers.solve_burgers_1d_conservative import solve_burgers_1d_conservative
from utils.dataset.burgers_1d_dataset  import Burgers_1D_dataset
from utils.dataset.grf import sample_grf_periodic

# -------------------------------------------------------------
# ★ 실험 설정: 아래 두 값만 바꿔서 4가지 경우를 각각 실행
#     1) IC_TYPE = "sin", SOLVER = "upwind"        → 기존 데이터셋 + 비보존형 솔버
#     2) IC_TYPE = "sin", SOLVER = "conservative"  → 기존 데이터셋 + 보존형 솔버
#     3) IC_TYPE = "grf", SOLVER = "upwind"        → GRF 데이터셋 + 비보존형 솔버
#     4) IC_TYPE = "grf", SOLVER = "conservative"  → GRF 데이터셋 + 보존형 솔버
#   코드를 고치지 않고 명령어로도 지정 가능:
#     python experiments/train_burgers.py grf conservative
# -------------------------------------------------------------
IC_TYPE = "sin"     # 초기조건 종류: "sin" 또는 "grf"
SOLVER = "upwind"   # 정답(수치해)을 만드는 솔버: "upwind" 또는 "conservative"

if len(sys.argv) == 3:
    IC_TYPE, SOLVER = sys.argv[1], sys.argv[2]
assert IC_TYPE in ("sin", "grf"), f"IC_TYPE 은 'sin' 또는 'grf': {IC_TYPE}"
assert SOLVER in ("upwind", "conservative"), f"SOLVER 는 'upwind' 또는 'conservative': {SOLVER}"

# 선택한 솔버 함수 (마지막 결과 그림의 정답 계산에 사용)
solver_fn = solve_burgers_1d if SOLVER == "upwind" else solve_burgers_1d_conservative

# -------------------------------------------------------------
# 0. 디렉터리 및 실험 고유 아카이브 설정
# -------------------------------------------------------------
# 실험 실행 시마다 중복 덮어쓰기 방지를 위한 타임스탬프 폴더 생성 (경우마다 폴더 분리)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = PROJECT_ROOT / "runs" / f"burgers_{IC_TYPE}_{SOLVER}" / f"exp_{timestamp}"
DATA_DIR = PROJECT_ROOT / "data" / "burgers"

EXP_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# 1. 하이퍼파라미터 및 SSD 데이터 캐싱 로드
# -------------------------------------------------------------
new_nx = 200
m_sensors = new_nx
epochs = 8000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 사용된 실험 파라미터 기록용 딕셔너리
config = {
    "equation": "Burgers_1D",
    "ic_type": IC_TYPE,
    "solver": SOLVER,
    "nx": new_nx,
    "m_sensors": m_sensors,
    "epochs": epochs,
    "lr": 0.001,
    "step_size": 2000,
    "gamma": 0.5,
    "hidden_dim": 256,
    "p": 128,
    "seed_train": 0,
    "seed_val": 1,
    "seed_test": 2,
    "seed_torch": 0,
    "device": str(device),
    "timestamp": timestamp
}


def load_or_make_dataset(split: str, num_samples: int, seed: int) -> dict:
    # 캐시 파일이 있으면 불러오고, 없으면 새로 만들어 저장 (같은 데이터를 매번 다시 계산하지 않기 위함)
    # 파일 이름에 초기조건·솔버·seed 를 넣어서, 설정이 바뀌면 다른 파일을 쓰도록 함
    cache = DATA_DIR / f"{split}_{IC_TYPE}_{SOLVER}_nx{new_nx}_n{num_samples}_seed{seed}.pt"
    if cache.exists():
        print(f"[SSD Cache Load] {split} 데이터 로드: {cache.name}")
        return torch.load(cache, weights_only=False)

    print(f"[Compute] {split} 데이터 신규 생성 중 ({IC_TYPE} + {SOLVER}, {num_samples} samples)...")
    u, y, target, x_sol, t_sol = Burgers_1D_dataset(
        num_samples=num_samples, m_sensors=m_sensors, nx=new_nx, pts_per_func=20,
        ic_type=IC_TYPE, solver=SOLVER, seed=seed
    )
    data = {"u": u, "y": y, "target": target, "x_sol": x_sol, "t_sol": t_sol}
    torch.save(data, cache)
    print(f"[SSD Cache Save] {split} 데이터 캐싱 완료: {cache.name}")
    return data


# train: 학습용 / val: best 모델 고르기용 / test: 최종 성능 확인용 (학습·모델 선택에 전혀 안 씀)
train_dict = load_or_make_dataset("train", 20000, seed=config["seed_train"])
val_dict = load_or_make_dataset("val", 4000, seed=config["seed_val"])
test_dict = load_or_make_dataset("test", 4000, seed=config["seed_test"])
x_sol = train_dict["x_sol"]

# 텐서 디바이스 이동
u_train, y_train, target_train = train_dict["u"].to(device), train_dict["y"].to(device), train_dict["target"].to(device)
u_val, y_val, target_val = val_dict["u"].to(device), val_dict["y"].to(device), val_dict["target"].to(device)
u_test, y_test, target_test = test_dict["u"].to(device), test_dict["y"].to(device), test_dict["target"].to(device)

# -------------------------------------------------------------
# 2. 모델, 옵티마이저, 스케줄러 구성
# -------------------------------------------------------------
print(f"2. DeepONet 모델 빌드 (Device: {device}, Case: {IC_TYPE} + {SOLVER})...")
torch.manual_seed(config["seed_torch"])  # 같은 설정이면 매번 같은 초기 가중치 → 결과 재현 가능
model_burgers = DeepONet(branch_dim=m_sensors, trunk_dim=2, hidden_dim=config["hidden_dim"], p=config["p"]).to(device)
optimizer = optim.Adam(model_burgers.parameters(), lr=config["lr"])
criterion = nn.MSELoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

# -------------------------------------------------------------
# 3. 학습 루프 및 체크포인트 보관
# -------------------------------------------------------------
print(f"==== Burgers DeepONet 학습 시작 (Output: {EXP_DIR.parent.name}/{EXP_DIR.name}) ====")
best_val_loss = float("inf")
history = {"epoch": [], "train_loss": [], "val_loss": [], "val_rel_l2": []}
start_time = time.time()

for epoch in range(1, epochs + 1):
    model_burgers.train()
    optimizer.zero_grad()

    pred = model_burgers(u_train, y_train)
    loss = criterion(pred, target_train)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model_burgers.parameters(), max_norm=1.0)
    optimizer.step()
    lr_scheduler.step()

    if epoch % 500 == 0 or epoch == 1:
        # val 데이터로 중간 성능 확인
        model_burgers.eval()
        with torch.no_grad():
            pred_val = model_burgers(u_val, y_val)
            val_loss = criterion(pred_val, target_val).item()
            rel_l2 = (torch.norm(pred_val - target_val) / torch.norm(target_val)).item()

        current_train_loss = loss.item()
        history["epoch"].append(epoch)
        history["train_loss"].append(current_train_loss)
        history["val_loss"].append(val_loss)
        history["val_rel_l2"].append(rel_l2)

        # 최저 Val Loss 갱신 시 타임스탬프 폴더에 가중치 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_burgers.state_dict(), EXP_DIR / "best_model.pth")

        print(f"Epoch [{epoch:4d}/{epochs}] | Train Loss: {current_train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Rel L2: {rel_l2:.4f}")

elapsed = time.time() - start_time
print(f"==== 학습 완료 (소요 시간: {elapsed:.2f}초) ====")

# -------------------------------------------------------------
# 4. Best 모델로 test 데이터 테스트
# -------------------------------------------------------------
model_burgers.load_state_dict(torch.load(EXP_DIR / "best_model.pth", weights_only=True))
model_burgers.eval()
with torch.no_grad():
    pred_test = model_burgers(u_test, y_test)
    test_loss = criterion(pred_test, target_test).item()
    test_rel_l2 = (torch.norm(pred_test - target_test) / torch.norm(target_test)).item()
print(f"==== Test 결과 ({IC_TYPE} + {SOLVER}) | Test Loss (MSE): {test_loss:.6f} | Test Rel L2: {test_rel_l2:.4f} ====")

# -------------------------------------------------------------
# 5. 아티팩트 저장 (Config, History, Loss Curve, 결과 모음 CSV)
# -------------------------------------------------------------
config["training_time_sec"] = elapsed
config["best_val_loss"] = best_val_loss
config["test_loss"] = test_loss
config["test_rel_l2"] = test_rel_l2

# 1) Config JSON 저장
with open(EXP_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

# 2) 학습 히스토리 JSON 저장
with open(EXP_DIR / "history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=4)

# 3) Loss Curve 플롯 저장
plt.figure(figsize=(8, 5))
plt.semilogy(history["epoch"], history["train_loss"], label="Train Loss (MSE)")
plt.semilogy(history["epoch"], history["val_loss"], label="Val Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Convergence History ({IC_TYPE} + {SOLVER})")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(EXP_DIR / "loss_curve.png", dpi=200)
plt.close()

# 4) 4가지 경우의 test 결과를 한 파일에 모아서 비교하기 위한 기록 (실행할 때마다 한 줄 추가)
results_path = PROJECT_ROOT / "runs" / "burgers_results.csv"
if not results_path.exists():
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("ic_type,solver,test_mse,test_rel_l2,best_val_mse,exp_dir\n")
with open(results_path, "a", encoding="utf-8") as f:
    f.write(f"{IC_TYPE},{SOLVER},{test_loss:.6e},{test_rel_l2:.6f},{best_val_loss:.6e},{EXP_DIR.parent.name}/{EXP_DIR.name}\n")

# -------------------------------------------------------------
# 6. Best 모델 예측 vs 정답 시각화
# -------------------------------------------------------------
print("3. Best 모델 결과 플롯 생성...")
with torch.no_grad():
    # 그림에 쓸 초기조건 2개
    #   - 첫 줄: -sin(πx) → 4가지 경우 모두 같은 입력이라 그림끼리 비교 가능
    #   - 둘째 줄: 이 경우의 초기조건 종류(sin/grf)에서 하나 (학습·테스트에 없는 함수)
    plot_ics = {"-sin(pi x)": -np.sin(np.pi * x_sol)}
    if IC_TYPE == "sin":
        plot_ics["-1.2 sin(3 pi x)"] = -1.2 * np.sin(3 * np.pi * x_sol)
    else:
        plot_ics["GRF sample"] = sample_grf_periodic(1, x_sol, seed=12345)[0]

    sensor_indices = np.linspace(0, len(x_sol) - 1, m_sensors, dtype=int)
    eval_times = [0.25, 0.5, 0.8]
    fig, axes = plt.subplots(len(plot_ics), 3, figsize=(15, 4 * len(plot_ics)))

    for row, (ic_name, u0_case) in enumerate(plot_ics.items()):
        # 정답: 이 경우에 선택한 솔버로 계산
        exact_sol_field = solver_fn(u0_case, x_sol, t_max=1.0, nt=100)
        u_sensor_input = torch.tensor(u0_case[sensor_indices], dtype=torch.float32).unsqueeze(0).to(device)

        for col, t_eval in enumerate(eval_times):
            t_idx = int(round(t_eval * 100))

            t_column = np.full((len(x_sol), 1), t_eval)
            coords_eval = np.column_stack([x_sol, t_column])
            coords_tensor = torch.tensor(coords_eval, dtype=torch.float32).to(device)

            u_input_repeated = u_sensor_input.repeat(len(x_sol), 1)
            pred_profile = model_burgers(u_input_repeated, coords_tensor).cpu().numpy().flatten()
            exact_profile = exact_sol_field[t_idx]

            ax = axes[row, col]
            ax.plot(x_sol, exact_profile, 'k-', label=f'Reference ({SOLVER})')
            ax.plot(x_sol, pred_profile, 'r--', label='DeepONet')
            ax.set_title(f"{ic_name}, t = {t_eval:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("u(x, t)")
            ax.grid(True)
            if col == 0:
                ax.legend()

    plt.tight_layout()
    pred_plot_path = EXP_DIR / "burgers_prediction.png"
    plt.savefig(pred_plot_path, dpi=200)
    print(f"결과물 아카이빙 완료: {EXP_DIR}")
    plt.show()
