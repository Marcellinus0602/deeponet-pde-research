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
from utils.dataset.burgers_1d_dataset  import Burgers_1D_dataset

# -------------------------------------------------------------
# 0. 디렉터리 및 실험 고유 아카이브 설정
# -------------------------------------------------------------
# 실험 실행 시마다 중복 덮어쓰기 방지를 위한 타임스탬프 폴더 생성
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
EXP_DIR = PROJECT_ROOT / "runs" / "burgers_fdm" / f"exp_{timestamp}"
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
    "nx": new_nx,
    "m_sensors": m_sensors,
    "epochs": epochs,
    "lr": 0.001,
    "step_size": 2000,
    "gamma": 0.5,
    "hidden_dim": 256,
    "p": 128,
    "device": str(device),
    "timestamp": timestamp
}

train_cache = DATA_DIR / f"train_data_nx{new_nx}_samples20k.pt"
test_cache = DATA_DIR / f"test_data_nx{new_nx}_samples4k.pt"

# [Train Data] 캐시 파일 존재 여부 확인 후 로드 또는 신규 생성
if train_cache.exists():
    print(f"[SSD Cache Load] 학습 데이터 로드: {train_cache}")
    train_dict = torch.load(train_cache, weights_only=False)
    u_train = train_dict["u"]
    y_train = train_dict["y"]
    target_train = train_dict["target"]
    x_sol = train_dict["x_sol"]
    t_sol = train_dict["t_sol"]
else:
    print(f"[Compute] 학습 데이터 신규 생성 중 (nx={new_nx}, 20,000 samples)...")
    u_train, y_train, target_train, x_sol, t_sol = Burgers_1D_dataset(
        num_samples=20000, m_sensors=m_sensors, nx=new_nx, pts_per_func=20
    )
    torch.save({
        "u": u_train, "y": y_train, "target": target_train, 
        "x_sol": x_sol, "t_sol": t_sol
    }, train_cache)
    print(f"[SSD Cache Save] 학습 데이터 캐싱 완료: {train_cache}")

# [Test Data] 캐시 파일 존재 여부 확인 후 로드 또는 신규 생성
if test_cache.exists():
    print(f"[SSD Cache Load] 테스트 데이터 로드: {test_cache}")
    test_dict = torch.load(test_cache, weights_only=False)
    u_test = test_dict["u"]
    y_test = test_dict["y"]
    target_test = test_dict["target"]
else:
    print(f"[Compute] 테스트 데이터 신규 생성 중 (nx={new_nx}, 4,000 samples)...")
    u_test, y_test, target_test, _, _ = Burgers_1D_dataset(
        num_samples=4000, m_sensors=m_sensors, nx=new_nx, pts_per_func=20
    )
    torch.save({
        "u": u_test, "y": y_test, "target": target_test
    }, test_cache)
    print(f"[SSD Cache Save] 테스트 데이터 캐싱 완료: {test_cache}")

# 텐서 디바이스 이동
u_train, y_train, target_train = u_train.to(device), y_train.to(device), target_train.to(device)
u_test, y_test, target_test = u_test.to(device), y_test.to(device), target_test.to(device)

# -------------------------------------------------------------
# 2. 모델, 옵티마이저, 스케줄러 구성
# -------------------------------------------------------------
print(f"2. DeepONet 모델 빌드 (Device: {device})...")
model_burgers = DeepONet(branch_dim=m_sensors, trunk_dim=2, hidden_dim=config["hidden_dim"], p=config["p"]).to(device)
optimizer = optim.Adam(model_burgers.parameters(), lr=config["lr"])
criterion = nn.MSELoss()
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

# -------------------------------------------------------------
# 3. 학습 루프 및 체크포인트 보관
# -------------------------------------------------------------
print(f"==== Burgers DeepONet 학습 시작 (Output: {EXP_DIR.name}) ====")
best_val_loss = float("inf")
history = {"epoch": [], "train_loss": [], "test_loss": [], "rel_l2": []}
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
        model_burgers.eval()
        with torch.no_grad():
            pred_val = model_burgers(u_test, y_test)
            val_loss = criterion(pred_val, target_test).item()
            rel_l2 = (torch.norm(pred_val - target_test) / torch.norm(target_test)).item()

        current_train_loss = loss.item()
        history["epoch"].append(epoch)
        history["train_loss"].append(current_train_loss)
        history["test_loss"].append(val_loss)
        history["rel_l2"].append(rel_l2)

        # 최저 Test Loss 갱신 시 타임스탬프 폴더에 가중치 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model_burgers.state_dict(), EXP_DIR / "best_model.pth")

        print(f"Epoch [{epoch:4d}/{epochs}] | Train Loss: {current_train_loss:.6f} | Test Loss: {val_loss:.6f} | Rel L2: {rel_l2:.4f}")

elapsed = time.time() - start_time
print(f"==== 학습 완료 (소요 시간: {elapsed:.2f}초) ====")

# -------------------------------------------------------------
# 4. 아티팩트 저장 (Config, History, Loss Curve)
# -------------------------------------------------------------
config["training_time_sec"] = elapsed
config["best_val_loss"] = best_val_loss

# 1) Config JSON 저장
with open(EXP_DIR / "config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4)

# 2) 학습 히스토리 JSON 저장
with open(EXP_DIR / "history.json", "w", encoding="utf-8") as f:
    json.dump(history, f, indent=4)

# 3) Loss Curve 플롯 저장
plt.figure(figsize=(8, 5))
plt.semilogy(history["epoch"], history["train_loss"], label="Train Loss (MSE)")
plt.semilogy(history["epoch"], history["test_loss"], label="Test Loss (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Convergence History")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(EXP_DIR / "loss_curve.png", dpi=200)
plt.close()

# -------------------------------------------------------------
# 5. Best 모델 복원 및 충격파 시각화
# -------------------------------------------------------------
print("3. Best 모델 로드 및 결과 플롯 생성...")
model_burgers.load_state_dict(torch.load(EXP_DIR / "best_model.pth", weights_only=True))
model_burgers.eval()

with torch.no_grad():
    u0_test_case = -np.sin(np.pi * x_sol)
    exact_sol_field = solve_burgers_1d(u0_test_case, x_sol, t_max=1.0, nt=100)

    sensor_indices = np.linspace(0, len(x_sol) - 1, m_sensors, dtype=int)
    u_sensor_input = torch.tensor(u0_test_case[sensor_indices], dtype=torch.float32).unsqueeze(0).to(device)

    eval_times = [0.25, 0.5, 0.8]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, t_eval in enumerate(eval_times):
        t_idx = int(t_eval * 100)

        t_column = np.full((len(x_sol), 1), t_eval)
        coords_eval = np.column_stack([x_sol, t_column])
        coords_tensor = torch.tensor(coords_eval, dtype=torch.float32).to(device)

        u_input_repeated = u_sensor_input.repeat(len(x_sol), 1)
        pred_profile = model_burgers(u_input_repeated, coords_tensor).cpu().numpy().flatten()
        exact_profile = exact_sol_field[t_idx]

        axes[idx].plot(x_sol, exact_profile, 'k-', label='Exact (FDM)')
        axes[idx].plot(x_sol, pred_profile, 'r--', label='DeepONet')
        axes[idx].set_title(f"t = {t_eval:.2f}")
        axes[idx].set_xlabel("x")
        axes[idx].set_ylabel("u(x, t)")
        axes[idx].grid(True)
        if idx == 0:
            axes[idx].legend()

    plt.tight_layout()
    pred_plot_path = EXP_DIR / "burgers_prediction.png"
    plt.savefig(pred_plot_path, dpi=200)
    print(f"결과물 아카이빙 완료: {EXP_DIR}")
    plt.show()