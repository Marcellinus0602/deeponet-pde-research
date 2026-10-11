# Burgers 1D DeepONet 비교 실험: 초기조건(sin / grf) × 솔버(upwind / conservative)
#
# 사용법 (프로젝트 루트에서):
#   python experiments/train_burgers_compare.py                          # 기본 3개 케이스 순서대로
#   python experiments/train_burgers_compare.py --cases sin:upwind       # 기존 조합을 같은 조건으로 재학습
#   python experiments/train_burgers_compare.py --cases grf:conservative --epochs 200   # 빠른 동작 확인
import sys
from pathlib import Path
from datetime import datetime

# 상위 폴더를 모듈 검색 경로에 안전하게 등록
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import argparse
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # 여러 케이스를 연속 실행하므로 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt

# 모듈 임포트
from models.deeponet import DeepONet
from utils.dataset.burgers_1d_operator_dataset import GRF_DEFAULTS, dataset_tag, load_or_create_dataset

DEFAULT_CASES = ["grf:conservative", "sin:conservative", "grf:upwind"]
DATA_DIR = PROJECT_ROOT / "data" / "burgers_compare"
RUNS_DIR = PROJECT_ROOT / "runs" / "burgers_compare"


def predict_full_grid(model: DeepONet, u_sensors: torch.Tensor, x_grid: torch.Tensor, t_grid: torch.Tensor,
                      device: torch.device) -> torch.Tensor:
    # branch 는 함수마다 1번, trunk 는 격자점마다 1번 계산 → (N, nt + 1, nx)
    # forward 의 sum(B * T) + bias 와 같은 값을 짝 데이터 없이 한 번에 계산
    t_mesh, x_mesh = torch.meshgrid(t_grid, x_grid, indexing="ij")
    coords = torch.stack([x_mesh.reshape(-1), t_mesh.reshape(-1)], dim=1).float().to(device)
    model.eval()
    with torch.no_grad():
        B = model.branch_net(u_sensors.to(device))   # (N, p)
        T = model.trunk_net(coords)                  # (P, p)
        out = B @ T.T + model.bias                   # (N, P)
    return out.reshape(len(u_sensors), len(t_grid), len(x_grid)).cpu()


def per_function_rel_l2(pred: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    # 함수마다 ||pred - ref|| / ||ref|| (전체 시공간 격자 기준)
    n = len(ref)
    return torch.linalg.norm((pred - ref).reshape(n, -1), dim=1) / torch.linalg.norm(ref.reshape(n, -1), dim=1)


def save_prediction_plot(pred_fields, ref_fields, x_grid, t_grid, title_ref, path, n_show=3,
                         eval_times=(0.25, 0.5, 0.8)):
    fig, axes = plt.subplots(n_show, len(eval_times), figsize=(15, 3.5 * n_show), squeeze=False)
    for row in range(n_show):
        for col, t_eval in enumerate(eval_times):
            t_idx = int(np.argmin(np.abs(t_grid - t_eval)))
            ax = axes[row, col]
            ax.plot(x_grid, ref_fields[row, t_idx], 'k-', label=title_ref)
            ax.plot(x_grid, pred_fields[row, t_idx], 'r--', label='DeepONet')
            ax.set_title(f"val #{row}, t = {t_grid[t_idx]:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("u(x, t)")
            ax.grid(True)
            if row == 0 and col == 0:
                ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def run_case(ic_type: str, solver: str, args, device: torch.device) -> dict:
    # -------------------------------------------------------------
    # 0. 디렉터리 및 실험 고유 아카이브 설정
    # -------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = RUNS_DIR / f"{ic_type}_{solver}" / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n==== [{ic_type} + {solver}] 실험 시작 (Output: {exp_dir.relative_to(PROJECT_ROOT)}) ====")

    # -------------------------------------------------------------
    # 1. 데이터 (train / val 은 같은 분포, 다른 seed)
    # -------------------------------------------------------------
    common = dict(ic_type=ic_type, solver=solver, pts_per_func=args.pts_per_func, m_sensors=args.nx,
                  nx=args.nx, nt=args.nt)
    train_kwargs = dict(num_funcs=args.num_train_funcs, seed=args.data_seed, **common)
    val_kwargs = dict(num_funcs=args.num_val_funcs, seed=args.data_seed + 1, **common)
    train_data = load_or_create_dataset(DATA_DIR, **train_kwargs)
    val_data = load_or_create_dataset(DATA_DIR, **val_kwargs)

    u_train, y_train, target_train = (train_data[k].to(device) for k in ("u", "y", "target"))
    u_val, y_val, target_val = (val_data[k].to(device) for k in ("u", "y", "target"))

    config = {
        "equation": "Burgers_1D",
        "ic_type": ic_type,
        "solver": solver,
        "grf": GRF_DEFAULTS if ic_type == "grf" else None,
        "nx": args.nx,
        "nt": args.nt,
        "m_sensors": args.nx,
        "num_train_funcs": args.num_train_funcs,
        "num_val_funcs": args.num_val_funcs,
        "pts_per_func": args.pts_per_func,
        "data_seed_train": args.data_seed,
        "data_seed_val": args.data_seed + 1,
        "torch_seed": args.torch_seed,
        "train_dataset": dataset_tag(**train_kwargs),
        "val_dataset": dataset_tag(**val_kwargs),
        "epochs": args.epochs,
        "lr": 0.001,
        "step_size": 2000,
        "gamma": 0.5,
        "hidden_dim": 256,
        "p": 128,
        "device": str(device),
        "timestamp": timestamp,
    }

    # -------------------------------------------------------------
    # 2. 모델, 옵티마이저, 스케줄러 구성 (기존 train_burgers.py 와 동일)
    # -------------------------------------------------------------
    torch.manual_seed(args.torch_seed)
    model = DeepONet(branch_dim=args.nx, trunk_dim=2, hidden_dim=config["hidden_dim"], p=config["p"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.MSELoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])

    # -------------------------------------------------------------
    # 3. 학습 루프 (best model 은 val loss 기준으로 저장)
    # -------------------------------------------------------------
    best_val_loss = float("inf")
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_rel_l2": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()

        pred = model(u_train, y_train)
        loss = criterion(pred, target_train)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        if epoch % args.eval_every == 0 or epoch == 1 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                pred_val = model(u_val, y_val)
                val_loss = criterion(pred_val, target_val).item()
                rel_l2 = (torch.norm(pred_val - target_val) / torch.norm(target_val)).item()

            history["epoch"].append(epoch)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val_loss)
            history["val_rel_l2"].append(rel_l2)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), exp_dir / "best_model.pth")

            print(f"Epoch [{epoch:4d}/{args.epochs}] | Train Loss: {loss.item():.6f} | "
                  f"Val Loss: {val_loss:.6f} | Val Rel L2: {rel_l2:.4f}")

    elapsed = time.time() - start_time
    print(f"==== 학습 완료 (소요 시간: {elapsed:.2f}초) ====")

    # -------------------------------------------------------------
    # 4. Best 모델로 val 전체 격자 평가 (함수별 상대 L2)
    # -------------------------------------------------------------
    model.load_state_dict(torch.load(exp_dir / "best_model.pth", weights_only=True))
    pred_fields = predict_full_grid(model, val_data["u_sensors"], val_data["x_grid"], val_data["t_grid"], device)
    rel_l2_funcs = per_function_rel_l2(pred_fields, val_data["fields"])

    config["training_time_sec"] = elapsed
    config["best_val_loss"] = best_val_loss
    config["val_full_grid_rel_l2_mean"] = rel_l2_funcs.mean().item()
    config["val_full_grid_rel_l2_median"] = rel_l2_funcs.median().item()
    config["val_full_grid_rel_l2_max"] = rel_l2_funcs.max().item()

    # -------------------------------------------------------------
    # 5. 아티팩트 저장 (Config, History, Loss Curve, 예측 그림)
    # -------------------------------------------------------------
    with open(exp_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    with open(exp_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)

    plt.figure(figsize=(8, 5))
    plt.semilogy(history["epoch"], history["train_loss"], label="Train Loss (MSE)")
    plt.semilogy(history["epoch"], history["val_loss"], label="Val Loss (MSE)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Convergence History ({ic_type} + {solver})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(exp_dir / "loss_curve.png", dpi=200)
    plt.close()

    save_prediction_plot(pred_fields.numpy(), val_data["fields"].numpy(), val_data["x_grid"].numpy(),
                         val_data["t_grid"].numpy(), f"Reference ({solver})", exp_dir / "burgers_prediction.png",
                         n_show=min(3, args.num_val_funcs))

    print(f"Val 전체 격자 Rel L2 (함수별 평균): {config['val_full_grid_rel_l2_mean']:.4f}")
    print(f"결과물 아카이빙 완료: {exp_dir}")
    return {"case": f"{ic_type}:{solver}", "exp_dir": str(exp_dir.relative_to(PROJECT_ROOT)),
            "best_val_loss": best_val_loss, "val_full_grid_rel_l2_mean": config["val_full_grid_rel_l2_mean"],
            "training_time_sec": elapsed}


def parse_case(text: str):
    ic_type, _, solver = text.partition(":")
    if ic_type not in ("sin", "grf") or solver not in ("upwind", "conservative"):
        raise argparse.ArgumentTypeError(f"'초기조건:솔버' 형식 (sin|grf):(upwind|conservative) 이어야 함: {text}")
    return ic_type, solver


def main():
    parser = argparse.ArgumentParser(description="Burgers 1D DeepONet: 초기조건 × 솔버 비교 학습")
    parser.add_argument("--cases", nargs="+", type=parse_case, default=[parse_case(c) for c in DEFAULT_CASES],
                        help="예: grf:conservative sin:conservative grf:upwind")
    parser.add_argument("--epochs", type=int, default=8000)
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--num_train_funcs", type=int, default=1000)
    parser.add_argument("--num_val_funcs", type=int, default=200)
    parser.add_argument("--pts_per_func", type=int, default=20)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--nt", type=int, default=100)
    parser.add_argument("--data_seed", type=int, default=0, help="train 은 이 값, val 은 +1")
    parser.add_argument("--torch_seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    summaries = [run_case(ic_type, solver, args, device) for ic_type, solver in args.cases]

    print("\n==== 전체 요약 ====")
    print(f"{'case':20s} {'best val MSE':>13s} {'val rel L2(함수별 평균)':>22s} {'time(s)':>9s}")
    for s in summaries:
        print(f"{s['case']:20s} {s['best_val_loss']:13.6f} {s['val_full_grid_rel_l2_mean']:22.4f} "
              f"{s['training_time_sec']:9.1f}")

    summary_path = RUNS_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=4, ensure_ascii=False)
    print(f"요약 저장: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
