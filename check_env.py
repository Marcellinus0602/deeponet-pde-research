import numpy as np
import pandas as pd
import sympy as sp
import torch

print("=== 환경 테스트 결과 ===")
print("NumPy 버전  :", np.__version__)
print("Pandas 버전 :", pd.__version__)
print("SymPy 버전  :", sp.__version__)
print("PyTorch 버전:", torch.__version__)
print("CUDA 지원 여부:", torch.cuda.is_available())
print("========================")
print("가상환경 및 라이브러리 정상 동작 확인 완료!")