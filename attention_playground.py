# attention_playground.py
import torch.nn as nn
import torch.nn.functional as F
import torch

torch.manual_seed(1337)
B, T, C = 4, 8, 2      # 문장 4개, 토큰 8개, 토큰당 숫자 2개 (눈으로 보기 쉽게 작게)
x = torch.randn(B, T, C)

# ---- 1단계: for문 ----
# xbow[:, t] = 0번부터 t번까지 토큰 벡터의 평균
# (bow = bag of words: 순서 무시하고 한 자루에 담아 평균낸다는 뜻)
xbow = torch.zeros(B, T, C)
for t in range(T):
    # x[:, :t+1]       -> (B, t+1, C)  모든 문장의, 0~t번 토큰
    # .mean(dim=1)     -> (B, C)       토큰 축(T)을 평균내서 없앰
    xbow[:, t] = x[:, :t + 1].mean(dim=1)

print("x[0]:\n", x[0])
print("xbow[0]:\n", xbow[0])

# ---- 2단계: 행렬곱 ----
# tril: 하삼각 행렬 (lower triangular). 대각선 포함 아래쪽만 1
wei = torch.tril(torch.ones(T, T))
# 각 행을 그 행의 합으로 나눔 -> 행마다 합이 1인 "가중치"
wei = wei / wei.sum(dim=1, keepdim=True)
print("wei:\n", wei)

# (T, T) @ (B, T, C) -> (B, T, C)
# wei에는 B 축이 없지만, PyTorch가 문장 4개 각각에 같은 wei를 곱해 줌 (broadcasting)
xbow2 = wei @ x

# 1단계 결과와 같은지 확인 (float 오차를 감안해서 "거의 같은지" 비교)
print("1단계 == 2단계:", torch.allclose(xbow, xbow2))

# ---- 3단계: softmax + mask ----
tril = torch.tril(torch.ones(T, T))

# (1) 점수표: 지금은 전부 0 (= 모든 토큰을 똑같이 중요하게 봄)
#     4단계에서 이 0들이 "학습으로 계산된 점수"로 바뀜
wei = torch.zeros((T, T))

# 이걸 주면 True 가 안나옴
#wei[3, 2] = 2.0   # 3번 토큰이 2번 토큰에게 점수를 더 줌

# (2) 미래 칸(tril이 0인 곳)을 -inf로 막기
wei = wei.masked_fill(tril == 0, float("-inf"))
print("mask 후:\n", wei[:4, :4])

# (3) softmax: 점수 -> 합이 1인 비율
#     exp(-inf) = 0 이라서 미래 칸은 자동으로 0%가 됨
#     나머지는 점수가 전부 0으로 같으니 균등 분배 -> 평균과 같음
wei = torch.softmax(wei, dim=-1)
print("softmax 후:\n", wei[:4, :4])

xbow3 = wei @ x
print("1단계 == 3단계:", torch.allclose(xbow, xbow3))

# ---- 4단계: 진짜 self-attention (head 1개) ----
torch.manual_seed(1337)
B, T, C = 4, 8, 32         # 이번엔 C를 32로 (좀 더 현실적으로)
x = torch.randn(B, T, C)
head_size = 16

# nn.Linear(C, 16): "C개 숫자 -> 16개 숫자"로 바꾸는 학습 가능한 행렬 (C, 16)
# bias=False: 행렬곱만 하고 상수는 안 더함
# 이 세 행렬이 attention의 파라미터야
key   = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)     # (B, T, 16)  각 토큰의 꼬리표
q = query(x)   # (B, T, 16)  각 토큰이 찾는 것
v = value(x)   # (B, T, 16)  각 토큰이 건네줄 내용

# 점수: 모든 (나, 상대) 쌍에 대해 q·k 내적
# k.transpose(-2, -1): (B, T, 16) -> (B, 16, T)  마지막 두 축을 뒤집음
# (B, T, 16) @ (B, 16, T) -> (B, T, T)   = 3단계의 점수표 자리
# * head_size**-0.5: √16 = 4로 나눔. 이유는 아래 설명
wei = q @ k.transpose(-2, -1) * head_size**-0.5

# 여기부터는 3단계와 완전히 같음
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)

# 비율대로 x가 아니라 v를 섞음
out = wei @ v  # (B, T, T) @ (B, T, 16) -> (B, T, 16)

print("out shape:", out.shape)
print("0번 조각의 비율표:\n", wei[0].round(decimals=2))
print("행 합:", wei[0].sum(dim=-1))

# ---- Ch4 파트 1 확인 ----
from model import MultiHeadAttention, MLP

x = torch.randn(4, 8, 32)
mha = MultiHeadAttention(n_embd=32, n_head=4, block_size=8)
mlp = MLP(n_embd=32)
print("MHA out:", mha(x).shape)
print("MLP out:", mlp(x).shape)
print("MHA 파라미터:", sum(p.numel() for p in mha.parameters()))
print("MLP 파라미터:", sum(p.numel() for p in mlp.parameters()))


