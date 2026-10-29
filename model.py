# model.py
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    """self-attention head 1개"""

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # register_buffer: 모델에 붙어 다니지만 학습은 안 되는 텐서
        # (model.to("mps") 하면 같이 옮겨지고, parameters()엔 안 들어감)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5   # (B, T, T)
        # tril은 block_size 크기라서, 실제 입력 길이 T만큼만 잘라 씀
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        att = F.softmax(scores, dim=-1)                           # (B, T, T)
        self.att = att
        return att @ v                                            # (B, T, head_size)

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        # 32차원을 head 4개가 8차원씩 나눠 가짐 -> 이어 붙이면 다시 32
        # 그래서 head를 늘려도 계산량은 head 1개(32차원)일 때와 비슷해
        assert n_embd % n_head == 0, "n_embd는 n_head로 나눠떨어져야 함"
        head_size = n_embd // n_head

        # nn.ModuleList: 모듈 목록. 그냥 파이썬 list에 넣으면 parameters()에 등록이 안 돼서
        # 옵티마이저가 못 찾고 학습이 안 됨 (조용히 실패하니까 주의)
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size) for _ in range(n_head)]
        )
        # 각 head 결과를 이어 붙인 뒤 한 번 섞어 주는 행렬 (출력 프로젝션)
        self.proj = nn.Linear(n_embd, n_embd)
        # 학습 때 일부 값을 랜덤으로 0으로 만듦 (과적합 방지). eval()이면 꺼짐
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 각 head: (B, T, 8) -> 마지막 축으로 이어 붙임 -> (B, T, 32)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))  # (B, T, 32)


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        # nn.Sequential: 순서대로 통과시키는 파이프라인
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 32 -> 128로 넓힘
            nn.GELU(),                      # 비선형 함수 (아래 설명)
            nn.Linear(4 * n_embd, n_embd),  # 128 -> 32로 다시 좁힘
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # nn.Linear는 마지막 축(C)에만 작용 -> 토큰끼리 섞이지 않고 토큰마다 따로 계산됨
        return self.net(x)  # (B, T, 32)

class Block(nn.Module):
    """Transformer 블록 1개: 소통(attention) + 소화(MLP)"""

    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        # 문서의 "TODO: 위 그림 그대로 두 줄로"가 이거야
        x = x + self.attn(self.ln1(x))  # 정규화 -> 토큰끼리 소통 -> 원본에 더함
        x = x + self.mlp(self.ln2(x))   # 정규화 -> 토큰별 소화   -> 원본에 더함
        return x                         # (B, T, C) 그대로

@dataclass(frozen=True)  # frozen: 만든 뒤 값 변경 불가 (설정값이 학습 중 바뀌는 사고 방지)
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128   # 한 번에 보는 최대 토큰 수 (컨텍스트 길이)
    n_layer: int = 4        # Block 몇 층
    n_head: int = 4         # Block당 head 수
    n_embd: int = 128       # 토큰 벡터 차원 (C)
    dropout: float = 0.1


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        c = config
        self.tok_emb = nn.Embedding(c.vocab_size, c.n_embd)  # (65, 128)  토큰 임베딩 표
        self.pos_emb = nn.Embedding(c.block_size, c.n_embd)  # (128, 128) 위치 임베딩 표
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.Sequential(
            *[Block(c.n_embd, c.n_head, c.block_size, c.dropout) for _ in range(c.n_layer)]
        )
        self.ln_f = nn.LayerNorm(c.n_embd)
        self.lm_head = nn.Linear(c.n_embd, c.vocab_size)       # 128 -> 65 점수

        # 모든 하위 모듈에 _init_weights를 적용
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # PyTorch 기본 초기화 대신 작은 값(std 0.02)으로 시작
        # -> 초기 logits가 0 근처라 loss가 4.17 근처에서 출발 (Ch2에서 zeros_로 본 그 효과)
        # GPT-2/nanoGPT 관례를 따른 값이야
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.config.block_size, f"T={T}가 block_size보다 큼"

        tok = self.tok_emb(idx)                                    # (B, T, C)
        # torch.arange(T): [0, 1, ..., T-1] 위치 번호
        # device=idx.device: 입력과 같은 장치(cpu/mps)에 만들어야 에러 안 남
        pos = self.pos_emb(torch.arange(T, device=idx.device))     # (T, C)
        x = self.drop(tok + pos)   # (B,T,C) + (T,C) -> broadcasting으로 모든 조각에 같은 위치 벡터를 더함

        x = self.blocks(x)         # (B, T, C)  Block 4층 통과
        x = self.ln_f(x)           # (B, T, C)
        logits = self.lm_head(x)   # (B, T, vocab_size)

        if targets is None:
            return logits, None
        # 여기부터는 BigramLM과 완전히 같음
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        for _ in range(max_new_tokens):
            # 위치 임베딩 표가 block_size(128)칸뿐이라, 그보다 길면 뒤쪽 128개만 잘라서 넣음
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                 # 마지막 위치만 (B, V)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

class BigramLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        # nn.Embedding(65, 65): 65행 65열짜리 학습 가능한 표
        # table[a] = "a 다음 글자들에 대한 점수 65개"
        # Ch1의 counts와 같은 모양. 다만 값이 랜덤으로 시작하고 학습으로 바뀜
        self.table = nn.Embedding(vocab_size, vocab_size)
        nn.init.zeros_(self.table.weight)  

    def forward(self, idx, targets=None):
        # idx: (B, T) 글자 번호들
        # 각 번호로 표의 행을 꺼내서 -> logits: (B, T, 65)
        # Ch0의 (4, 8, 16)이랑 같은 패턴이야. C 자리가 16 대신 65일 뿐
        logits = self.table(idx)

        if targets is None:
            return logits, None

        # cross_entropy는 (N, C) 점수와 (N,) 정답을 원함
        # 그래서 (B, T, 65) -> (B*T, 65), (B, T) -> (B*T,)로 펼침
        # 문장 경계는 의미 없고, 그냥 "예측 B*T개를 한 줄로 세운다"는 뜻
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()  # 생성할 땐 학습 안 하니까 기울기 계산 끔 (빠르고 메모리 절약)
    def generate(self, idx, max_new_tokens: int):
        # idx: (B, T)에서 시작해서 한 글자씩 뒤에 붙여 나감
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]          # 마지막 위치만 -> (B, 65)
            probs = F.softmax(logits, dim=-1)  # 점수 -> 확률
            nxt = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, nxt], dim=1)  # 뒤에 이어 붙임 -> (B, T+1)
        return idx
