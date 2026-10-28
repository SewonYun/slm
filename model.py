# model.py
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
