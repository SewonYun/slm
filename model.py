# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


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
