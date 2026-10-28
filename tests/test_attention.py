# tests/test_attention.py
import torch
from model import Head

B, T, C, H = 2, 8, 32, 16  # 조각 2개, 토큰 8개, 토큰당 32차원, head_size 16


def make_head():
    torch.manual_seed(0)  # 매번 같은 랜덤 파라미터로 시작 (테스트 재현성)
    return Head(n_embd=C, head_size=H, block_size=T)


def test_output_shape():
    head = make_head()
    x = torch.randn(B, T, C)
    assert head(x).shape == (B, T, H)


def test_no_future_leak():
    head = make_head()
    x = torch.randn(B, T, C)
    out1 = head(x)

    # 마지막 토큰만 다른 값으로 바꿈
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, C)
    out2 = head(x2)

    # 앞쪽 T-1개 위치는 미래(마지막 토큰)를 못 보니까 결과가 그대로여야 함
    assert torch.allclose(out1[:, :-1], out2[:, :-1])
    # 마지막 위치는 자기 자신이 바뀌었으니 달라져야 함
    # (이게 없으면 "아무것도 안 하는 Head"도 테스트를 통과해버림)
    assert not torch.allclose(out1[:, -1], out2[:, -1])


def test_attention_rows_sum_to_one():
    head = make_head()
    head(torch.randn(B, T, C))
    row_sums = head.att.sum(dim=-1)  # (B, T)
    assert torch.allclose(row_sums, torch.ones(B, T))

