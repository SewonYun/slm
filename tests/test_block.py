# tests/test_block.py
import torch
from model import Block

B, T, C = 2, 8, 32


def make_block():
    torch.manual_seed(0)
    return Block(n_embd=C, n_head=4, block_size=T)


def test_shape_preserved():
    # 입력과 출력 shape이 같아야 여러 층으로 쌓을 수 있음
    x = torch.randn(B, T, C)
    assert make_block()(x).shape == x.shape


def test_stackable():
    # 실제로 3층 쌓아서 통과되는지
    torch.manual_seed(0)
    blocks = torch.nn.Sequential(*[Block(C, 4, T) for _ in range(3)])
    x = torch.randn(B, T, C)
    assert blocks(x).shape == x.shape


def test_no_future_leak():
    # Ch3과 같은 테스트. Block 전체에서도 미래를 못 봐야 함
    block = make_block()
    x = torch.randn(B, T, C)
    out1 = block(x)
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(B, C)
    out2 = block(x2)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-6)
    assert not torch.allclose(out1[:, -1], out2[:, -1])

