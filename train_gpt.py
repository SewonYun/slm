# train.py
import time
from dataclasses import asdict

import torch
import matplotlib
matplotlib.use("Agg")  # 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt

from tokenizer import load_text, CharTokenizer
from model import GPT, GPTConfig

# Mac GPU가 있으면 mps, 없으면 cpu
device = "mps" if torch.backends.mps.is_available() else "cpu"


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    # 모델이 mps에 있으면 입력도 mps에 있어야 함 (섞이면 에러)
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters=100):
    out = {}
    model.eval()   # dropout 끄기
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # dropout 다시 켜기
    return out


if __name__ == "__main__":
    torch.manual_seed(1337)
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    # ---- 설정 ----
    cfg = GPTConfig(vocab_size=tok.vocab_size)   # block_size 128, 4층, 4head, 128차원
    batch_size = 32
    max_steps = 5000
    eval_interval = 500
    lr = 1e-3

    model = GPT(cfg).to(device)   # 파라미터를 GPU 메모리로 옮김
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | 파라미터: {n_params:,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # ---- 학습 ----
    history = []
    t0 = time.time()
    for step in range(max_steps + 1):
        if step % eval_interval == 0:
            l = estimate_loss(model, train_data, val_data, cfg.block_size, batch_size)
            history.append((step, l["train"], l["val"]))
            print(f"step {step:5d} | train {l['train']:.4f} | val {l['val']:.4f} | {time.time() - t0:.0f}s")

        x, y = get_batch(train_data, cfg.block_size, batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # ---- 저장 ----
    # state_dict: 파라미터 + 버퍼(tril)를 이름별로 담은 dict
    # config는 dict로 바꿔서 저장 (클래스 객체를 그대로 저장하면 불러올 때 문제가 생김, 아래 설명)
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, "ckpt.pt")
    print("저장: ckpt.pt")

    # ---- loss 그래프 ----
    steps, tr, va = zip(*history)
    plt.plot(steps, tr, label="train")
    plt.plot(steps, va, label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("loss.png")
    print("저장: loss.png")

