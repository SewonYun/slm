# train.py
import torch
from tokenizer import load_text, CharTokenizer
from model import BigramLM


def get_batch(data: torch.Tensor, block_size: int, batch_size: int):
    # 긴 글 안에서 시작 위치를 batch_size개 랜덤으로 고름
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # 각 위치에서 block_size만큼 잘라서 쌓음 -> (B, T)
    x = torch.stack([data[i : i + block_size] for i in ix])
    # 정답은 한 칸 밀린 것 -> (B, T)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

@torch.no_grad()  # 채점만 하니까 기울기 계산 끔
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters=200):
    # 배치 하나로 재면 운에 따라 들쭉날쭉하니까 200번 재서 평균
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

if __name__ == "__main__":
    torch.manual_seed(42)
    text = load_text()
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    # 90% 학습용, 10% 검증용
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    x, y = get_batch(train_data, block_size=8, batch_size=4)
    print("x shape:", x.shape)   # (4, 8)
    print("x[0]:", tok.decode(x[0].tolist()))
    print("y[0]:", tok.decode(y[0].tolist()))  # x[0]에서 한 글자 밀린 문자열

    # ---- 하이퍼파라미터 ----
    block_size = 8
    batch_size = 32
    max_steps = 5000
    eval_interval = 500
    lr = 1e-2

    model = BigramLM(tok.vocab_size)
    # 옵티마이저에 "고칠 숫자 목록"(표 4,225칸)을 넘김
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(max_steps + 1):
        # 500 step마다 성적표 출력
        if step % eval_interval == 0:
            l = estimate_loss(model, train_data, val_data, block_size, batch_size)
            print(f"step {step:5d} | train {l['train']:.4f} | val {l['val']:.4f}")

        x, y = get_batch(train_data, block_size, batch_size)
        _, loss = model(x, y)                  # 1. 예측하고 채점
        optimizer.zero_grad(set_to_none=True)  # 0. 지난번 기울기 지우기
        loss.backward()                        # 2. 칸마다 고칠 방향 계산
        optimizer.step()                       # 3. 조금 고침

    # ---- 학습 끝나고 생성 ----
    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long)  # 0번 = '\n'에서 시작, shape (B=1, T=1)
    out = model.generate(start, max_new_tokens=300)
    print(tok.decode(out[0].tolist()))

    # ---- 과적합 테스트: 같은 배치만 반복 ----
    print("----- overfit test -----")
    model2 = BigramLM(tok.vocab_size)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-2)
    x, y = get_batch(train_data, block_size=8, batch_size=1)  # 아주 작은 배치 하나 고정
    print("배치:", repr(tok.decode(x[0].tolist())), "->", repr(tok.decode(y[0].tolist())))
    for step in range(1000):
        _, loss = model2(x, y)
        opt2.zero_grad(set_to_none=True)
        loss.backward()
        opt2.step()
        if step % 200 == 0:
            print(step, loss.item())
    print("최종:", loss.item())


