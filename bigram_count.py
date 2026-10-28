# bigram_count.py
import math
import torch
from tokenizer import load_text, CharTokenizer

def bigram_loss(counts: torch.Tensor, ids: list[int]) -> float:
    #probs = counts.float()
    probs = (counts + 1).float()
    probs = probs / probs.sum(dim=1, keepdim=True)

    ids_t = torch.tensor(ids)
    a = ids_t[:-1]   # 마지막 빼고  = zip의 왼쪽
    b = ids_t[1:]    # 첫 번째 빼고 = zip의 오른쪽

    # 텐서 인덱스로 인덱스 텐서를 넣으면 한 번에 여러 칸을 꺼냄
    # p[i] = probs[a[i], b[i]]   -> shape (N,)
    # 앞에서 본 zip for문을 for문 없이 한 줄로 한 거야
    p = probs[a, b]

    return (-p.log()).mean().item()

def count_bigrams(ids: list[int], vocab_size: int) -> torch.Tensor:
    # (65, 65) 표를 0으로 채워서 시작
    # counts[a, b] = "a 다음에 b가 나온 횟수"
    counts = torch.zeros((vocab_size, vocab_size), dtype=torch.int64)

    # zip(ids, ids[1:]): 이웃한 두 개씩 묶기
    # [5, 7, 2] -> (5, 7), (7, 2)
    for a, b in zip(ids, ids[1:]):
        counts[a, b] += 1
    return counts


def generate(counts: torch.Tensor, start_id: int, length: int, g: torch.Generator) -> list[int]:
    # 횟수를 확률로 바꾸기: 각 행을 그 행의 합으로 나눔
    # keepdim=True -> 합의 shape이 (65, 1)로 유지돼서
    # (65, 65) / (65, 1) 이 행 단위로 나눠짐 (broadcasting)
    probs = counts.float()
    probs = probs / probs.sum(dim=1, keepdim=True)

    out = [start_id]
    cur = start_id
    for _ in range(length):
        # probs[cur]: 현재 글자 다음에 올 65개 글자의 확률 (합 1)
        # multinomial: 그 확률대로 1개 뽑기 (확률 높은 게 자주 뽑힘)
        nxt = torch.multinomial(probs[cur], num_samples=1, generator=g).item()
        out.append(nxt)
        cur = nxt
    return out


if __name__ == "__main__":
    text = load_text()
    tok = CharTokenizer(text)
    ids = tok.encode(text)

    print("카운트 중... (몇 초~십몇 초 걸려)")
    counts = count_bigrams(ids, tok.vocab_size)

    # 검증: q 다음에 가장 많이 나온 글자는?
    q = tok.stoi["q"]
    top = counts[q].argmax().item()
    print("q 다음 최빈 글자:", repr(tok.itos[top]), "/", counts[q, top].item(), "번")

    # 시드 고정: 매번 같은 결과가 나오게 (디버깅용)
    g = torch.Generator().manual_seed(42)
    start = tok.stoi["\n"]  # 줄바꿈에서 시작
    out = generate(counts, start, 300, g)
    print("----- 생성 결과 -----")
    print(tok.decode(out))

    print("----- loss -----")
    print("찍기 기준:", -math.log(1 / tok.vocab_size))   # 약 4.17
    print("바이그램 :", bigram_loss(counts, ids))         # 약 2.45 근처

    weird = tok.encode("qz")
    print("이상한 쌍:", bigram_loss(counts, weird))
