# tokenizer.py
import os
import urllib.request

DATA_PATH = "data/input.txt"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text() -> str:
    # 파일이 없을 때만 다운로드 (캐시 같은 개념)
    if not os.path.exists(DATA_PATH):
        os.makedirs("data", exist_ok=True)
        urllib.request.urlretrieve(URL, DATA_PATH)
    with open(DATA_PATH, encoding="utf-8") as f:
        return f.read()


class CharTokenizer:
    def __init__(self, text: str):
        # set(text): 텍스트에 등장한 글자를 중복 없이 모음
        # sorted: 순서를 고정해야 실행할 때마다 번호가 같아짐
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        # 양방향 매핑 테이블. TS로 치면 Map<string, number>, Map<number, string>
        self.stoi = {ch: i for i, ch in enumerate(chars)}  # string to int
        self.itos = {i: ch for i, ch in enumerate(chars)}  # int to string

    def encode(self, s: str) -> list[int]:
        # 글자 하나씩 번호로 바꿈. "hi" -> [46, 47]
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        # 번호를 글자로 바꿔서 이어 붙임
        return "".join(self.itos[i] for i in ids)


# 이 파일을 직접 실행할 때만 동작 (다른 파일에서 import하면 실행 안 됨)
if __name__ == "__main__":
    text = load_text()
    print("전체 글자 수:", len(text))
    print("앞부분:\n", text[:200])

    tok = CharTokenizer(text)
    print("vocab_size:", tok.vocab_size)
    print("글자 목록:", repr("".join(tok.itos[i] for i in range(tok.vocab_size))))

    ids = tok.encode("hello")
    print("encode:", ids)
    print("decode:", tok.decode(ids))
