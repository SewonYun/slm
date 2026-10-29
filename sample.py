# sample.py
import torch
from tokenizer import load_text, CharTokenizer
from model import GPT, GPTConfig

device = "mps" if torch.backends.mps.is_available() else "cpu"

ckpt = torch.load("ckpt.pt", map_location=device)
cfg = GPTConfig(**ckpt["config"])     # dict -> 설정 객체로 복원
model = GPT(cfg).to(device)           # 빈 모델(랜덤 파라미터) 만들고
model.load_state_dict(ckpt["model"])  # 저장된 파라미터로 덮어씀
model.eval()                          # dropout 끄기 (사용 단계)

tok = CharTokenizer(load_text())      # 같은 텍스트로 만들면 같은 번호표가 나옴 (sorted라서)
start = torch.zeros((1, 1), dtype=torch.long, device=device)   # '\n'에서 시작
out = model.generate(start, max_new_tokens=500)
print(tok.decode(out[0].tolist()))

