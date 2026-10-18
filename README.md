# tiny-gpt

"Attention Is All You Need"를 공부한 뒤, 작은 GPT를 PyTorch로 처음부터 직접 만들어 보는 개인 스터디 저장소입니다.

- 목표: Mac(Apple Silicon, `mps`)에서 약 1,000만 파라미터 모델을 학습해 짧은 영어 동화 생성하기

## 진도

| 챕터 | 주제 | 주요 파일 | 완료 태그 | 상태 |
| --- | --- | --- | --- | --- |
| 0 | 준비 | `.venv`, 폴더 구조 | `ch0` | [ ] |
| 1 | 언어 모델이란 (세기 기반 바이그램) | `tokenizer.py`, `bigram_count.py` | `ch1` | [ ] |
| 2 | 신경망 학습 루프 | `train.py`, `model.py` | `ch2` | [ ] |
| 3 | Self-Attention | `model.py`, `tests/test_attention.py` | `ch3` | [ ] |
| 4 | Transformer 블록 | `model.py` | `ch4` | [ ] |
| 5 | GPT 조립과 첫 학습 (Tiny Shakespeare) | `model.py`, `train.py`, `sample.py` | `ch5` | [ ] |
| 6 | BPE 토크나이저 | `tokenizer.py` | `ch6` | [ ] |
| 7 | 규모 키우기 (TinyStories) | `prepare_data.py`, `train.py` | `ch7` | [ ] |
| 8 | 최신 구조 (RMSNorm, RoPE, SwiGLU, GQA) | `model.py` | `ch8` | [ ] |
| 9 | 추론 최적화 (KV cache, 샘플링) | `model.py`, `sample.py` | `ch9` | [ ] |
| 10 | (선택) SFT | `sft.py`, `chat.py` | `ch10` | [ ] |

## 환경 세팅

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy matplotlib tqdm datasets pytest
python -c "import torch; print(torch.backends.mps.is_available())"  # True면 성공
```

## 폴더 구조

```text
tiny-gpt/
  data/        # 다운로드한 데이터 (git에서 제외)
  tests/       # 챕터별 완료 확인 테스트
  notes.md     # 배운 것과 실험 결과 기록
  *.py         # 챕터를 진행하며 추가
```

## 커밋 규칙

챕터 안에서는 작은 단위로 자주 커밋하고, 챕터를 마치면 태그를 붙입니다.

**메시지 형식**: `<type>(chN): <설명>`

| type | 언제 | 예시 |
| --- | --- | --- |
| `feat` | 구현 과제 하나를 끝냈을 때 | `feat(ch3): causal mask를 적용한 Head 구현` |
| `test` | 완료 확인 테스트를 추가했을 때 | `test(ch3): 미래 토큰 누설 테스트 추가` |
| `fix` | 버그를 고쳤을 때 | `fix(ch2): x와 y가 한 칸 밀리지 않던 문제 수정` |
| `exp` | 실험 결과를 기록했을 때 | `exp(ch5): 모델 크기별 val loss 비교` |
| `docs` | notes.md, README를 고쳤을 때 | `docs(ch5): 진도표 업데이트` |

**챕터를 마칠 때**

```bash
pytest                                  # 완료 확인 테스트가 모두 통과하는지
# README 진도표의 상태를 [x]로 바꾸고 notes.md에 결과 기록
git add -A
git commit -m "docs(ch3): Ch3 완료, 진도표 업데이트"
git tag ch3
```

**나중에 되돌아볼 때**

```bash
git tag                       # 끝낸 챕터 목록
git log --oneline ch2..ch3    # Ch3에서 한 작업만 보기
git diff ch4 ch8 -- model.py  # GPT-2 구조와 최신 구조의 차이 보기
git checkout ch5              # Ch5 끝난 시점의 코드로 이동 (돌아올 땐 git checkout main)
```
