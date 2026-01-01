# Solar-Open-100B vs GLM-4.5-Air: 가중치 파생 분석

## 최종 결론: [Solar-Open-100B는 GLM-4.5-Air에서 파생되었다는 주장](https://github.com/sionic-ai/solar-vs-glm)의 근거는 충분하지 않습니다.

[solar-vs-glm](https://github.com/sionic-ai/solar-vs-glm)의 핵심 근거는 같은 모델 내에서 서로 다른 레이어끼리 구한 cosine sim(0.377)보다 solar[i]-glm[i]의 cosine sim(0.989)보다 월등히 크다는 사실입니다.

하지만 0.377이라는 숫자는 GLM[0]과 GLM[10], GLM[20], GLM[30], GLM[40] 사이에 잰 cosine similarity(0.37, 0.37, 0.37, 0.37)의 평균값으로, 이렇게 작은 값이 나오는 이유는 0번째 레이어는 다른 레이어에 비해 다른 특성을 가지고 있기 때문입니다. 실제로 0번째 레이어를 제외한 다른 레이어끼리 거리를 재면 0.99가 나옵니다.

0번째 레이어의 `input_layernorm.weight`는 분포가 다른 레이어와 크게 다릅니다. 예를 들어 GLM의 layer0은 mean≈0.01, std≈0.027인 반면, layer10/20/30/40은 mean이 0.49~0.83 수준으로 크게 올라갑니다. Solar도 layer0 평균이 ~0.016로 낮고 이후 레이어는 0.23~0.35 수준으로 커집니다. 이런 스케일 차이 때문에 layer0과 다른 레이어를 섞어 비교하면 cosine이 낮게 나오는 것이 자연스럽습니다.

```
       Pairwise GLM comparisons among [0,10,20,30,40]
  GLM[0] vs GLM[10]: cos=0.375810
  GLM[0] vs GLM[20]: cos=0.375664
  GLM[0] vs GLM[30]: cos=0.377255
  GLM[0] vs GLM[40]: cos=0.379035
  GLM[10] vs GLM[20]: cos=0.998163
  GLM[10] vs GLM[30]: cos=0.997360
  GLM[10] vs GLM[40]: cos=0.996732
  GLM[20] vs GLM[30]: cos=0.997556
  GLM[20] vs GLM[40]: cos=0.996758
  GLM[30] vs GLM[40]: cos=0.999041
```

따라서 GLM[i] vs GLM[j]의 cosine similarity보다 GLM[i] vs SOLAR[i]의 cosine similarity가 월등히 크다는 것은 사실이 아니며 이는 두 모델이 유사하다는 근거가 될 수 없습니다.

## Usage of AI Agent

기존 [solar-vs-glm](https://github.com/sionic-ai/solar-vs-glm)의 코드와 리드미는 매우 heavy하게 LLM-generated된 것으로 보입니다. 리드미 스타일과 여러 파일이 정리되지 않고 나열되어 있는 모습 그리고 마크다운 파일도 FINAL_PROOF_REPORT.md와 README.md가 혼재되어 있는 등 Claude Code 혹은 그에 준하는 AI Agent를 이용해 작업된 것으로 보이며, 또한 전반적인 퀄리티를 보았을 때 인간의 개입은 크지 않았을 것으로 보입니다. 코드 상에도 굉장히 unusual한 구현이 많은데, 제가 Codex에게 질의한 질문과 답변을 아래에 참고하시라고 첨부하겠습니다. 굳이 제가 따로 설명을 덧붙이지 않아도 아래 답변을 읽어보는 것만으로 코드가 어떤 상태인지 바로 파악이 가능하시리라 생각합니다.

AI agent 자체가 문제가 있다기보다는 인간이 AI agent의 결과물에 대해 충분히 검수하지 않고 책임지지 않는 상황이 문제로 보여집니다. **저도 이 결과물을 AI로 생성했습니다.** 저는 오늘 저녁 디저트로 딸기를 먹으며 옆에 노트북을 펴놓고 20분동안 단 한 줄도 직접 코딩하지 않고 Codex에게 9번의 자연어 질의를 수행하여 문제점을 찾고 결론을 냈습니다. AI agent는 강력한 도구이지만, 마법의 도구가 아닙니다. 사용자는 누구든지 자신이 지시하는 작업과 그에 대한 결과물에 대한 온전한 책임을 져야 할 것입니다. [Codex와 저의 질의를 참고삼으라고 첨부하겠습니다.](./conversation_with_codex.txt)

### Q.

여기서 같은 모델에서 서로 다른 레이어끼리 cosine sim 0.37 나왔다는데 관련 코드 찾아줘 어떻게 구현한건지

### A.

```
• AGENTS.md는 이 리포에서 찾지 못했어요. 대신 요청하신 0.37(동일 모델 다른 레이어) cosine 계산
  코드는 definitive_proof.py에 있습니다.

  같은 모델 레이어 간 cosine(≈0.377) 계산 위치:

  - definitive_proof.py:176부터 “Within-model baseline” 섹션에서 GLM 레이어 0 vs 10/20/30/40의
  input_layernorm.weight를 가져와 cosine을 계산합니다.
  - 동일한 출력 포맷이 README.md에 그대로 정리돼 있습니다 (README.md:190).

  핵심 구현 흐름 요약:

  - 레이어별 input_layernorm.weight를 HF safetensors에서 raw bytes로 가져옴 (fetch_raw_bytes)
  - bf16 raw를 fp32로 디코딩 (decode_bf16)
  - flatten 후 cosine 계산 (cosine)
  - 레이어 0 vs 10/20/30/40 반복

  관련 코드 (요약, 동일 로직):

  # definitive_proof.py:176
  for layer_j in [10, 20, 30, 40]:
      key_i = "model.layers.0.input_layernorm.weight"
      key_j = f"model.layers.{layer_j}.input_layernorm.weight"

      raw_i = fetch_raw_bytes(...)
      raw_j = fetch_raw_bytes(...)

      arr_i = decode_bf16(raw_i)
      arr_j = decode_bf16(raw_j)

      cos = cosine(arr_i, arr_j)
      print(f"  GLM[0] vs GLM[{layer_j}]: cos={cos:.6f}")

  cosine 함수 정의:

  - definitive_proof.py:76

  원하시면 이 계산이 어디서 호출되는지(실행 진입점)나, 다른 스크립트에서 동일 결과를 재현하는
  경로도 정리해 드릴게요.
```

## How to Reproduce my result

just run `python3 definitive_proof.py`
