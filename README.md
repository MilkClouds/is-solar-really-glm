# Solar-Open-100B vs GLM-4.5-Air: 가중치 파생 분석

## 최종 결론: [Solar-Open-100B는 GLM-4.5-Air에서 파생되었다는 주장](https://github.com/sionic-ai/solar-vs-glm)의 근거는 충분하지 않습니다.

[solar-vs-glm](https://github.com/sionic-ai/solar-vs-glm)의 핵심 근거는 같은 모델 내에서 서로 다른 레이어끼리 구한 cosine sim(0.377)보다 solar[i]-glm[i]의 cosine sim(0.989)보다 월등히 크다는 사실입니다.

하지만 0.377이라는 숫자는 GLM[0]과 GLM[10], GLM[20], GLM[30], GLM[40] 사이에 잰 cosine similarity(0.37, 0.37, 0.37, 0.37)의 평균값으로, 이렇게 작은 값이 나오는 이유는 0번째 레이어는 다른 레이어에 비해 다른 특성을 가지고 있기 때문입니다. 실제로 0번째 레이어를 제외한 다른 레이어끼리 거리를 재면 0.99가 나옵니다.

0번째 레이어의 `input_layernorm.weight`는 분포가 다른 레이어와 크게 다릅니다. 예를 들어 GLM의 layer0은 mean≈0.01, std≈0.027인 반면, layer10/20/30/40은 mean이 0.49\~0.83 수준으로 크게 올라갑니다. Solar도 layer0 평균이 \~0.016로 낮고 이후 레이어는 0.23\~0.35 수준으로 커집니다. 이런 스케일 차이 때문에 layer0과 다른 레이어를 섞어 비교하면 cosine이 낮게 나오는 것이 자연스럽습니다.

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

<details>
<summary>각 레이어별 layernorm의 min, max, mean, std, head, tail</summary>

```
upstage__Solar-Open-100B_layer0_input_layernorm.npy
shape=(4096,) dtype=float32 min=-0.002472 max=+0.640625 mean=+0.016252 std=+0.040566
head[10]: +0.010986, +0.095703, +0.011230, +0.009766, +0.011475, +0.010742, +0.025391, +0.011475, +0.014099, +0.007141
tail[10]: +0.069336, +0.069336, +0.054688, +0.050781, +0.062500, +0.057129, +0.055664, +0.053223, +0.380859, +0.322266

upstage__Solar-Open-100B_layer10_input_layernorm.npy
shape=(4096,) dtype=float32 min=+0.007629 max=+0.578125 mean=+0.235737 std=+0.040956
head[10]: +0.213867, +0.248047, +0.225586, +0.190430, +0.195312, +0.195312, +0.279297, +0.212891, +0.220703, +0.219727
tail[10]: +0.263672, +0.300781, +0.277344, +0.261719, +0.287109, +0.265625, +0.273438, +0.273438, +0.417969, +0.417969

upstage__Solar-Open-100B_layer20_input_layernorm.npy
shape=(4096,) dtype=float32 min=+0.051758 max=+0.644531 mean=+0.321994 std=+0.031619
head[10]: +0.339844, +0.328125, +0.330078, +0.324219, +0.296875, +0.314453, +0.332031, +0.371094, +0.312500, +0.308594
tail[10]: +0.406250, +0.402344, +0.398438, +0.406250, +0.417969, +0.402344, +0.394531, +0.414062, +0.566406, +0.542969

upstage__Solar-Open-100B_layer30_input_layernorm.npy
shape=(4096,) dtype=float32 min=+0.053711 max=+0.617188 mean=+0.293781 std=+0.032517
head[10]: +0.306641, +0.300781, +0.271484, +0.287109, +0.273438, +0.308594, +0.291016, +0.304688, +0.265625, +0.296875
tail[10]: +0.392578, +0.371094, +0.367188, +0.390625, +0.390625, +0.398438, +0.375000, +0.398438, +0.582031, +0.617188

upstage__Solar-Open-100B_layer40_input_layernorm.npy
shape=(4096,) dtype=float32 min=+0.034180 max=+0.660156 mean=+0.349162 std=+0.033590
head[10]: +0.335938, +0.363281, +0.322266, +0.324219, +0.335938, +0.335938, +0.335938, +0.330078, +0.296875, +0.363281
tail[10]: +0.417969, +0.410156, +0.410156, +0.414062, +0.449219, +0.406250, +0.406250, +0.425781, +0.632812, +0.660156

zai-org__GLM-4.5-Air_layer0_input_layernorm.npy
shape=(4096,) dtype=float32 min=-0.000389 max=+0.566406 mean=+0.010872 std=+0.027050
head[10]: +0.003662, +0.006409, +0.003220, +0.007507, +0.009033, +0.007263, +0.009094, +0.007538, +0.008057, +0.008362
tail[10]: +0.037842, +0.007355, +0.006683, +0.005768, +0.009216, +0.008728, +0.005768, +0.009216, +0.012329, +0.008179

zai-org__GLM-4.5-Air_layer10_input_layernorm.npy
shape=(4096,) dtype=float32 min=+0.016479 max=+1.585938 mean=+0.496988 std=+0.046547
head[10]: +0.458984, +0.507812, +0.542969, +0.494141, +0.482422, +0.480469, +0.542969, +0.566406, +0.457031, +0.441406
tail[10]: +1.585938, +0.492188, +0.496094, +0.466797, +0.527344, +0.519531, +0.458984, +0.515625, +0.482422, +0.492188

zai-org__GLM-4.5-Air_layer20_input_layernorm.npy
shape=(4096,) dtype=float32 min=+0.121582 max=+1.703125 mean=+0.647851 std=+0.055031
head[10]: +0.613281, +0.644531, +0.675781, +0.671875, +0.632812, +0.656250, +0.667969, +0.750000, +0.609375, +0.605469
tail[10]: +1.703125, +0.632812, +0.632812, +0.585938, +0.617188, +0.675781, +0.625000, +0.671875, +0.667969, +0.628906

zai-org__GLM-4.5-Air_layer30_input_layernorm.npy
shape=(4096,) dtype=float32 min=+0.030151 max=+2.078125 mean=+0.774669 std=+0.058217
head[10]: +0.781250, +0.761719, +0.878906, +0.785156, +0.730469, +0.714844, +0.785156, +0.812500, +0.753906, +0.753906
tail[10]: +2.078125, +0.710938, +0.789062, +0.746094, +0.753906, +0.761719, +0.730469, +0.738281, +0.777344, +0.738281

zai-org__GLM-4.5-Air_layer40_input_layernorm.npy
shape=(4096,) dtype=float32 min=+0.159180 max=+2.531250 mean=+0.831595 std=+0.073827
head[10]: +0.832031, +0.800781, +0.945312, +0.886719, +0.789062, +0.761719, +0.792969, +0.816406, +0.792969, +0.765625
tail[10]: +2.531250, +0.750000, +0.800781, +0.812500, +0.796875, +0.796875, +0.773438, +0.769531, +0.863281, +0.785156
```


</details>

따라서 GLM[i] vs GLM[j]의 cosine similarity보다 GLM[i] vs SOLAR[i]의 cosine similarity가 월등히 크다는 것은 사실이 아니며 이는 두 모델이 유사하다는 근거가 될 수 없습니다.

## Usage of AI Agent

기존 [solar-vs-glm](https://github.com/sionic-ai/solar-vs-glm)의 코드와 리드미는 매우 heavy하게 LLM-generated된 것으로 보입니다. 리드미 스타일과 여러 파일이 정리되지 않고 나열되어 있는 모습 그리고 마크다운 파일도 FINAL_PROOF_REPORT.md와 README.md가 혼재되어 있는 등 Claude Code 혹은 그에 준하는 AI Agent를 이용해 작업된 것으로 보이며, 또한 전반적인 퀄리티를 보았을 때 인간의 개입은 크지 않았을 것으로 보입니다. 코드 상에도 굉장히 unusual한 구현이 많은데, 제가 Codex에게 질의한 질문과 답변을 아래에 참고하시라고 첨부하겠습니다. 굳이 제가 따로 설명을 덧붙이지 않아도 아래 답변을 읽어보는 것만으로 코드가 어떤 상태인지 바로 파악이 가능하시리라 생각합니다.

<details>
<summary>일반적이지 않은 solar-vs-glm의 코드 구현</summary>

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

### Q.

이거 구현 제대로된 거 맞아? 왜 그런식으로 계산해?

### A.
```
• 구현 “그대로 믿기엔 문제 많다”가 맞아요. 특히 0.37 같은 값은 코드상 계산 자체가 신뢰성이 낮습
  니다. 핵심 문제만 짚을게요.

  핵심 문제

  - definitive_proof.py:72에서 모든 텐서를 bf16로 강제 디코딩합니다. safetensors 헤더의 dtype를
  무시해서 fp16/fp32 텐서는 잘못 해석되고, cosine 값이 그냥 깨진 값이 될 수 있어요. 이게 0.37
  같은 값의 가장 그럴듯한 원인입니다.
  - definitive_proof.py:185에서 “within-model baseline”을 레이어0 vs 10/20/30/40의
  input_layernorm.weight로 잡는데, LayerNorm 가중치는 평균이 1 근처라 raw cosine이 구조적으로
  높게 나오거나(보통) 의미가 왜곡됩니다. baseline으로 쓰기엔 부적절합니다.
  - definitive_proof.py:185에서 표본이 너무 작고(4쌍), 레이어0에만 고정돼 있어 편향됩니다. 정상
  적인 baseline은 랜덤 레이어 쌍을 충분히 많이 뽑아 분포를 봐야 합니다.
  - definitive_proof.py:210에서 cross-model 비교도 dtype/shape 검증 없이 raw bytes를 바로 bf16
  으로 해석합니다. 이것도 잘못된 비교 가능성이 큽니다.

  왜 그렇게 계산했냐면

  - “같은 모델 다른 레이어의 유사도”를 기준선으로 쓰고, Solar-GLM이 그보다 높으면 파생 가능성이
  있다는 논리인데, 현재 구현은 dtype 무시 + LayerNorm raw cosine이라는 잘못된 기반 위에서 굴러
  갑니다. 그래서 숫자가 나와도 의미가 약해요.

  원하면 내가:

  1. dtype 기반 디코딩으로 고치고
  2. LayerNorm은 centered cosine(예: w-1)로 계산하고
  3. within-model baseline을 랜덤 레이어쌍 분포로 바꿔서
  4. 0.37의 재현 가능성과 의미를 다시 확인해줄게.

  어떻게 진행할까?

  1. 최소 수정(디코딩+baseline)
  2. 전체 비교 파이프라인 정리 (LN/attn/embedding까지 정규화 포함)
```
</details>

AI agent 자체가 문제가 있다기보다는 인간이 AI agent의 결과물에 대해 충분히 검수하지 않고 책임지지 않는 상황이 문제로 보여집니다. **저도 이 결과물을 AI로 생성했습니다.** 저는 오늘 저녁 디저트로 딸기를 먹으며 옆에 노트북을 펴놓고 20분동안 단 한 줄도 직접 코딩하지 않고 Codex에게 9번의 자연어 질의를 수행하여 문제점을 찾고 결론을 냈습니다. 그만큼 AI agent는 강력한 도구이지만, 마법의 도구가 아닙니다. 사용자는 누구든지 자신이 지시하는 작업과 그에 대한 결과물에 대한 온전한 책임을 져야 할 것입니다. [Codex와 저의 질의를 참고를 위해 첨부하겠습니다.](./conversation_with_codex.txt)

## How to Reproduce my result

just run `python3 definitive_proof.py` and `python3 view_saved_weights.py`
