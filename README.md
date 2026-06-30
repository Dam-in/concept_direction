# Concept Direction

Discovering generalizable concept directions in diffusion models **without text supervision**.

<img width="1885" height="584" alt="image" src="https://github.com/user-attachments/assets/53ea7fa8-d831-4fb8-9ee4-958b5b01458a" />


## Idea
- FFN gate hyperplane이 이미지 집합을 의미적으로 coherent한 region으로 나눔
- region 간 activation 차이 → **concept direction** → UNet injection으로 steering

## Quick start
1. `config/{concept}.yaml` 수정 (원하는 target concept으로 수정)
2. 노트북 순서대로 실행:
   - `01_extract_activation` → 데이터 생성
   - `02_load_and_analyze` → W2/b2, hyperplane
   - `03_neuron_selection` → plane 선택, coherence
   - `04_injection_single` / `05_injection_multi` → steering
3. 결과: `analysis/results/{concept}/`

## Poster
[KCC_poster_damin.pdf](https://github.com/user-attachments/files/29503288/KCC_poster_damin.pdf)
