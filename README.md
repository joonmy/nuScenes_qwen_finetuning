# nuScenes-QA VLM Fine-tuning & Evaluation

이 레포지토리는 Qwen3-VL-8B-Instruct 모델을 기반으로 nuScenes-QA 데이터셋을 가공하고, 학습 및 평가하기 위한 코드를 포함하고 있습니다.

## 1. 개요
nuScenes-QA 데이터를 VLM 학습에 적합한 대화형 포맷으로 변환하여 Hugging Face에 업로드하고, 이를 활용해 시각 언어 모델(VLM)을 파인튜닝한 뒤 성능을 검증합니다.

## 2. 주요 파일 설명
- `upload_dataset.py`: 원본 nuScenes-QA 데이터를 6개 시점의 이미지와 대화 포맷(`prompt`, `completion`)으로 변환하여 Hugging Face Hub에 업로드하는 스크립트.
- `train_vlm.py`: SFTTrainer를 사용한 VLM 학습 스크립트.
- `train_vlm.sh`: 학습 파라미터(LoRA 설정, Batch Size 등)가 포함된 실행 스크립트.
- `eval_vlm.py`: 학습된 모델을 불러와 정답률(Accuracy)을 측정하는 평가 스크립트.

## 3. 데이터 준비 (Data Preparation)
`upload_dataset.py`를 사용하여 데이터를 VLM 학습용 포맷으로 가공하고 업로드합니다.
- 원본 데이터의 6개 카메라 이미지(`CAM_FRONT`, `CAM_BACK` 등)를 하나의 리스트로 통합합니다.
- 질문과 답변을 `Qwen3-VL`이 이해할 수 있는 멀티모달 대화 포맷으로 구성합니다.

```bash
python upload_dataset.py
```

## 4. 학습 과정 (Training)
`train_vlm.sh`를 통해 학습을 진행합니다.
- **Dataset**: `JMandy/nuscenes_qa_mini_day` (가공된 데이터셋)
- **Model**: `Qwen/Qwen3-VL-8B-Instruct`
- **Method**: LoRA (Low-Rank Adaptation)
- **Epochs**: 5

```bash
bash train_vlm.sh
```

## 5. 평가 과정 (Evaluation)
학습 완료 후 저장된 체크포인트를 사용하여 검증 데이터셋(`validation` split)에 대한 성능을 측정합니다.
- 평가 항목: 생성된 답변과 실제 정답(Ground Truth)의 일치 여부.
- 실행 방법:
```bash
python eval_vlm.py
```

## 6. 결과 (Results)
- **Dataset Size**: 2229 samples (validation)
- **Performance**: Final Accuracy: 0.6101 (1360/2229)

## 7. 데이터셋 (Dataset)
- **가공된 데이터셋**: [JMandy/nuscenes_qa_mini_day](https://huggingface.co/datasets/JMandy/nuscenes_qa_mini_day)
