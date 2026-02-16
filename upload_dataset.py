import numpy as np
from PIL import Image
from datasets import load_dataset, Features, Sequence, Image as DatasetsImage

def generate_dataset():
    dataset_name = "KevinNotSmile/nuscenes-qa-mini"
    config = "day"
    
    # 1. 데이터 로드
    print(f"Loading {config} set...")
    ds_dict = load_dataset(dataset_name, config, data_files={
        # "train": "day-train/*.arrow", 
        "validation": "day-validation/*.arrow"
    })
    cam_keys = [
        'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
    ]
    # 2. 변환 함수 정의
    def transform_to_images(example):
        
        q_text = example["question"].strip()
        a_text = example["answer"].strip()
        
        # 'prompt' 컬럼 형식 구성 (이미지 6장에 맞게 마커 6개 생성)
        content = [{"text": None, "type": "image"} for _ in range(6)]
        content.append({"text": f"{q_text}\n", "type": "text"})
        
        prompt = [
            {
                "content": content,
                "role": "user"
            }
        ]
        
        # 'completion' 컬럼 형식 구성
        completion = [
            {
                "content": [
                    {"text": a_text, "type": "text"}
                ],
                "role": "assistant"
            }
        ]

        return {
            "images": [
                Image.fromarray(np.array(example['CAM_FRONT_LEFT'], dtype=np.uint8)),
                Image.fromarray(np.array(example['CAM_FRONT'], dtype=np.uint8)),
                Image.fromarray(np.array(example['CAM_FRONT_RIGHT'], dtype=np.uint8)),
                Image.fromarray(np.array(example['CAM_BACK_LEFT'], dtype=np.uint8)),
                Image.fromarray(np.array(example['CAM_BACK'], dtype=np.uint8)),
                Image.fromarray(np.array(example['CAM_BACK_RIGHT'], dtype=np.uint8)),
            ],
            "prompt": prompt,
            "completion": completion
        }
    print("Converting raw arrays to Pillow images...")

    ds_dict = ds_dict.map(
        transform_to_images, 
        remove_columns=ds_dict["validation"].column_names,
        desc="Processing images"
    )

    print("Uploading to Hugging Face...")
    ds_dict.push_to_hub("JMandy/nuscenes_qa_mini_day_valid")
    print("Successfully pushed to hub!")

if __name__ == "__main__":
    generate_dataset()