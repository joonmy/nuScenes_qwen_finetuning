import json
import os
from collections import defaultdict
from PIL import Image
from datasets import Dataset, DatasetDict, Features, Sequence, Value, Image as DatasetsImage
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

def load_nuscenes_qa_to_hf(dataroot='/home/mlp/jmlee/nuScenes', 
                             train_questions_path='/home/mlp/jmlee/nuScenes/NuScenes_train_questions.json',
                             val_questions_path='/home/mlp/jmlee/nuScenes/NuScenes_val_questions.json',
                             repo_id="JMandy/nuscenes_qa_grouped"):
    
    print("Initializing NuScenes API...")
    nusc = NuScenes(version='v1.0-trainval', dataroot=dataroot, verbose=False)
    
    camera_names = [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT'
    ]
    
    def get_grouped_qa(json_path):
        """JSON 파일을 읽어서 sample_token별로 질문/답변을 그룹화합니다."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        grouped = defaultdict(list)
        for item in data['questions']:
            grouped[item['sample_token']].append({
                "question": item['question'],
                "answer": item['answer']
            })
        return grouped

    def gen_examples(questions_json_path):
        grouped_data = get_grouped_qa(questions_json_path)
        
        for sample_token, qa_pairs in tqdm(grouped_data.items(), desc=f"Processing {os.path.basename(questions_json_path)}"):
            try:
                # 이미지 경로 리스트 생성 (PIL 객체 대신 경로만 저장)
                sample = nusc.get('sample', sample_token)
                image_paths = []
                for cam in camera_names:
                    cam_data = nusc.get('sample_data', sample['data'][cam])
                    img_path = os.path.join(dataroot, cam_data['filename'])
                    image_paths.append(img_path)
                
                # 각 QA 페어마다 하나의 로우를 생성
                for qa in qa_pairs:
                    q_text = qa['question'].strip()
                    a_text = qa['answer'].strip()

                    # 'prompt' 컬럼 형식 구성 (이미지 6장에 맞게 마커 6개 생성)
                    content = [{"text": None, "type": "image"} for _ in range(6)]
                    content.append({"text": f"{q_text}\n", "type": "text"})
                    
                    prompt = [{"content": content, "role": "user"}]
                    
                    # 'completion' 컬럼 형식 구성
                    completion = [{"content": [{"text": a_text, "type": "text"}], "role": "assistant"}]

                    yield {
                        "images": image_paths,
                        "sample_token": sample_token,
                        "prompt": prompt,
                        "completion": completion
                    }
            except Exception as e:
                print(f"Error processing sample {sample_token}: {e}")
                continue

    def resize_images(example):
        """한 샘플의 모든 이미지를 224x224로 리사이즈"""
        # cast_column 이후이므로 images는 PIL 객체 리스트임
        example["images"] = [img.resize((224, 224), Image.Resampling.LANCZOS) for img in example["images"]]
        return example

    # 1. 초기 구조 정의 (경로 기반)
    initial_features = Features({
        "images": Sequence(Value("string")),
        "sample_token": Value("string"),
        "prompt": [
            {
                "content": [{"text": Value("string"), "type": Value("string")}],
                "role": Value("string")
            }
        ],
        "completion": [
            {
                "content": [{"text": Value("string"), "type": Value("string")}],
                "role": Value("string")
            }
        ]
    })

    print("Creating Train Dataset (Flattened + Formatted)...")
    train_ds = Dataset.from_generator(gen_examples, gen_kwargs={"questions_json_path": train_questions_path}, features=initial_features)
    
    print("Creating Val Dataset (Flattened + Formatted)...")
    val_ds = Dataset.from_generator(gen_examples, gen_kwargs={"questions_json_path": val_questions_path}, features=initial_features)

    
    # 마지막에 이미지를 Pillow 객체로 캐스팅 (업로드 시 자동 인코딩됨)
    print("Casting 'images' column to Image features...")
    train_ds = train_ds.cast_column("images", Sequence(DatasetsImage()))
    val_ds = val_ds.cast_column("images", Sequence(DatasetsImage()))

    # 3. .map()을 이용해 병렬로 리사이즈 (224x224)
    print("Resizing all images to 224x224 (Parallel processing)...")
    train_ds = train_ds.map(resize_images, num_proc=32, desc="Resizing Train Images")
    val_ds = val_ds.map(resize_images, num_proc=32, desc="Resizing Val Images")

    ds_dict = DatasetDict({
        "train": train_ds,
        "validation": val_ds
    })
    
    print(f"Uploading to Hugging Face: {repo_id} ...")

    ds_dict.push_to_hub(repo_id)
    print("Dataset uploaded successfully.")
    return ds_dict

if __name__ == "__main__":    
    load_nuscenes_qa_to_hf()