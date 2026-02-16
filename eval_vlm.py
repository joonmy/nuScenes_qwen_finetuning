import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from datasets import load_dataset
from tqdm import tqdm
import json
import os

def evaluate():
    model_id = "./output/qwen3_vl_8b_instruct_lora"
    base_model_id = "Qwen/Qwen3-VL-8B-Instruct"
    dataset_name = "JMandy/nuscenes_qa_mini_day_valid"
    
    print(f"Loading processor: {base_model_id}")
    processor = AutoProcessor.from_pretrained(base_model_id, trust_remote_code=True)
    
    print(f"Loading model: {model_id}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    )
    model.eval()

    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="validation") # push_to_hub defaults to 'train' if not specified otherwise

    results = []
    correct = 0
    total = 0

    print("Starting evaluation...")
    # Evaluate a subset or full dataset? Let's do full since it's "mini"
    for i, example in enumerate(tqdm(dataset)):
        images = example["images"]
        prompt = example["prompt"]
        ground_truth = example["completion"][0]["content"][0]["text"].strip().lower()

        # Prepare inputs for Qwen3-VL
        # Each image in prompt content has {"text": None, "type": "image"}
        # We need to replace these with the actual images for the processor
        
        # The prompt is already in conversational format.
        # However, the processor expectations might vary.
        # Usually: processor(text=[text], images=[images], return_tensors="pt")
        
        # Let's use the apply_chat_template if available
        text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        
        inputs = processor(
            text=[text],
            images=[images],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=50)
            
        # Extract the new tokens only
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip().lower()

        results.append({
            "question": prompt[0]["content"][-1]["text"].strip(),
            "answer": ground_truth,
            "prediction": output_text
        })

        if output_text == ground_truth or ground_truth in output_text:
            correct += 1
        total += 1

        if i < 5:
            print(f"\nSample {i}:")
            print(f"Q: {results[-1]['question']}")
            print(f"A: {results[-1]['answer']}")
            print(f"P: {results[-1]['prediction']}")

    accuracy = correct / total if total > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Save results
    with open("eval_results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "details": results
        }, f, indent=4)
    print("Results saved to eval_results.json")

if __name__ == "__main__":
    evaluate()
