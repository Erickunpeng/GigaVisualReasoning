import os
import json
import random
import openslide
import numpy as np
from src.subtyping.subtyping_baseline import process_random_roi, process_gpt_selected_roi
from src.vqa.vqa_evaluate import evaluate_vqa
from src.vqa.questions import get_vqa_for_sample
from src.subtyping.roi_agent import ROIAgent
import src.subtyping.slide_utils
import config
from src.vqa.questions import extract_all_sample_id, get_selected_svs_files
import src.subtyping.subtyping_prompt as prompt
from utils.openai_client import get_openai_response_text_only
from utils.file_utils import initialize_directories, get_svs_files_from_folders

def process_slide(file_path, cancer_type, output_path, baseline_type, final_prompt):
    file_name = os.path.basename(file_path)
    sample_id = os.path.basename(file_name).split('.')[0]
    image = openslide.OpenSlide(file_path)
    if baseline_type == "random":
        result = process_random_roi(image, sample_id, cancer_type, output_path, final_prompt)
    elif baseline_type == "gpt":
        result = process_gpt_selected_roi(image, sample_id, cancer_type, output_path, final_prompt)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
    return result

def process_vqa_slide(file_path, cancer_type, output_path, baseline_type):
    file_name = os.path.basename(file_path)
    sample_id = os.path.basename(file_name)[:12]
    vqa_questions = get_vqa_for_sample(sample_id)
    if not vqa_questions:
        print(f"Skipping {sample_id}: No VQA questions found.")
        return None
    messages = [
        {
            "role": "user",
            "content": f"Q: {q['question']}\nChoices: {', '.join(q['choices'])}"
        }
        for q in vqa_questions
    ]
    final_prompt = prompt.get_final_prompt(cancer_type, "vqa", messages)
    # print(final_prompt)
    slide_result = process_slide(file_path, cancer_type, output_path, baseline_type, final_prompt)
    if slide_result is None:
        print(f"Skipping {sample_id}: No valid ROI found.")
        return None
    sample_id, analysis_result, output_path, baseline_type = slide_result
    if isinstance(analysis_result, str):
        analysis_result = [ans.strip() for ans in analysis_result.split(",")]
    evaluation_result = evaluate_vqa(sample_id, vqa_questions, analysis_result)
    save_vqa_baseline_result(sample_id, evaluation_result, output_path, baseline_type)
    return evaluation_result

def save_vqa_baseline_result(sample_id, evaluation_result, output_path, baseline_type):
    sample_result_path = os.path.join(output_path, sample_id)
    os.makedirs(sample_result_path, exist_ok=True)
    result_file = os.path.join(sample_result_path, f"{baseline_type}_baseline_result.json")
    with open(result_file, "w") as f:
        json.dump(evaluation_result, f, indent=4)
    print(f"VQA Results saved for {sample_id} in {result_file}")


def main(cancer_type, baseline_type="random", n=5):
    output_dir = os.path.join(config.OUTPUT_DIR, "wsivqa_baseline_results")
    base_path, output_path = initialize_directories(cancer_type, output_path=output_dir)
    svs_files = []
    with open("sample_id_path.txt", "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1) 
            if len(parts) == 2:
                svs_files.append(parts[1])
    total_files = len(svs_files)
    results = []
    for idx, file_name in enumerate(svs_files, start=1):
        try:
            result = process_vqa_slide(file_name, cancer_type, output_path, baseline_type)
            if result:
                results.append(result)
            print(f"Processed {idx}/{total_files}: {file_name}")
        except Exception as e:
            print(f"ERROR processing {file_name}: {e}")
            continue
    results_file = os.path.join(output_path, f"{baseline_type}_baseline_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    cancer_type = "BRCA"
    baseline_type = "gpt"  # random / gpt
    num_samples = -1
    main(cancer_type, baseline_type, num_samples)
