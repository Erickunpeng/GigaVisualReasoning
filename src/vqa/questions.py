import os
import json
from collections import defaultdict
import config
from utils.file_utils import find_svs_file
import random

def get_vqa_for_sample(sample_id):
    data_dir = os.path.join(config.ROOT_DIR, "data/question_list")
    file_names = ["WsiVQA_train.json", "WsiVQA_val.json", "WsiVQA_test.json"]
    vqa_data = []
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                if item["Id"] == sample_id:
                    if "Question" not in item or "Choice" not in item:
                        # print(f"Warning: Missing 'Question' or 'Choice' in sample {sample_id}: {item}")
                        continue
                    question_text = item["Question"]
                    answer_text = item["Answer"]
                    choices = item["Choice"] if isinstance(item["Choice"], list) else []
                    vqa_data.append({
                        "question": question_text,
                        "choices": choices,
                        "answer": answer_text
                    })
    return vqa_data

def extract_all_sample_id(num_samples=-1):
    data_dir = os.path.join(config.ROOT_DIR, "data/question_list")
    file_names = ["WsiVQA_train.json", "WsiVQA_val.json", "WsiVQA_test.json"]
    sample_ids = set()
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                if "Id" in item:
                    sample_ids.add(item["Id"])
                if num_samples != -1 and len(sample_ids) >= num_samples:
                    print(f"Reached extraction limit: {num_samples} samples.")
                    return sample_ids
        print(f"Added from {file_name}")
    print(f"Total unique sample IDs extracted: {len(sample_ids)}")
    return sample_ids

def get_selected_svs_files(cancer_type, n=-1):
    svs_files = []
    sample_ids = extract_all_sample_id(num_samples=n)
    print(f"DEBUG: Extracted {len(sample_ids)} unique sample IDs")
    for sample_id in sample_ids:
        try:
            svs_path = find_svs_file(sample_id, cancer_type)
            if svs_path:
                svs_files.append(svs_path)
        except Exception as e:
            print(f"ERROR finding SVS file for {sample_id}: {e}")
            continue
    # print(svs_files[:5])
    if n > 0:
        selected_samples = random.sample(svs_files, min(n, len(svs_files)))
        svs_files = selected_samples
        print(f"Selected {len(selected_samples)} random samples for evaluation.")
    return svs_files

if __name__ == "__main__":
    sample_id = "TCGA-A2-A0YK"
    vqa_results = get_vqa_for_sample(sample_id)
    if vqa_results:
        print(f"\nVQA for Sample ID: {sample_id}")
        for qa in vqa_results:
            print(f"  - {qa['question']}")
            print(f"    Answer: {qa['answer']}")
    else:
        print(f"No VQA data found for Sample ID: {sample_id}.")
