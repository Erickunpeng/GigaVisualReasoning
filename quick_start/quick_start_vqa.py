import os
import random
import config
from src.subtyping.subtyping_evaluate import (
    get_svs_files_from_folders,
    save_results
)
from utils.file_utils import initialize_directories
from utils.file_utils import find_svs_file
from src.vqa.vqa_evaluate import process_vqa_slide

def get_random_sample_ids(txt_file="sample_ids.txt", n=5):
    with open(txt_file, "r") as f:
        sample_ids = [line.strip() for line in f.readlines()]
    if n > len(sample_ids):
        print(f"Warning: Requested {n} samples, but only {len(sample_ids)} available.")
        n = len(sample_ids)
    return random.sample(sample_ids, n) 

def main(sample_id, cancer_type="BRCA"):
    quick_start_output_dir = os.path.join(config.QUICK_START_DIR, "wsivqa_results")
    data_dir, output_path = initialize_directories(cancer_type, output_path=quick_start_output_dir)
    sample_svs_path = find_svs_file(sample_id, cancer_type)
    print(sample_svs_path)
    if not sample_svs_path:
        print(f"Error: SVS file for sample {sample_id} not found.")
        return
    print(f"Processing sample {sample_id}...")
    result = process_vqa_slide(sample_svs_path, cancer_type, output_path, True)
    if result:
        save_results([result], output_path, accuracy=None, f1_scores=None, macro_f1=None)
        print(f"VQA evaluation completed for sample {sample_id}. Results saved.")
    else:
        print(f"VQA evaluation failed for sample {sample_id}.")

if __name__ == "__main__":
    cancer_type = "BRCA"
    n = 5
    sample_ids = get_random_sample_ids("sample_ids.txt", n)
    for k, sample_id in enumerate(sample_ids, start=1):
        print(f"Processing {k}/{n}: Sample ID = {sample_id}")
        main(sample_id, cancer_type=cancer_type)
    # sample_id = "TCGA-BH-A42T"
    # main(sample_id, cancer_type=cancer_type)
