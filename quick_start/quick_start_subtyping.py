import os
import json
import random
import config
import multiprocessing
from src.subtyping import subtyping_prompt as prompt
from src.subtyping.subtyping_evaluate import process_slide, save_results
from utils.file_utils import get_svs_files_from_repo, get_svs_files_from_folders

def process_sample(svs_path, cancer_type, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    sample_id = os.path.basename(svs_path).split('.')[0]
    result_file = os.path.join(output_dir, sample_id, "result.json")
    if os.path.exists(result_file):
        print(f"Skipping {sample_id}, result already exists.")
        return None
    messages = prompt.get_iteration_messages(cancer_type)
    try:
        result = process_slide(svs_path, cancer_type, output_dir, messages)
    except Exception as e:
        print(f"[ERROR] Failed to process {svs_path}: {e}")
        result = None
    return result

def main(cancer_type="BRCA", n_samples=10, mode="multiple", num_workers=10):
    random.seed(80)
    # output_dir = os.path.join("/data/TCGA-Demo/output/subtyping", cancer_type, f"{config.NUM_ITER}", "roi_output")
    output_dir = os.path.join(config.QUICK_START_DIR, cancer_type, "roi_output")
    svs_files = get_svs_files_from_folders(config.CANCER_FOLDER_MAP, cancer_type)
    # if cancer_type == "HEP":
    #     svs_files = get_svs_files_from_repo("TCGA-CHOL")
    #     print(len(svs_files))
    if not svs_files:
        print(f"No SVS files found for cancer type: {cancer_type}")
        return
    if n_samples > 0:
        selected_samples = random.sample(svs_files, min(n_samples, len(svs_files)))
    else:
        selected_samples = svs_files
    print(f"Selected {len(selected_samples)} random samples for evaluation.")
    with multiprocessing.Pool(processes=min(num_workers, len(selected_samples))) as pool:
        results = pool.starmap(process_sample, [(svs_path, cancer_type, output_dir) for svs_path in selected_samples])
    results = [res for res in results if res is not None]
    save_results(results, output_dir, accuracy=None, f1_scores=None, macro_f1=None)  # Quick start doesn't need F1 metrics

if __name__ == "__main__":
    cancer_type = "BRCA"
    n_samples = 1
    num_workers = 5
    main(cancer_type, n_samples=n_samples, num_workers=num_workers)
