import os
import json
import textwrap
import pickle
import config
from nltk.translate.bleu_score import SmoothingFunction
import src.report.report_utils as utils

def extract_text_from_textract(file_path, wrap_width=80):
    with open(file_path, "rb") as file:
        data = pickle.load(file)  # Load Amazon Textract JSON from .p file
    lines = []
    for block in data.get("Blocks", []):
        if block["BlockType"] == "LINE":  # Only extract meaningful lines
            text = block["Text"].strip()
            if text:
                lines.append(text)
    formatted_text = "\n".join(textwrap.fill(line, wrap_width) for line in lines)    
    return formatted_text

def get_reference_text(groundtruth_folder, sample_id):
    groundtruth_sample_files = [
        os.path.join(groundtruth_folder, f)
        for f in os.listdir(groundtruth_folder)
        if f.startswith(sample_id)
    ]
    if not groundtruth_sample_files:
        print(f"Groundtruth not found for sample {sample_id}. Skipping.")
        return None
    reference_text = ""
    for file_path in sorted(groundtruth_sample_files):
        reference_text += extract_text_from_textract(file_path) + "\n"
    return reference_text

def save_reference_text(sample_id, reference_text):
    output_dir = os.path.join(config.ROOT_DIR, "tests/examples")
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{sample_id}.txt")
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(reference_text.strip())


def compare_reports(candidate_folder, groundtruth_folder, overwrite, mode="roiagent"):
    # Iterate through the candidate folder
    os.makedirs(candidate_folder, exist_ok=True)
    if mode == "baseline1":
      for sample_dir in os.listdir(candidate_folder):
        sample_path = os.path.join(candidate_folder, sample_dir)
        if not os.path.isdir(sample_path):
            continue
        sample_id = sample_dir[:12]
        for subfolder in os.listdir(sample_path):  # 1 to 21
            sub_path = os.path.join(sample_path, subfolder)
            if not os.path.isdir(sub_path):
                continue
            report_name = mode
            comparison_path = os.path.join(sub_path, f"{report_name}_comparison.json")
            if os.path.exists(comparison_path) and not overwrite:
                print(f"Comparison already exists for {sample_id}/{subfolder}. Skipping.")
                continue
            report_path = os.path.join(sub_path, f"{report_name}report.txt")
            if not os.path.exists(report_path):
                print(f"Report not found for {sample_id}/{subfolder}. Skipping.")
                continue
            with open(report_path, "r", encoding="utf-8") as file:
                candidate_text = file.read().strip()
            reference_text = get_reference_text(groundtruth_folder, sample_id)
            if reference_text is None:
                continue
            save_reference_text(sample_id, reference_text)
            rouge_scores = utils.calculate_rouge(reference_text, candidate_text)
            gpt_eval_score = utils.calculate_gpt_eval_score(reference_text, candidate_text)
            comparison_result = {
                "sample_id": sample_id,
                "gpt_eval_score": gpt_eval_score,
                "rouge_scores": rouge_scores,
                "reference_text": reference_text,
                "candidate_text": candidate_text,
            }
            comparison_path = os.path.join(config.OUTPUT_DIR, "report", cancer_type, "vote_baseline_output", sample_dir, subfolder)
            os.makedirs(comparison_path, exist_ok=True)
            with open(os.path.join(comparison_path, f"{report_name}_comparison.json"), "w", encoding="utf-8") as f:
                json.dump(comparison_result, f, indent=4)
            print(f"Saved comparison for {sample_id}/{subfolder}")
    else:
        for sample_dir in os.listdir(candidate_folder):
            sample_path = os.path.join(candidate_folder, sample_dir)
            if not os.path.isdir(sample_path):
                continue
            sample_id = sample_dir[:12]
            report_name = mode if mode in ["baseline1", "baseline2"] else ""
            comparison_path = os.path.join(sample_path, f"{report_name}_comparison.json")
            if os.path.exists(comparison_path) and not overwrite:
                print(f"Comparison already exists for sample {sample_id}. Skipping.")
                continue
            candidate_path = os.path.join(sample_path, f"{report_name}report.txt")
            if not os.path.exists(candidate_path):
                print(f"Candidate report not found for sample {sample_id}. Skipping.")
                continue
            with open(candidate_path, "r", encoding="utf-8") as file:
                candidate_text = file.read().strip()

            reference_text = get_reference_text(groundtruth_folder, sample_id)
            if reference_text == None:
                continue
            save_reference_text(sample_id, reference_text)

            # Calculate BLEU and ROUGE scores
            # bleu_score = calculate_bleu(reference_text, candidate_text)
            rouge_scores = utils.calculate_rouge(reference_text, candidate_text)
            gpt_eval_score = utils.calculate_gpt_eval_score(reference_text, candidate_text)

            # Save comparison results
            comparison_result = {
                "sample_id": sample_id,
                # "bleu_score": bleu_score,
                "gpt_eval_score": gpt_eval_score,
                "rouge_scores": rouge_scores,
                "reference_text": reference_text,
                "candidate_text": candidate_text,
            }

            # Save the comparison.json in the candidate sample folder
            comparison_path = os.path.join(sample_path, f"{report_name}_comparison.json")
            groundtruth_path = os.path.join(sample_path, "groundtruth.txt")
            with open(comparison_path, "w", encoding="utf-8") as file:
                json.dump(comparison_result, file, indent=4)

            print(f"Comparison results saved for sample {sample_id}: {comparison_path}")


def main(cancer_type, overwrite=True, mode="roiagent"):
    groundtruth_folder = config.VQA_META_DATA_DIR
    candidate_folder = ""
    if mode == "roiagent":
        candidate_folder = os.path.join(config.QUICK_START_DIR, cancer_type, "report_output")
    elif mode == "baseline2":
        candidate_folder = os.path.join(config.OUTPUT_DIR, "report", cancer_type, "baseline_output")
    elif mode == "baseline1":
        candidate_folder = os.path.join(config.OUTPUT_DIR, "report", cancer_type, "vote_baseline_output")
    else:
        raise ValueError(f"Invalid mode: {mode}")
    compare_reports(candidate_folder, groundtruth_folder, overwrite, mode=mode)


if __name__ == "__main__":
    cancer_type = "HEP"
    overwrite = False
    mode = "baseline1"
    main(cancer_type, overwrite=overwrite, mode=mode)