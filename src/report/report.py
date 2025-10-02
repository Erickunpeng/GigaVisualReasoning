import os
import base64
import config
from utils.openai_client import get_openai_response_base64, get_openai_response_base64_with_multiple_images
import src.report.report_prompt as prompt
import random

def get_selected_samples(input_dir, n):
    # input_dir = os.path.join("output/subtyping", "BRCA", "majority_vote_baseline")
    all_samples = [sample_id for sample_id in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, sample_id))]
    if n == -1:
        selected_samples = all_samples  # Process all samples
    elif n > len(all_samples):
        raise ValueError(f"Requested {n} samples, but only {len(all_samples)} available.")
    else:
        selected_samples = random.sample(all_samples, n)  # Randomly select 'n' samples
    return selected_samples

def generate_reports_for_samples(input_dir, output_dir, sample_id, cancer_type, overwrite, n, n_iter, mode="roiagent"):
    os.makedirs(output_dir, exist_ok=True)
    if mode == "baseline1":
        sample_path = os.path.join(input_dir, sample_id, n_iter)
    else:
        sample_path = os.path.join(input_dir, sample_id)
    if not os.path.isdir(sample_path):
        return

    report_name = mode if mode in ["baseline1", "baseline2"] else ""
    if mode == "baseline1":
        sample_output_dir = os.path.join(output_dir, sample_id, n_iter)
    else:
        sample_output_dir = os.path.join(output_dir, sample_id)
    report_path = os.path.join(sample_output_dir, f"{report_name}report.txt")
    if os.path.exists(report_path) and not overwrite:
        print(f"Report already exists for sample {sample_id}. Skipping.")
        return

    png_files = []
    if mode == "roiagent":
        png_files = [
            os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.endswith(".png")
        ]
    elif mode == "baseline1":
        png_files = [
            os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.startswith("random") and f.endswith(".png")
        ]
    elif mode == "baseline2":
        png_files = [
            os.path.join(sample_path, f) for f in os.listdir(sample_path) if f.startswith("gpt") and f.endswith(".png")
        ]
    
    if not png_files:
        print(f"No PNG files found for sample {sample_id}. Skipping.")
        return

    vqa_prompt = prompt.generate_scientific_report(cancer_type)

    image_paths = png_files[:3]

    if len(image_paths) == 1:
        report_content = get_openai_response_base64(vqa_prompt, image_paths[0])
    else:
        report_content = get_openai_response_base64_with_multiple_images(vqa_prompt, image_paths)

    if report_content is None:
        print(f"Failed to generate report for sample {sample_id}.")
        return

    os.makedirs(sample_output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write(report_content)

    print(f"Report generated and saved for sample {sample_id}: {report_path}")


def main(cancer_type, n, overwrite=True, mode="roiagent"):
    if mode == "roiagent":
        input_directory = os.path.join(config.QUICK_START_DIR, cancer_type, "roi_output")
        output_directory = os.path.join(config.QUICK_START_DIR, cancer_type, "report_output")
    elif mode == "baseline2":
        input_directory = os.path.join(config.OUTPUT_DIR, "subtyping", cancer_type, "baseline_output")
        output_directory = os.path.join(config.OUTPUT_DIR, "report", cancer_type, "baseline_output")
    elif mode == "baseline1":
        input_directory = os.path.join(config.OUTPUT_DIR, "subtyping", cancer_type, "majority_vote_baseline")
        output_directory = os.path.join(config.OUTPUT_DIR, "report", cancer_type, "vote_baseline_output")  
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    selected_samples = get_selected_samples(input_directory, n)
    for sample_id in selected_samples:
        if mode == "baseline1":
            for i in range(21):
                generate_reports_for_samples(input_directory, output_directory, sample_id, cancer_type, overwrite, n, str(i+1), mode=mode)
        else:
            generate_reports_for_samples(input_directory, output_directory, sample_id, cancer_type, overwrite, n, 0, mode=mode)

if __name__ == "__main__":
    cancer_type = "HEP"
    num_samples = -1
    mode = "baseline1"
    main(cancer_type, num_samples, mode=mode)