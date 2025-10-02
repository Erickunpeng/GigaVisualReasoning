import os
import base64
import json
import config
import pandas as pd
from sksurv.metrics import concordance_index_censored
import numpy as np
from src.subtyping.slide_utils import get_oncotree_code
from utils.openai_client import get_openai_response_base64_with_multiple_images

def generate_few_shot_examples(base_dir, cancer_type):
    cancer_path = os.path.join(base_dir, cancer_type)
    few_shot_examples = []
    if os.path.isdir(cancer_path):
        for filename in os.listdir(cancer_path):
            if filename.endswith(".png"):
                survival_months = int(filename.split("_")[0])
                full_path = os.path.join(cancer_path, filename)
                few_shot_examples.append({
                    "roi_image": full_path,
                    "survival_months": survival_months
                })
    return few_shot_examples

def get_risk_level(months):
    if months <= 12:
        return 2  # High risk
    elif months <= 36:
        return 1  # Medium risk
    else:
        return 0  # Low risk


def generate_survival_prediction_prompt(cancer_type, few_shot_examples):
    prompt = (
        "You are an expert pathologist specializing in histological image analysis. "
        "Your task is to predict the risk level (0, 1, or 2) for a patient based on several provided region of interest (ROI) from a histopathology slide.\n\n"
        "The risk levels are defined as follows:\n"
        "- 0 = Low risk (long survival time, e.g., > 36 months)\n"
        "- 1 = Medium risk (moderate survival time, e.g., 12-36 months)\n"
        "- 2 = High risk (short survival time, e.g., < 12 months)\n\n"

        "**Instruction:**\n"
        "1. The first 3 images are example ROI images with known risk levels.\n"
        "2. The remaining images are new ROI images from a patient with "
        f"{cancer_type} cancer. Your task is to analyze all remaining ROIs and assign a risk level (0, 1, or 2) for this slide.\n"
        "3. Return only a SINGLE predicted risk level for all remaining images (except the frist three images). For example: 1\n"
        "4. ONLY return a SINGLE number from 0, 1, and 2, even though you are not sure. Do NOT include ANY other information in the response!\n\n"

        "**Few-shot Example Risk Levels:**\n"
    )
    for i, example in enumerate(few_shot_examples, 1):
        prompt += f"Example {i}: Risk Level = {get_risk_level(example['survival_months'])}\n"
    
    prompt += "\nNow analyze the remaining images (Starts at the fourth image) and return only the predicted risk level."
    return prompt

def get_survival_info(sample_id):
    df = pd.read_csv(config.META_DATA_DIR)
    result = df[df['Patient ID'] == sample_id[:12]]
    if result.empty:
        return None, None
    survival_months = result.iloc[0]['Overall Survival (Months)']
    survival_status = result.iloc[0]['Overall Survival Status']
    if pd.isna(survival_months) or pd.isna(survival_status):
        return None, None
    is_deceased = "DECEASED" in str(survival_status)
    return float(survival_months), is_deceased


def get_risk_levels(cancer_type, input_dir, mode, n=10):
    few_shot_examples = generate_few_shot_examples(
        os.path.join(config.ROOT_DIR, "data/survival_examples"), cancer_type)
    final_prompt = generate_survival_prediction_prompt(cancer_type, few_shot_examples)
    subtypes = config.CANCER_SUBTYPE_MAP.get(cancer_type) # eg. IDC, ILC
    pred_labels_dict = {subtype: [] for subtype in subtypes}
    event_times_dict = {subtype: [] for subtype in subtypes}
    event_indicators_dict = {subtype: [] for subtype in subtypes}
    sample_list = os.listdir(input_dir)
    few_shot_images = [example['roi_image'] for example in few_shot_examples]
    for idx, sample_folder in enumerate(sample_list):
        sample_path = os.path.join(input_dir, sample_folder)
        if not os.path.isdir(sample_path):
            continue
        sample_id = sample_folder[:12]
        sample_subtype = get_oncotree_code(sample_id)
        if sample_subtype not in subtypes:
            continue  # skip

        roi_image_paths = [
            os.path.join(sample_path, f)
            for f in os.listdir(sample_path)
            if f.endswith(".png") and "overview" not in f.lower()
        ]
        if mode == "random" or mode == "gpt":
            roi_image_paths = [
                os.path.join(sample_path, f)
                for f in os.listdir(sample_path)
                if f.startswith(f"{mode}") and f.endswith(".png") and "overview" not in f.lower()
            ]
        roi_image_paths = few_shot_images + roi_image_paths

        try:
            response = get_openai_response_base64_with_multiple_images(final_prompt, roi_image_paths)
            if not response:
                raise ValueError("Empty response")
            pred_label = int(response)
        except Exception as e:
            print(f"[ERROR] Failed on {sample_id}: {e}")
            continue
        true_label, event_indicator = get_survival_info(sample_id)
        labels = [0, 1, 2]
        if pred_label in labels and (true_label is not None) and (event_indicator is not None):
            pred_labels_dict[sample_subtype].append(pred_label)
            event_times_dict[sample_subtype].append(true_label)
            event_indicators_dict[sample_subtype].append(event_indicator)
            print(f"Sample processed {idx+1} / {len(sample_list)}")
        if idx > n and n != -1:
            break
    return pred_labels_dict, event_times_dict, event_indicators_dict

def compute_c_index(pred_labels, event_times, event_indicators):
    assert len(event_times) == len(event_indicators) == len(pred_labels), "Input lengths do not match!"
    c_index, *_ = concordance_index_censored(
        event_indicator=np.array(event_indicators),
        event_time=np.array(event_times),
        estimate=np.array(pred_labels)
    )
    return c_index

def save_prediction_results(cancer_type, subtype, output_dir, pred_labels, event_times, event_indicators, n, mode):
    output_path = os.path.join(output_dir, f"{cancer_type}_{subtype}_{mode}_risk_predictions_{n}.csv")
    df = pd.DataFrame({
        "predicted_label": pred_labels,
        "event_time": event_times,
        "event_indicator": event_indicators
    })
    df.to_csv(output_path, index=False)
    print(f"[{subtype}] Prediction results saved to {output_path}")


if __name__ == "__main__":
    cancer_type = "BRCA"
    mode = "gpt"
    n = -1
    input_dir = ""
    if mode == "roiagent":
        input_dir = os.path.join(config.QUICK_START_DIR, cancer_type, "roi_output")
    elif mode == "random":
        input_dir = os.path.join(config.OUTPUT_DIR, "subtyping", cancer_type, "majority_vote_baseline")
    elif mode == "gpt":
        input_dir = os.path.join(config.OUTPUT_DIR, "subtyping", cancer_type, "baseline_output")

    output_dir = os.path.join(config.OUTPUT_DIR, "survival", cancer_type, f"{mode}_baseline")
    os.makedirs(output_dir, exist_ok=True)

    pred_labels_dict, event_times_dict, event_indicators_dict = get_risk_levels(cancer_type, input_dir, mode, n)
    for subtype in pred_labels_dict:
        preds = pred_labels_dict[subtype]
        times = event_times_dict[subtype]
        indicators = event_indicators_dict[subtype]
        if len(preds) == 0:
            print(f"[{subtype}] No valid samples found, skipping...")
            continue
        c_index = compute_c_index(preds, times, indicators)
        print(f"[{subtype}] Processed {len(preds)} samples. Concordance Index (C-index): {c_index:.4f}")
        save_prediction_results(cancer_type, subtype, output_dir, preds, times, indicators, n, mode)


