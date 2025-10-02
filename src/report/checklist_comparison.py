import os
import json
import config
import random
from src.report.report_prompt import generate_checklist_prompt
from utils.openai_client import get_openai_response_text_only
from utils.file_utils import initialize_directories

def extract_gpt_answers(response):
    try:
        answers = json.loads(response.strip())
        if isinstance(answers, list):
            return answers
        else:
            print("Error: GPT response is not a valid list.")
            return None
    except json.JSONDecodeError:
        print("Error: GPT response is not in valid JSON format.")
        return None

def compare_reports(reference_text, candidate_text, cancer_type,):
    vqa_file = os.path.join("data/eval_questions", f"{cancer_type}_eval_vqa.json")
    with open(vqa_file, "r", encoding="utf-8") as f:
        vqa_questions = json.load(f)
    prompt = generate_checklist_prompt(reference_text, candidate_text, vqa_questions)
    response = get_openai_response_text_only(prompt)
    incorrect_questions = []
    correct_questions = []
    answers = None
    try:
        answers = json.loads(response.strip())
        if not isinstance(answers, list):
            print("Error: GPT response is not a valid list.")
            return None
    except json.JSONDecodeError:
        print("Error: GPT response is not in valid JSON format.")
        return None

    # Compare with ground truth answers
    total_questions = len(answers)
    for i, answer in enumerate(answers):
        question_id = vqa_questions[i]["ID"]
        if answer == 0:  
            correct_questions.append(question_id)
        else:  
            incorrect_questions.append(question_id)

    correct_count = answers.count(0)  # Count the number of correct (0) responses
    accuracy = correct_count / total_questions

    return {
        "accuracy": accuracy,
        "total_questions": total_questions,
        "correct_count": correct_count,
        "incorrect_questions": incorrect_questions,
        "correct_questions": correct_questions
    }

def main(input_path, cancer_type, n_samples, mode):
    file_name = "_comparison.json"
    if mode == "baseline1" or mode == "baseline2":
        file_name = f"{mode}_comparison.json"
    comparison_files = []
    if mode == "baseline1":
        for sample_dir in os.listdir(input_path):
            sample_path = os.path.join(input_path, sample_dir)
            if not os.path.isdir(sample_path):
                continue
            for subfolder in map(str, range(1, 22)):  # 1 ~ 21
                sub_path = os.path.join(sample_path, subfolder)
                if not os.path.isdir(sub_path):
                    continue
                comp_file = os.path.join(sub_path, f"{mode}_comparison.json")
                if os.path.exists(comp_file):
                    comparison_files.append(comp_file)
    else:
        for root, _, files in os.walk(input_path):
            for file in files:
                if file == file_name:
                    comparison_files.append(os.path.join(root, file))
    if not comparison_files:
        print(f"No comparison files found in {input_path}.")
        return
    if n_samples > 0:
        comparison_files = random.sample(comparison_files, min(n_samples, len(comparison_files)))
    print(f"Processing {len(comparison_files)} comparison files...")

    results = []
    for file in comparison_files:
        file_path = os.path.join(input_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        sample_id = data.get("sample_id", "UNKNOWN")
        reference_text = data.get("reference_text", "").strip()
        candidate_text = data.get("candidate_text", "").strip()
        if not reference_text or not candidate_text:
            print(f"Skipping {sample_id}, missing reference or candidate text.")
            continue

        evaluation_result = compare_reports(reference_text, candidate_text, cancer_type)
        if evaluation_result:
            evaluation_result["sample_id"] = sample_id
            sample_output_file = os.path.join(os.path.dirname(file_path), f"{mode}_checklist_eval.json")
            with open(sample_output_file, "w", encoding="utf-8") as f:
                json.dump(evaluation_result, f, indent=4)
            results.append(evaluation_result)

    # Save results to checklist_eval.json in the same directory
    output_file = os.path.join(input_path, f"{mode}_checklist_eval.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Checklist evaluation completed. Results saved to {output_file}.")



if __name__ == "__main__":
    mode = "baseline1" # roiagent / baseline1 / baseline2
    cancer_type = "UTERUS"
    input_path = ""
    if mode == "roiagent":
        input_path = os.path.join(config.QUICK_START_DIR, cancer_type, "report_output")
    elif mode == "baseline2":
        input_path = os.path.join(config.OUTPUT_DIR, "report", cancer_type, "baseline_output")
    elif mode == "baseline1":
        input_path = os.path.join(config.OUTPUT_DIR, "report", cancer_type, "vote_baseline_output")
    n_samples = -1
    main(input_path, cancer_type, n_samples, mode)

