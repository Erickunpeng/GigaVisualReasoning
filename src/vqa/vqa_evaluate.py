import os
import json
import config
import openslide
import signal
import multiprocessing
import random
from src.subtyping.roi_agent import ROIAgent
from utils.openai_client import azure_config_list
from src.subtyping.subtyping_evaluate import (
    save_results
)
from utils.file_utils import initialize_directories
from src.vqa.questions import get_vqa_for_sample, extract_all_sample_id, get_selected_svs_files
from utils.file_utils import find_svs_file, get_svs_files_from_folders

def timeout_handler(signum, frame):
    raise TimeoutError("ROI Agent response timeout.")

def process_vqa_slide(file_path, cancer_type, output_path, overwrite):
    file_name = os.path.basename(file_path)
    sample_id = os.path.basename(file_name)[:12]
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(600)

    try:
        vqa_questions = get_vqa_for_sample(sample_id)
        if not vqa_questions:
            print(f"Skipping {sample_id}, no VQA questions found.")
            return None
        sample_output_dir = os.path.join(output_path, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)
        if not overwrite:
            result_path = os.path.join(sample_output_dir, "result.json")
            if os.path.exists(result_path):
                print(f"result.json already exists for {sample_id}.")
                return None
        image = openslide.OpenSlide(file_path)
        # Perform VQA using ROI Agent
        roi_agent = ROIAgent(
            image=image,
            cancer_type=cancer_type,
            name="VQA ROI Agent",
            llm_config={"config_list": [azure_config_list[0]], "max_tokens": 3000},
            n_iters=10,
            mode="multiple",
            task="vqa"
        )
        roi_agent.working_dir = sample_output_dir
        roi_agent.sample_id = sample_id
        messages = [
            {
                "role": "user",
                "content": f"Q: {q['question']}\nChoices: {', '.join(q['choices'])}"
            }
            for q in vqa_questions
        ]
        analysis_result = roi_agent._reply_user(messages=messages)
        signal.alarm(0)
        evaluation_result = evaluate_vqa(sample_id, vqa_questions, roi_agent.result)
        result_file = os.path.join(sample_output_dir, "result.json")
        with open(result_file, "w") as f:
            json.dump(evaluation_result, f, indent=4)

        return evaluation_result
    except TimeoutError:
        print(f"WARNING: Sample {sample_id} timed out. Skipping.")
        return None

def evaluate_vqa(sample_id, vqa_questions, analysis_result):
    correct_count = 0
    total_questions = len(vqa_questions)
    total_predictions = len(analysis_result)
    detailed_results = []
    if total_predictions < total_questions:
        analysis_result += ["N/A"] * (total_questions - total_predictions)  # Fill "N/A"
    elif total_predictions > total_questions:
        analysis_result = analysis_result[:total_questions]

    for i, qa in enumerate(vqa_questions):
        question_text = qa["question"]
        expected_answer = qa["answer"]
        predicted_answer = analysis_result[i] 
        is_correct = predicted_answer.lower() == expected_answer.lower()
        detailed_results.append({
            "question": question_text,
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        })
        if is_correct:
            correct_count += 1
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    return {
        "sample_id": sample_id,
        "accuracy": accuracy,
        "details": detailed_results
    }

def retry_evaluation(file_path, cancer_type, output_path, max_retries=5, accuracy_threshold=0.3, overwrite=False):
    for attempt in range(max_retries):
        print(f"Processing {file_path} (Attempt {attempt + 1}/{max_retries})")
        evaluation_result = process_vqa_slide(file_path, cancer_type, output_path, overwrite)
        if evaluation_result is None:
            return None
        accuracy = evaluation_result.get("accuracy", 0)
        if accuracy >= accuracy_threshold:
            return evaluation_result
        print(f"Accuracy {accuracy:.2f} below threshold {accuracy_threshold}. Retrying...")
    print(f"Max retries reached for {file_path}. Skipping.")
    return None

def main(cancer_type, n=2, num_workers=15, accuracy_threshold=0.3, max_retries=5, overwrite=False):
    vqa_output_dir = os.path.join(config.OUTPUT_DIR, "wsivqa_results")
    data_dir, output_path = initialize_directories(cancer_type, output_path=vqa_output_dir)
    svs_files = []
    with open("sample_id_path.txt", "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1) 
            if len(parts) == 2:
                svs_files.append(parts[1])
    results = []
    if n > 0:
        svs_files = random.sample(svs_files, min(n, len(svs_files)))
    total_files = len(svs_files)
    num_workers = min(num_workers, len(svs_files)) if svs_files else 1
    print(f"Processing {total_files} SVS files using {num_workers} workers...")
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.starmap(
            retry_evaluation, [(file, cancer_type, output_path, max_retries, accuracy_threshold, overwrite) for file in svs_files]
        )
    results = [r for r in results if r is not None]
    evaluated_samples = len(results)
    print(f"Total Evaluated Samples: {evaluated_samples}/{total_files}")
    save_results(results, output_path, accuracy=None, f1_scores=None, macro_f1=None)

if __name__ == "__main__":
    cancer_type = "BRCA"
    num_samples = -1
    num_workers = 15
    accuracy_threshold = 0.4
    max_retries = 5
    main(cancer_type, n=num_samples, num_workers=num_workers, accuracy_threshold=accuracy_threshold, max_retries=max_retries)
