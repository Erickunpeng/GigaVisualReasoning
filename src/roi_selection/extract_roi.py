import os
import re
import numpy as np
import random
import config
import openslide

from src.subtyping.roi_agent import ROIAgent
from utils.openai_client import azure_config_list
from src.subtyping.slide_utils import draw_bbox_on_overview_roi_all_tasks
from src.vqa.questions import get_vqa_for_sample
from src.subtyping.subtyping_prompt import get_iteration_messages
from utils.file_utils import get_svs_files_from_folders, find_svs_file


def extract_roi_with_query(file_path, task_type, cancer_type, output_dir):
    file_name = os.path.basename(file_path)
    sample_id = os.path.basename(file_name).split('.')[0]
    sample_output_dir = os.path.join(output_dir, sample_id)
    os.makedirs(sample_output_dir, exist_ok=True)
    image = openslide.OpenSlide(file_path)
    roi_agent = ROIAgent(
        image=image,
        cancer_type=cancer_type,
        name="ROI Agent",
        llm_config={"config_list": [azure_config_list[0]], "max_tokens": 3000},
        n_iters=config.NUM_ITER,
        task=task_type,
        to_predict=False
    )
    roi_agent.working_dir = sample_output_dir
    roi_agent.sample_id = sample_id
    messages = get_reply_messages(task_type, cancer_type, sample_id[:12])
    if not messages:
        raise ValueError("Invalid query messages!")
    roi_agent._reply_user(messages=messages)
    return roi_agent.final_bbox_info, roi_agent.overview_image

def get_reply_messages(task_type, cancer_type, sample_id):
    if task_type == "subtyping":
        return get_iteration_messages(cancer_type)
    elif task_type == "survival":
        return [{"content": f"What is the survival risk level for this {cancer_type} cancer image? "
        "Select from the following options: Low (>36 months), Intermediate (12-36 months), or High (<12 months)."}]
    elif task_type == "report":
        return [{"content": f"Generate a concise pathology report for this {cancer_type} cancer slide. "
        "Select an ROI that best captures diagnostic features such as tumor architecture, cellular morphology, and relevant markers."}]
    elif task_type == "vqa":
        vqa_questions = get_vqa_for_sample(sample_id)
        if not vqa_questions:
            print(f"Skipping {sample_id}, no VQA questions found.")
            return None
        return [
            {
                "role": "user",
                "content": f"Q: {q['question']}\nChoices: {', '.join(q['choices'])}"
            }
            for q in vqa_questions
        ]
    
def iterating_all_tasks(tasks, cancer_type, sample_id=""):
    bbox_info_list = []
    for task_type in tasks:
        output_dir = os.path.join("roi_selection_output", task_type, cancer_type)
        sample_path = find_svs_file(sample_id, cancer_type)
        if not sample_path:
            print("The given sample id is not valid!")
        else:
            final_bbox_info, overview_image = extract_roi_with_query(sample_path, task_type, cancer_type, output_dir)
            bbox_info_list.append(final_bbox_info)
            print(f"Processed {sample_id} for {task_type}")
    save_path = os.path.join("roi_selection_output", f"{sample_id}_all_tasks.png")
    colors = ["blue", "red", "black", "orange"]
    draw_bbox_on_overview_roi_all_tasks(overview_image, bbox_info_list, save_path, colors)
    print(f"Successfully generates all task ROIs for {cancer_type} {sample_id}")
    

def iterating_single_task(task_type, cancer_type, n=2):
    output_dir = os.path.join("roi_selection_output", task_type, cancer_type)
    svs_files = get_svs_files_from_folders(config.CANCER_FOLDER_MAP, cancer_type)
    if n > 0:
        svs_files = random.sample(svs_files, min(n, len(svs_files)))
    total_files = len(svs_files)
    for idx, file_name in enumerate(svs_files, start=1):
        extract_roi_with_query(file_name, task_type, cancer_type, output_dir)
        print(f"Processed {idx}/{total_files}: {file_name}")

if __name__ == "__main__":
    cancer_type = "RCC"
    is_single_tasks = True
    if is_single_tasks:
        task_type = "subtyping"
        n = 1
        iterating_single_task(task_type, cancer_type, n=n)
    else: # Manually input a specific WSI
        tasks = ["subtyping", "report", "survival", "vqa"]
        sample_id = "TCGA-A2-A0YK-01Z-00-DX1"
        iterating_all_tasks(tasks, cancer_type, sample_id=sample_id)