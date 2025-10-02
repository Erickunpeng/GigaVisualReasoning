import os
import config
from utils.openai_client import get_openai_response_text_only

def generate_scientific_report(cancer_type):
    examples = read_cancer_examples_as_text(cancer_type)
    prompt = (
        f"Generate a comprehensive pathology report for a patient diagnosed with {cancer_type}.\n"
        "This report should be professional, medically accurate, and well-structured.\n\n"
        "You are provided with some real-world pathology reports as reference.\n"
        "These examples are meant to offer insights into the type of information that might be included in such reports,\n"
        "but you do NOT need to follow their format or structure exactly. Instead, generate a report using your own medical knowledge,\n"
        "ensuring it is detailed, logical, and clinically relevant.\n\n"
        "Here are some pathology report examples for reference:\n\n"
    )
    prompt += examples
    prompt += (
        "\nNow, based on your medical expertise, generate a detailed pathology report for a patient with the same cancer type.\n"
        "Make sure your report is structured professionally, with accurate clinical descriptions and findings."
    )
    return prompt

def read_cancer_examples_as_text(cancer_type):
    cancer_type = cancer_type.lower()
    examples_dir = os.path.join(config.ROOT_DIR, "data/report_examples")
    file_path = os.path.join(examples_dir, f"{cancer_type}.txt")

    if not os.path.exists(file_path):
        print(f"Warning: No example file found for {cancer_type}.")
        return ""

    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()

def get_gpt_eval_prompt(reference, candidate):
    """
    Generate a prompt for GPT to evaluate the similarity between two scientific reports.
    """
    return (
        "You are an expert in scientific pathology report evaluation. Your task is to compare two pathology reports "
        "and assign a similarity score on a scale from 0 to 10. A score of 10 means the reports contain almost the same "
        "medical findings, while a score of 0 means they discuss completely different content.\n\n"
        "Ignore irrelevant details such as patient name, sample ID, date, physician name, and other administrative information.\n"
        "The reports may have different formats, but only focus on comparing their medical content, including diagnoses, observations, "
        "and clinical details.\n\n"
        f"Ground Truth Report:\n{reference}\n\n"
        f"Generated Report:\n{candidate}\n\n"
        "Provide only a single numerical score from 0 to 10, without explanation:"
    )

def generate_checklist_prompt(reference_text, candidate_text, vqa_questions):
    formatted_questions = "\n".join([
        f"Q{i+1}: {q['Question']}\nChoices: {', '.join(q['Choice'])}"
        for i, q in enumerate(vqa_questions)
    ])
    
    prompt = f"""
You are an expert pathologist simulating the process of filling out a TCGA enrollment form based on pathology reports.
You will compare the **reference pathology report** and the **candidate pathology report** to evaluate their consistency.

Each question corresponds to a section in the form, and the choices are the available options. Your task is to determine **whether the candidate report provides the same answer as the reference report** for each question.

**Reference Report:**
\"\"\"{reference_text}\"\"\"

**Candidate Report:**
\"\"\"{candidate_text}\"\"\"

**Checklist Questions:**
{formatted_questions}

**Instructions:**
- For each question, determine if the candidate report provides the same answer as the reference report.
- **If the answers are identical, return 0. If they are different, return 1.**
- Ignore minor wording differences, focus on the meaning.
- If the candidate report does not provide an answer, assume it differs from the reference and mark it as **1**.
- **Only return a JSON array of 0s and 1s, strictly in order, without any additional text.**
- **Do NOT include any extra text. The output must ONLY be a valid JSON array.

**Output Format (Example):**
[0, 0, 1, 1, 0, ...]

Now, generate your response.
"""
    return prompt

