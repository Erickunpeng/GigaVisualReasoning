from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from utils.openai_client import get_openai_response_text_only
import src.report.report_prompt as prompt
import re
import nltk

def preprocess_text(text):
    text = re.sub(r"(?i)(Patient\s*:\s*.*)", "", text)  # Patient Name
    text = re.sub(r"(?i)(Specimen\s*#\s*:.*)", "", text)  # Specimen ID
    text = re.sub(r"(?i)(UUID\s*:\s*[A-Z0-9-]+)", "", text)  # UUID
    text = re.sub(r"(?i)(Reported\s*:\s*.*)", "", text)  # Reported Date
    text = re.sub(r"(?i)(Reviewed\s*:\s*.*)", "", text)  # Reviewer Info
    text = re.sub(r"(?i)(Physician\s*\(s\)\s*:\s*.*)", "", text)  # Physician Name
    text = re.sub(r"(?i)(FMP/SSN\s*:\s*.*)", "", text)  # FMP/SSN

    keywords = ["FINAL DIAGNOSIS", "SURGICAL PATHOLOGY REPORT", "SPECIMEN", "MICROSCOPIC DESCRIPTION"]
    
    for keyword in keywords:
        match = re.search(rf"(?i){keyword}.*", text, re.DOTALL)
        if match:
            text = match.group(0) 
            break

    text = re.sub(r"\n\s*\n", "\n", text)
    text = text.strip()
    return text

def calculate_bleu(reference, candidate):
    reference = preprocess_text(reference)
    candidate = preprocess_text(candidate)

    reference_tokens = reference.split()
    candidate_tokens = candidate.split()

    smooth = SmoothingFunction().method4  # Prevent zero scores for low n-gram match
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smooth)

    return bleu_score

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores

def calculate_gpt_eval_score(reference, candidate):
    """
    Use GPT to evaluate the similarity between a generated scientific report and the ground truth.
    """
    reference = preprocess_text(reference)
    candidate = preprocess_text(candidate)
    response = get_openai_response_text_only(prompt.get_gpt_eval_prompt(reference, candidate))
    try:
        score = int(response.strip())
        score = max(0, min(score, 10))
    except ValueError:
        print(f"WARNING: Unexpected GPT response: {response}")
        score = None 

    return score
    