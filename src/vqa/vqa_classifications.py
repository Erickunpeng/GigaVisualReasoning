import re
from typing import Literal

Category = Literal[
    "Diagnosis",
    "Stage",
    "Grade",
    "Size",
    "Margin/Involvement",
    "IHC/Biomarker",
    "Survival/Prognosis",
    "Other Pathology Findings",
    "Other",
]

def classify_vqa_question(q: str) -> Category:
    """Classify a WSI-VQA question into a semantic category."""
    if not q:
        return "Other"
    s = q.lower().strip()
    # ---- quick helpers ----
    def has(patterns):
        return any(re.search(p, s) for p in patterns)
    # ---- category dictionaries (regex, not plain substrings) ----
    # Diagnosis > Stage > Grade > Size > Margin/Involvement > IHC > Survival > Other Findings > Other
    DIAGNOSIS_PATTERNS = [
        r"\bdiagnos(ed|is|e|is of|is based on)\b",
        r"\b(final )?diagnosis\b",
        r"\bpatholog(y|ical) (diagnosis|type)\b",
        r"\bhistolog(ical|y|ic)(_|\s)?type\b",
        r"\bcarcinoma (type|subtype)\b",
        r"\bwhat type of (tumou?r|cancer|carcinoma)\b",
        r"\bsubtype\b",
        r"\bmedullary|ductal|lobular|adenocarcinoma|squamous|mucinous|papillary|phyllodes\b",
    ]

    STAGE_PATTERNS = [
        r"\bstage\b",                    # stage ii, stage iib...
        r"\b(p|c)?t[0-4][a-d]?\b",       # pT2, pt4a
        r"\b(p|c)?n[0-3][a-c]?\b",       # pN1, pN2b
        r"\b(p|c)?m[0-1]\b",             # pM0, cM1
        r"\b(ajcc|tnm)\b",
    ]

    GRADE_PATTERNS = [
        r"\bnottingham\b",
        r"\bbloom-?richardson\b",
        r"\bhistolog(ic(al)?)? grade\b",
        r"\bnuclear grade\b",
        r"\bgrade\s*(i{1,3}v?|[1-4])\b",      # grade 1/2/3/4 or roman
        r"\bscore\s*\d\s*/\s*9\b",            # 6/9, 7/9, 9/9
        r"\btubule formation\b|\bmitotic\b|\bnuclear pleomorphism\b",
    ]

    SIZE_PATTERNS = [
        r"\bsize\b|\bdimension(s)?\b|\bdiameter\b|\bmeasure(ment)?s?\b",
        r"\b\d+(\.\d+)?\s*(cm|mm)\b",
        r"\b\d+(\.\d+)?\s*x\s*\d+(\.\d+)?\s*(x\s*\d+(\.\d+)?)?\s*(cm|mm)\b",
        r"\bmaximum (dimension|diameter)\b|\bapproximately\b",
        r"\brange\b.*\bcm\b",
    ]

    MARGIN_PATTERNS = [
        r"\bmargin(s)?\b",                     # surgical margins, resection margins
        r"\b(clear|negative|positive)\s+margin(s)?\b",
        r"\bclosest margin\b|\bdistance to (the )?margin\b",
        r"\b(involvement|involved)\b.*\bmargin(s)?\b",
        r"\b(lymph|angio)vascular (invasion|involvement)\b",
        r"\bvascular invasion\b|\blymphatic invasion\b",
    ]

    IHC_PATTERNS = [
        r"\b(er|estrogen receptor)\b",
        r"\b(pr|progesterone receptor)\b",
        r"\bher[- ]?2\b|\bneu\b|\b(her2/neu)\b",
        r"\bimmuno(histo)?chem(istry|ical)\b|\bimmunostain(s|ing)?\b",
        r"\bki-?67\b|\bcd\d+\b|\bcytokeratin\b",
        r"\bamplification\b|\boverexpression\b|\bequivocal\b",
    ]

    SURVIVAL_PATTERNS = [
        r"\bvital[_\s-]?status\b|\balive\b|\bdead\b|\bdeceased\b",
        r"\bsurvival (time|months?)\b|\bos\b|\bdss\b|\bpfs\b",
    ]

    OTHER_FINDINGS_PATTERNS = [
        r"\bnecros(is|e|tic)|comedo\b",
        r"\bfibro(sis|cystic)\b|\bfibrotic\b",
        r"\binflamm(ation|atory)\b|\bgranuloma(s)?\b|\blactation-?like\b",
        r"\bpaget'?s\b|\bperineural\b",
        r"\bmitos(es|is)\b per \d+\s*hpfs?\b",
        r"\bcalcification(s)?\b|microcalcification\b",
        r"\bnonproliferative\b|\bapocrine metaplasia\b|\bhyperplasia\b",
    ]

    # ---- classification by priority ----
    if has(DIAGNOSIS_PATTERNS):
        return "Diagnosis"
    if has(STAGE_PATTERNS):
        return "Stage"
    if has(GRADE_PATTERNS):
        return "Grade"
    if has(SIZE_PATTERNS):
        return "Size"
    if has(MARGIN_PATTERNS):
        return "Margin/Involvement"
    if has(IHC_PATTERNS):
        return "IHC/Biomarker"
    if has(SURVIVAL_PATTERNS):
        return "Survival/Prognosis"
    if has(OTHER_FINDINGS_PATTERNS):
        return "Other Pathology Findings"
    return "Other"
