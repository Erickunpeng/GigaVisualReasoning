# Project directory paths
ROOT_DIR = "/abs/path/to/GigaVisualReasoning"
# Your folder path for TCGA Whole Slide Images
DATA_DIR = "/abs/path/to/TCGA/slides"
OUTPUT_DIR = f"{ROOT_DIR}/output"
QUICK_START_DIR = f"{ROOT_DIR}/quick_start_output"
# Your folder path for TCGA Meta Data
META_DATA_DIR = "/abs/path/to/TCGA_pancancer_clinical_data.csv"
# Your folder path for ground-truth reports
VQA_META_DATA_DIR = f"{ROOT_DIR}/aws_response"

# n_iters
NUM_ITER = 10

# Cancer types
CANCER_SUBTYPE_MAP = {
    "BRCA": ["IDC", "ILC"], # Breast: Breast Invasive Ductal Carcinoma vs. Breast Invasive Lobular Carcinoma
    "COLON": ["COAD", "READ"], # Bowel: Colon Adenocarcinoma vs. Rectal Adenocarcinoma
    "ESO": ["ESCA", "ESCC", "STAD"], # Esophagus / Stomach: Esophageal Adenocarcinoma vs. Esophageal Squamous Cell Carcinoma vs. Stomach Adenocarcinoma
    "HEP": ["CHOL", "HCC"], # Biliary Tract: Cholangiocarcinoma vs. Hepatocellular Carcinoma
    "LUNG": ["LUAD", "LUSC"], # Lung: Lung Adenocarcinoma vs. Lung Squamous Cell Carcinoma
    "RCC": ["CCRCC", "CHRCC", "PRCC"], # Kidney: Renal Clear Cell Carcinoma vs. Chromophobe Renal Cell Carcinoma vs. Papillary Renal Cell Carcinoma
    "GLIOMA": ["GBM", "ODG"], # CNS / Brain: Glioblastoma Multiforme vs. Oligodendroglioma
    "ADREN": ["ACC", "PHC"], # Adrenal Gland: Adrenocortical Carcinoma vs. Pheochromocytoma
    "CERVIX": ["CESC", "ECAD"], # Cervix: Cervical Squamous Cell Carcinoma vs. Endocervical Adenocarcinoma
    "PLEURA": ["PLBMESO", "PLEMESO", "PLSMESO"], # Pleura: Pleural Mesothelioma, Biphasic Type vs. Pleural Mesothelioma, Epithelioid Type vs. Pleural Mesothelioma, Sarcomatoid Type
    "SOFT": ["DDLS", "LMS", "MFS", "MFH"], # Soft Tissue: Dedifferentiated Liposarcoma vs. Leiomyosarcoma vs. Myxofibrosarcoma vs. Undifferentiated Pleomorphic Sarcoma/Malignant Fibrous Histiocytoma/High-Grade Spindle Cell Sarcoma
    "TESTIS": ["SEM", "MGCT"], # Testis: Seminoma vs. Mixed Germ Cell Tumor
    "UTERUS": ["UEC", "USC"], # Uterus: Uterine Endometrioid Carcinoma vs. Uterine Serous Carcinoma/Uterine Papillary Serous Carcinoma
}

CANCER_FOLDER_MAP = {
    "BRCA": ["TCGA-BRCA"],  # Breast Cancer
    "COLON": ["TCGA-COAD", "TCGA-READ"],  # Colorectal Cancer
    "ESO": ["TCGA-ESCA", "TCGA-STAD"],  # Esophagogastric Cancer
    "HEP": ["TCGA-CHOL", "TCGA-LIHC"],  # Hepatobiliary Cancer
    "LUNG": ["TCGA-LUAD", "TCGA-LUSC"],  # NSCLC
    "RCC": ["TCGA-KIRC", "TCGA-KICH", "TCGA-KIRP"],  # RCC
    "GLIOMA": ["TCGA-GBM", "TCGA-LGG"],  # Glioma
    "ADREN": ["TCGA-ACC", "TCGA-PCPG"], # Adrenal Gland
    "CERVIX": ["TCGA-CESC"], # Cervix
    "PLEURA": ["TCGA-MESO"], # Pleura
    "SOFT": ["TCGA-SARC"], # Soft Tissue
    "TESTIS": ["TCGA-TGCT"], # Testis
    "UTERUS": ["TCGA-UCEC"], # Uterus
}

