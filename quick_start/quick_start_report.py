import os
from src.report.report import generate_reports_for_samples
from src.report.report_evaluate import compare_reports
import config


def main(cancer_type, n, overwrite=True):
    # input_directory = os.path.join(config.QUICK_START_DIR, cancer_type, "roi_output")
    # output_directory = os.path.join(config.QUICK_START_DIR, cancer_type, "report_output")
    input_directory = os.path.join("roi_selection_output/report", cancer_type)
    output_directory = os.path.join("roi_selection_output/report_output", cancer_type)
    groundtruth_folder = config.VQA_META_DATA_DIR
    generate_reports_for_samples(input_directory, output_directory, cancer_type, overwrite, n)
    compare_reports(output_directory, groundtruth_folder, overwrite)

if __name__ == "__main__":
    cancer_type = "BRCA"
    n = -1  # Set n = -1 if you want to evaluate all samples in that folder
    overwrite = True  # Set to False to skip existing reports
    main(cancer_type, n, overwrite)