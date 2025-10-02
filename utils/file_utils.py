import config
import os

def find_svs_file(sample_id, cancer_type):
    for folder in config.CANCER_FOLDER_MAP.get(cancer_type, []):
        folder_path = os.path.join(config.DATA_DIR, folder)
        for root, _, files in os.walk(folder_path):  # Recursively search all subdirectories
            for file in files:
                if file.startswith(sample_id) and file.endswith(".svs"):
                    return os.path.join(root, file)  # Return the first matching file
    print(f"WARNING: No SVS file found for {sample_id} in {cancer_type}.")
    return None

def get_svs_files_from_folders(cancer_folder_map, cancer_type):
    # cancer_folder_map: config.CANCER_FOLDER_MAP
    svs_files = []
    for folder in cancer_folder_map.get(cancer_type, []):
        folder_path = os.path.join(config.DATA_DIR, folder)
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".svs"):
                    svs_files.append(os.path.join(root, file))
    if not svs_files:
        raise ValueError(f"No SVS files found for cancer type: {cancer_type}")
    return svs_files

def get_svs_files_from_repo(tcga_repo):
    svs_files = []
    repo_path = os.path.join(config.DATA_DIR, tcga_repo)
    if not os.path.exists(repo_path):
        raise ValueError(f"TCGA repository {tcga_repo} does not exist: {repo_path}")
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".svs"):
                svs_files.append(os.path.join(root, file))
    if not svs_files:
        print(f"Warning: No SVS files found in {tcga_repo}")
    return svs_files

def initialize_directories(cancer_type, output_path=None):
    data_dir = config.DATA_DIR
    if not output_path:
        output_path = os.path.join(config.OUTPUT_DIR, "subtyping", cancer_type, "roi_output")
    os.makedirs(output_path, exist_ok=True)
    return data_dir, output_path

def count_samples_per_cancer_type(cancer_type):
    if cancer_type not in config.CANCER_FOLDER_MAP:
        print(f"Error: Cancer type '{cancer_type}' not found in CANCER_FOLDER_MAP.")
        return {}
    cancer_counts = {}
    repo_counts = {}
    folders = config.CANCER_FOLDER_MAP[cancer_type]
    total_samples = 0
    for folder in folders:
        folder_path = os.path.join(config.DATA_DIR, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist. Skipping...")
            repo_counts[folder] = 0
            continue
        sample_count = sum(len([f for f in files if f.endswith(".svs")]) for _, _, files in os.walk(folder_path))
        repo_counts[folder] = sample_count
        total_samples += sample_count
    cancer_counts[cancer_type] = {
        "total_samples": total_samples,
        "repo_counts": repo_counts
    }
    print(f"\n{cancer_type}: {total_samples} samples")
    for repo, count in repo_counts.items():
        print(f"  - {repo}: {count} samples")
    return cancer_counts


if __name__ == "__main__":
    cancer_type = "HEP"
    sample_counts = count_samples_per_cancer_type(cancer_type)