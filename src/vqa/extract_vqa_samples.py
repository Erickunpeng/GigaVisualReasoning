import os
import json
import config
from utils.file_utils import find_svs_file

def extract_all_sample_ids_to_txt(output_file="sample_ids.txt"):
    data_dir = os.path.join(config.ROOT_DIR, "data", "question_list")
    file_names = ["WsiVQA_train.json", "WsiVQA_val.json", "WsiVQA_test.json"]
    sample_ids = set()
    for file_name in file_names:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found.")
            continue
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                if "Id" in item:
                    sample_ids.add(item["Id"])
    with open(output_file, "w") as f:
        for sample_id in sorted(sample_ids):
            f.write(sample_id + "\n")
    print(f"Extracted {len(sample_ids)} unique sample IDs and saved to {output_file}.")


def extract_svs_paths(sample_ids_file="sample_ids.txt", output_file="sample_id_path.txt", cancer_type="BRCA"):
    if not os.path.exists(sample_ids_file):
        print(f"Error: {sample_ids_file} not found.")
        return
    with open(sample_ids_file, "r") as f:
        sample_ids = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(sample_ids)} sample IDs from {sample_ids_file}")
    svs_paths = []
    for sample_id in sample_ids:
        try:
            svs_path = find_svs_file(sample_id, cancer_type)
            if svs_path:
                svs_paths.append(f"{sample_id} {svs_path}")
                print(f"Found SVS file for {sample_id}: {svs_path}")
            else:
                print(f"WARNING: No SVS file found for {sample_id}")
        except Exception as e:
            print(f"ERROR processing {sample_id}: {e}")
            continue
    with open(output_file, "w") as f:
        f.write("\n".join(svs_paths))
    print(f"SVS file paths saved to {output_file}")


if __name__ == "__main__":
    # extract_all_sample_ids_to_txt("sample_ids.txt")
    extract_svs_paths("sample_ids.txt", "sample_id_path.txt", "BRCA")
