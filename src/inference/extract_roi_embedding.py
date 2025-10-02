import torch, timm
from torchvision import transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.layers import SwiGLUPacked
from PIL import Image
import os
import glob
import config
import numpy as np
from tqdm import tqdm

MODELS = [
    {"name": "gigapath", "id": "hf_hub:prov-gigapath/prov-gigapath"},
    {"name": "UNI", "id": "hf_hub:MahmoodLab/UNI"},
    {"name": "H-optimus-0",  "id": "hf_hub:bioptimus/H-optimus-0"},
]
MODEL_ID = {m["name"]: m["id"] for m in MODELS}

def build_encoder_and_transform(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name not in MODEL_ID:
        raise ValueError(f"Unknown model_name: {model_name}")
    model_id = MODEL_ID[model_name]
    if model_name == "UNI":
        model = timm.create_model(
            model_id, pretrained=True, pretrained_strict=False,
            num_classes=0, global_pool="avg",
            fc_norm=False, init_values=1e-6,
        ).eval().to(device)
    elif model_name == "gigapath":
        model = timm.create_model(
            model_id, pretrained=True, pretrained_strict=False,
            num_classes=0, global_pool="avg",
        ).eval().to(device)
    elif model_name == "Virchow":
        model = timm.create_model(model_id, 
        pretrained=True, 
        mlp_layer=SwiGLUPacked, 
        act_layer=torch.nn.SiLU).eval().to(device)
        cfg = resolve_data_config(model.pretrained_cfg, model=model)
        transform = create_transform(**cfg)
        return model, transform, device
    elif model_name == "H-optimus-0":
        model = timm.create_model(
            model_id,
            pretrained=True,
            pretrained_strict=False,
            num_classes=0, 
            global_pool="avg",
        ).eval().to(device)
        cfg = resolve_data_config({}, model=model)
        transform = create_transform(**cfg, is_training=False)
        return model, transform, device
    else:
        model = timm.create_model(
            model_id, pretrained=True, pretrained_strict=False,
            num_classes=0, global_pool="avg",
        ).eval().to(device)
    cfg = resolve_data_config({}, model=model)
    transform = create_transform(**cfg, is_training=False)
    return model, transform, device

ENCODER, TRANSFORM, DEVICE = None, None, None

def extract_embedding_from_image(img_path, model_name):
    global ENCODER, TRANSFORM, DEVICE
    if ENCODER is None:
        ENCODER, TRANSFORM, DEVICE = build_encoder_and_transform(model_name)
    img = Image.open(img_path).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = ENCODER(x)
        feat = out
    return feat.squeeze(0).cpu().numpy()

def extract_embeddings_from_folder(root_dir, save_dir, mode, model_name):
    os.makedirs(save_dir, exist_ok=True)
    slide_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    print(f"Found {len(slide_dirs)} slides")
    for slide_id in tqdm(slide_dirs):
        slide_folder = os.path.join(root_dir, slide_id)
        roi_candidates = glob.glob(os.path.join(slide_folder, "roi_*.png"))
        gpt_candidate = os.path.join(slide_folder, "gpt_selected_roi.png")
        if len(roi_candidates) == 0 and not os.path.exists(gpt_candidate):
            print(f"[!] No ROI image in {slide_id}, skipped.")
            continue
        save_path = os.path.join(save_dir, f"{slide_id}.npy")
        if os.path.exists(save_path):
            print(f"[!] Embedding already exists for {slide_id}, skipped.")
            continue
        try:
            if mode == "roi":
                roi_path = sorted(roi_candidates)[0]
                embedding = extract_embedding_from_image(roi_path, model_name)
            elif mode == "gpt":
                embedding = extract_embedding_from_image(gpt_candidate, model_name)
            else:
                print(f"[!] Unknown mode: {mode}, skipped.")
                continue
            np.save(save_path, embedding)
        except Exception as e:
            print(f"[ERROR] Failed processing {slide_id}: {e}")

if __name__ == "__main__":
    cancer_types = ["COLON", "LUNG", "RCC", "GLIOMA", "HEP", "ESO",
                "ADREN", "CERVIX", "PLEURA", "SOFT", "TESTIS", "UTERUS"]
    for cancer_type in cancer_types:
        # cancer_type = "BRCA"
        model_name = "H-optimus-0" # gigapath / UNI / H-optimus-0
        mode = "gpt" # roi / gpt
        if mode == "roi":
            root_dir = os.path.join(config.QUICK_START_DIR, cancer_type, "roi_output")
            save_dir = os.path.join("inference_output/roi/", cancer_type, model_name)
        elif mode == "gpt":
            root_dir = os.path.join(config.OUTPUT_DIR, "subtyping", cancer_type, "baseline_output")
            save_dir = os.path.join("inference_output/gpt_baseline/", cancer_type, model_name)
        extract_embeddings_from_folder(root_dir, save_dir, mode, model_name)
