# 🧠 GigaVisualReasoning

**GigaVisualReasoning** is a GPT-guided visual reasoning framework for computational pathology.  
It enables multi-stage ROI (Region of Interest) selection and multimodal reasoning for tasks such as:

- 🧬 Cancer subtyping  
- 📈 Survival prediction  
- 💬 Visual question answering (VQA)  
- 🧾 Clinical report generation  

---

## 📂 Project Structure

<pre>
GigaVisualReasoning/
├── config.py                &lt;global configuration: paths, subtype maps, constants&gt;
├── data/                    &lt;metadata, prompts, sample lists (optional)&gt;
├── quick_start/             &lt;one-click demo entry scripts&gt;
│   ├── quick_start_report.py
│   ├── quick_start_subtyping.py
│   └── quick_start_vqa.py
├── src/                     &lt;core library code&gt;
│   ├── inference/           &lt;feature/embedding extraction + inference glue&gt;
│   ├── report/              &lt;pathology report generation + evaluation&gt;
│   ├── roi_selection/       &lt;GPT-guided ROI selection logic&gt;
│   ├── subtyping/           &lt;cancer subtype prompting + postprocessing&gt;
│   ├── survival/            &lt;survival risk task utilities&gt;
│   └── vqa/                 &lt;visual question answering modules&gt;
├── utils/                   &lt;shared helpers: OpenAI/Azure client, file utils, etc.&gt;
└── README.md
</pre>


---

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/GigaVisualReasoning.git
cd GigaVisualReasoning
pip install -r requirements.txt
```

## Setup
### Step 1 — Edit `config.py` (set your own absolute paths)

#### REQUIRED: set these to your environment
```bash
ROOT_DIR = "/abs/path/to/GigaVisualReasoning"  
DATA_DIR = "/abs/path/to/TCGA/slides"   
META_DATA_DIR = "/abs/path/to/TCGA_pancancer_clinical_data.csv"
```

#### The expected TCGA WSI data folder has the following structure:
<pre>
DATA_DIR/
└── &lt;CANCER_TYPE_FOLDER&gt;/
    ├── &lt;SAMPLE_FOLDER_UUID_OR_BARCODE&gt;/
    │   └── &lt;TCGA-....&gt;.svs
    └── ...
</pre>

### Step 2 — Edit OpenAI API Settings

This project supports **Azure Managed Identity (recommended)** and **API Key**.  
Choose ONE method and follow the steps below.

#### Option A — Azure OpenAI (Managed Identity)
**Edit `openai_client.py`** and replace the `TODO` values:

#### Option B — OpenAI (API Key)
**Edit `openai_client.py`** to use a plain API key  
When initializing GVR Agent using Option B, change the `"config_list"` param to following setting and replace your `open_api_key`:
```bash
"config_list": [{"api_key": openai_api_key, "model": "gpt-4o"}]
```

---

## 🚀 Quick Start

You can try demo tasks directly using the scripts under `quick_start/` and `extract_roi.py` under `src/roi_selection`:

**A. ROI extraction (multi-task or single-task)**  
run  
`python src/roi_selection/extract_roi.py`   
to let the ROI Agent pick informative regions per task. You need to modify 'cancer_type' and 'task_type'

**B. Quick start scripts (single task end-to-end)** — run one script per task under `quick_start/` to perform ROI selection **and** final prediction/response.

```bash
# 1. Run a quick cancer subtyping example
python quick_start/quick_start_subtyping.py

# 2. Generate a clinical report
python quick_start/quick_start_report.py

# 3. Run a visual question answering example
python quick_start/quick_start_vqa.py
```

## Contact
Please feel free to submit a Github issue if you have any questions or find any bugs. We do not guarantee any support, but will do our best if we can help.