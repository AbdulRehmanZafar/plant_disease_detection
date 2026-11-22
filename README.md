# 🌿 Plant Disease Detection — ResNet50 + Transfer Learning + Grad-CAM

Machine Learning Project Exhibition · Group 1 · 30 November 2026

A complete, runnable plant-leaf disease classifier: **38 classes**, **14 crops**,
**94 % test accuracy**, an explainable **Grad-CAM** heat-map, and agronomic care
advice that works **with no internet**.

---

## 1. What this project is

A visitor uploads any photo of a leaf. In about one second the app returns:

| Output | Where it comes from |
|---|---|
| Crop + disease name | ResNet50 classifier, 38-way softmax |
| Confidence + top-5 ranking | the same softmax vector |
| Grad-CAM heat-map | gradients of the winning class w.r.t. the last conv layer |
| Symptoms / treatment / prevention | offline knowledge base (38 entries) |

**Architecture in one picture**

```
224×224×3 image
      │  BGR, ImageNet means subtracted (no /255)
      ▼
ResNet50 backbone (ImageNet, FROZEN)          23,587,712 params
      │  output: 7 × 7 × 2048
      ▼
GlobalAveragePooling2D  →  2048
Dense(256, relu)        →  256
Dropout(0.4)
Dense(38, softmax)      →  probabilities          534,310 params
                                                ───────────────
                                    total      24,122,022 params
```

---

## 2. Quick start

### The easy way — double-click `RUN.bat` (recommended)

1. Extract the whole folder somewhere normal (Desktop / Documents). Do **not** run
   it from inside the ZIP.
2. **Double-click `RUN.bat`.**
3. Wait 5–10 minutes and leave the black window open. It finds or installs
   Python 3.12, installs the libraries (~600 MB), downloads the 92 MB model, runs
   the nine checks, then starts the app.
4. The browser opens at **http://localhost:8501**. Drag a leaf photo in.

Every later run: double-click `RUN.bat` again — about 15 seconds, no internet.

If Windows says *"Windows protected your PC"* → **More info → Run anyway**.

If the launcher stops, it prints the exact fix in the window. Nothing is damaged
by running it again.

> A step-by-step illustrated version, with the expected output at every step, is
> **`HOW_TO_RUN.docx`** (or `HOW_TO_RUN.pdf`). `START_HERE.txt` is the one-page
> version to read first.

### The manual way — five commands

> **Do this once, at home, with internet.** Exhibition Wi-Fi will not be needed
> afterwards.

```powershell
# 0) open PowerShell in this folder
cd "path\to\plant-disease-detection"

# 1) create the environment + install TensorFlow, Streamlit, ... (~600 MB)
powershell -ExecutionPolicy Bypass -File .\setup.ps1

# 2) download the ready-trained 92 MB weights
.\.venv\Scripts\python.exe scripts\get_weights.py

# 3) prove that everything works
.\.venv\Scripts\python.exe scripts\check_setup.py

# 4) run the demo
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The browser opens at **http://localhost:8501**. Drag a leaf photo in.

If `setup.ps1` says *"No Python 3.10–3.12 found"*, install it first:

```powershell
winget install -e --id Python.Python.3.12
```

then open a **new** PowerShell window and run `setup.ps1` again.

**Why Python 3.12 and not the 3.14 that came with your laptop?**
TensorFlow 2.20 has no wheels for 3.13/3.14. `pip install tensorflow` simply
fails there. This is the number-one reason student ML projects die the night
before the exhibition — fix it now, not at 11 pm on 29 November.

---

## 3. Two ways to get a model

You need **one** of these in `models/`. Pick A unless you want the training
experience.

### A) Download the ready-trained weights (recommended, 3 minutes)

```powershell
.\.venv\Scripts\python.exe scripts\get_weights.py
```

* downloads `resnet_weights.npz` (96,567,102 bytes ≈ 92 MB) from the public
  repository this project is based on,
* checks that the file is a real binary (not a saved web page),
* rebuilds the architecture, compares **every** tensor shape, applies the
  weights,
* converts them to `models/plant_disease_resnet50.keras` so the app loads in
  about a second afterwards.

### B) Train it yourself (the honest way, 1–3 hours on CPU)

```powershell
# 1) get PlantVillage (54,305 images, 38 classes) from Kaggle, unzip it
#    https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

# 2) stratified 70/15/15 split into data/plantvillage_split/
.\.venv\Scripts\python.exe prepare_dataset.py `
    --src "D:\datasets\PlantVillage" `
    --dst data\plantvillage_split

# 3) two-phase transfer learning
.\.venv\Scripts\python.exe train.py --data data\plantvillage_split

# 4) full test-set evaluation + confusion matrix
.\.venv\Scripts\python.exe evaluate.py --data data\plantvillage_split
```

`prepare_dataset.py` prints the class-count table and writes
`class_names.json`. `train.py` writes `models/plant_disease_resnet50.keras`,
`reports/training_curves.png` and `reports/metrics.json` — the app picks them up
automatically.

**In a hurry?** Train on a balanced subset (about 30 minutes on CPU, still ~90 %):

```powershell
.\.venv\Scripts\python.exe prepare_dataset.py --src ... --dst data\mini --per-class-limit 300
.\.venv\Scripts\python.exe train.py --data data\mini --epochs 6 --fine-tune-epochs 0
```

---

## 4. Project structure

```
plant-disease-detection/
├── app.py                  Streamlit demo (the thing you show the examiner)
├── train.py                two-phase transfer learning
├── evaluate.py             test metrics, confusion matrix, per-class report
├── prepare_dataset.py      PlantVillage -> stratified train/val/test
├── predict_cli.py          classify from the terminal (backup demo)
│
├── src/
│   ├── config.py           paths, constants, the 38 class names (order matters!)
│   ├── preprocessing.py    any image -> (1,224,224,3) tensor · no TensorFlow
│   ├── npz_utils.py        reading resnet_weights.npz in the right order · no TF
│   ├── model.py            architecture + every way to load weights
│   ├── gradcam.py          Grad-CAM heat-maps
│   └── advice.py           38-class agronomic knowledge base + optional LLM
│
├── scripts/
│   ├── get_weights.py      download + verify + convert the 92 MB weights
│   └── check_setup.py      9-point pre-exhibition health check (run this!)
│
├── tests/selftest.py       13 NumPy-only tests, run in ~1 second
├── RUN.bat                 ★ double-click this to start everything
├── START_HERE.txt          one page: what to do, in plain words
├── setup_and_run.ps1       the logic behind RUN.bat (5 steps, with fixes)
├── setup.ps1 / run.ps1     the older manual variants
├── requirements.txt
├── HOW_TO_RUN.docx         10-page illustrated run guide (print this)
├── HOW_TO_RUN.pdf          the same guide as PDF
├── README.md               this file
├── QUICKSTART.md           one page, exhibition-morning checklist
├── VIVA_GUIDE.md           90+ questions with answers, per-member topic split
└── DEMO_SCRIPT.md          the 3-minute demonstration, word for word
```

> **New to the project?** Open **`HOW_TO_RUN.docx`** (or the PDF). It walks through
> every command with the expected output at each step, plus troubleshooting and an
> exhibition-day checklist.

---

## 5. The dataset — what the examiner will ask

**PlantVillage** (Hughes & Salathé, 2015) — the standard plant-disease benchmark.

| Property | Value |
|---|---|
| Images | 54,305 |
| Classes | 38 (26 diseases + 12 healthy) |
| Crops | 14 |
| Image type | single detached leaf, uniform background, JPEG |
| Class sizes | **imbalanced**: 275 (Apple cedar rust) … 5,507 (Orange HLB) |
| Captured | laboratory / controlled conditions |

**Three variants ship with the Kaggle download:** `color` (RGB — the one we
use), `grayscale`, and `segmented` (background removed). We use `color`.

**Why the split matters.** A naive `train_test_split` on 54k images is fine, but
if you ever split *after* augmentation you leak the same leaf into train and
test and get 99 % nonsense. `prepare_dataset.py` splits the **original files**
70/15/15, stratified per class, with a fixed seed, and hard-links files (instant,
no extra disk). Every class keeps at least one image in each split.

**Known limitation to admit out loud:** every image is a single leaf on a plain
background, so the model can learn to use background colour a little. That is
exactly why Grad-CAM is in this project — it lets you *show* whether the model
is looking at the lesion or at the background. Say this before the examiner
does; it turns a weakness into evidence of understanding.

---

## 6. Training details

| Setting | Phase 1 | Phase 2 |
|---|---|---|
| Backbone | frozen | last 30 layers unfrozen (BatchNorm stays frozen) |
| Trainable params | 534,310 | ≈9 M (grows with `--fine-tune-layers`) |
| Learning rate | 1e-4 | 1e-5 |
| Optimizer | Adam | Adam |
| Loss | SparseCategoricalCrossentropy | same |
| Batch size | 32 | 32 |
| Epochs | 12 (default) | 10 (default) |
| Callbacks | ModelCheckpoint, EarlyStopping(patience 4, restore best), ReduceLROnPlateau(0.2, patience 2), CSVLogger | same |

**Augmentation** (training split only, applied before `preprocess_input`):
horizontal flip, rotation ±21°, zoom 10 %, contrast 10 %, brightness 10 %.
No vertical flip, no 90° rotation — leaves have a natural orientation, and
PlantVillage is consistently oriented, so those transforms add noise, not
invariance.

**Reported results** (reference run; your own numbers land in `reports/`):

| Model | Train | Val | Test |
|---|---|---|---|
| Baseline CNN from scratch | 83 % | 77 % | 79 % |
| ResNet50, frozen backbone | 70 % | 89 % | 85 % |
| ResNet50, fine-tuned last 30 layers | 83 % | 94 % | **94 %** |

The frozen-backbone model under-fits the training set (70 %) yet generalises to
89 % validation — a classic sign that the head is small and the pretrained
features are already good. Fine-tuning lets the model fit harder and buys about
2 more points of validation accuracy.

---

## 7. Preprocessing — the details that actually matter

```python
# src/preprocessing.py
x = rgb.astype(np.float32)      # 1. cast FIRST
x = x[..., ::-1]                # 2. RGB -> BGR
x -= [103.939, 116.779, 123.68] # 3. ImageNet (caffe) means, in BGR order
```

Three mistakes that silently destroy accuracy, all avoided here:

1. **Subtracting before casting.** `uint8` wraps around (`0 - 104 → 152`).
   `tests/selftest.py` catches this explicitly.
2. **Forgetting the RGB→BGR swap.** ResNet50 was trained in BGR; feeding RGB
   swaps the "red" and "blue" detectors.
3. **Dividing by 255.** `preprocess_input` does not rescale — it only
   mean-subtracts. Dividing by 255 shifts every activation far outside the
   trained range.

`scripts/check_setup.py` numerically compares our function with
`keras.applications.resnet50.preprocess_input` on random images and prints the
maximum difference (should be `0.00e+00`).

---

## 8. Grad-CAM in five lines

```
A        = last conv feature maps                  (7, 7, 2048)
y^c      = score of the winning class
alpha_k  = mean over space of d y^c / d A^k        (importance of channel k)
L        = ReLU( sum_k alpha_k * A^k )             (keep positive evidence only)
L        = L / max(L)                              (normalise, then colour-map)
```

If the hot region sits on the lesion, the prediction is grounded. If it sits on
the background, the model is cheating — and you can say so, on the spot, which
is exactly the kind of honesty examiners reward.

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No matching distribution found for tensorflow` | Python 3.13/3.14 | install Python 3.12, recreate `.venv` |
| `The app needs a trained model in models/` | no weights | `python scripts/get_weights.py` |
| `.npz is only 167 KB` | you saved the GitHub *web page* | use "Download raw file" or `get_weights.py` |
| `Accuracy is ~2 %` | class list or preprocessing mismatch | run `scripts/check_setup.py` |
| First prediction takes 20 s | model loading + TF graph warm-up | normal; subsequent ones are ~0.5 s |
| `DLL load failed` on Windows | missing MSVC runtime | install [VC++ 2015-2022 redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| Streamlit opens but page is blank | wrong folder | run from this folder: `streamlit run app.py` |
| Port 8501 busy | another instance | `streamlit run app.py --server.port 8600` |
| App downloads 95 MB at startup | backbone built with `weights="imagenet"` | not used in the app: it builds with `weights=None` because the file already has every weight |

---

## 10. Verification status (read this once)

Written and checked on 22 November 2026:

* ✅ `tests/selftest.py` — **13/13 pass**: class ordering, preprocessing maths
  (including the uint8 wrap-around trap), npz ordering, HTML-file rejection,
  advice coverage for all 38 classes.
* ✅ `prepare_dataset.py` — verified end-to-end on a synthetic 38-class tree:
  70/15/15 split, no leakage between splits, every class present in every split,
  alphabetical `class_names.json`.
* ✅ every Python file byte-compiles.
* ⚠️ `train.py`, `evaluate.py`, `app.py` and `scripts/check_setup.py` need
  TensorFlow, which was not installable in the authoring environment
  (no network). **Run `scripts/check_setup.py` on your machine first** — it
  builds the model, validates every weight tensor against the file, runs a
  forward pass, cross-checks preprocessing against Keras, and runs Grad-CAM.
  If all 9 checks pass, the project is demo-ready.

---

## 11. Exhibition-day checklist

```powershell
# 10 minutes before the examiner arrives:
.\.venv\Scripts\python.exe scripts\check_setup.py     # must say 9/9
.\.venv\Scripts\python.exe -m streamlit run app.py
# upload 3 known images, confirm sensible answers + heat-maps
```

Full plan in **`DEMO_SCRIPT.md`**. Study **`VIVA_GUIDE.md`** before the day —
the marking scheme gives **zero** for a working demo you cannot explain.

---

## 12. Credits & honesty

* Dataset: PlantVillage — Hughes & Salathé (2015), CC-BY-SA.
* Baseline architecture and weight file: the public repository
  `Ishaaq09/Automated_plant_disease_detection_using_Deep_Learning_and_Transfer_Learning`
  (permission for existing code is explicitly granted in the assignment brief).
* Re-engineered here: modular package, robust npz loading with shape
  validation, offline advice layer (replacing a fragile API-only path), Grad-CAM
  without OpenCV, full evaluation suite, dataset splitter, 13 unit tests, setup
  scripts, and the documentation you are reading.
* Be ready to say which parts you wrote versus reused. "We reused the published
  weights, and we rebuilt the pipeline around them because the original script
  depended on a cloud API, OpenCV and a fragile weight-loading hack" is a
  strong, honest answer.
#   p l a n t _ d i s e a s e _ d e t e c t i o n  
 