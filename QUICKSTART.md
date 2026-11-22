# QUICKSTART — 3 steps

## 1. Extract the folder

Right-click the ZIP → **Extract All** → pick a normal place (Desktop, Documents).
**Do not** double-click things inside the ZIP preview.

## 2. Double-click `RUN.bat`

That is the whole installation. A black window appears and does everything:

| What it does | Time |
|---|---|
| finds Python, installs 3.12 if the laptop does not have it | 1–3 min |
| installs the libraries (~600 MB) | 3–8 min |
| downloads the 92 MB trained model | 1–3 min |
| runs the 9-point check | 20 s |
| starts the app | 5 s |

Leave the window open. First run: **5–10 minutes**, needs internet.

## 3. Upload a leaf photo

The browser opens at **http://localhost:8501**. Drag a photo into the page.

---

## Every later run

Double-click `RUN.bat` again. About 15 seconds, **no internet needed**.

---

## If Windows blocks it

*"Windows protected your PC"* → **More info** → **Run anyway**.

## If the window shows an error

It prints the exact fix, in plain words. Nothing is damaged by running
`RUN.bat` again — every finished step is skipped.

| The message says | Do this |
|---|---|
| Python 3.12 is required | `winget install -e --id Python.Python.3.12`, close the window, run `RUN.bat` again. Or install from python.org and tick *Add python.exe to PATH* |
| Installing the libraries failed | Check the internet / mobile hotspot, then run `RUN.bat` again. Or: `powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1 -Repair` |
| The model could not be downloaded | Retry. Or bring `resnet_weights.npz` on a USB stick: `python scripts\get_weights.py --from "E:\resnet_weights.npz"` |
| Some checks failed | Read the FAIL lines; usually a half-finished install → run with `-Repair` |

---

## Manual commands (only if you need them)

```powershell
cd "path\to\plant-disease-detection"

.\.venv\Scripts\python.exe scripts\check_setup.py          # 9/9 = ready
.\.venv\Scripts\python.exe -m streamlit run app.py         # start the app
.\.venv\Scripts\python.exe predict_cli.py leaf.jpg --gradcam out\cam.png
```

---

## The four files that matter, if you need to explain the project

| File | One line |
|---|---|
| `src/preprocessing.py` | image → 224×224, RGB→BGR, subtract `[103.939, 116.779, 123.68]` |
| `src/model.py` | ResNet50 (frozen) + GAP + Dense 256 + Dropout 0.4 + Dense 38 softmax |
| `src/gradcam.py` | gradient of the class score w.r.t. `conv5_block3_out` |
| `src/advice.py` | 38-class offline treatment/prevention knowledge base |

## Three numbers, three sentences

* **38 classes, 14 crops, 54,305 images, 94 % test accuracy.**
* **24,122,022 parameters, of which 534,310 (the head) are trained in phase 1.**
* **Freeze first, then unfreeze the last 30 layers at a 10× smaller learning rate.**

---

## If the examiner asks something you truly cannot answer

> "I don't know that off the top of my head — but here is where it lives and how
> I would find out."

Then name the file. Honesty plus navigational competence reads far better than
invented confidence.
