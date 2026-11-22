# DEMO SCRIPT — 30 November 2026, 12:30 PM

Total marks: 10, individual. The demonstration is where most of them are won.

---

## A. 10 minutes before (do not skip)

```powershell
cd "path\to\plant-disease-detection"

# 1) pre-flight: must print 9/9 checks passed
.\.venv\Scripts\python.exe scripts\check_setup.py

# 2) start the app
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then, in the browser at `http://localhost:8501`:

1. Upload **three known images** and confirm each gives a sensible answer:
   * a tomato leaf with early blight,
   * a healthy leaf,
   * something clearly different (a potato late blight image).
2. Confirm the Grad-CAM overlay appears and lands on the leaf.
3. Leave the app open on the upload screen. **Do not** leave a previous result
   on screen — the examiner should see the build-up.

**Checklist**

- [ ] Laptop charged, charger in the bag
- [ ] `.venv` works (test it, do not assume)
- [ ] weights present: `models/plant_disease_resnet50.keras` (~92 MB)
- [ ] `samples/` has 8–10 test images named by their true class
- [ ] phone hotspot available as backup internet (only needed if something must
      be re-downloaded)
- [ ] USB stick with the whole project folder **including** `models/`
- [ ] second terminal open, in the project folder, ready for `predict_cli.py`
- [ ] brightness up, browser zoom ~110 % so the examiner can read it
- [ ] you know which of the four members is presenting which section

---

## B. The 3-minute demonstration (word for word)

### 0:00 — Frame it (15 s)

> "This is a plant disease detector. You give it a photo of a leaf, it tells you
> which of 38 diseases it is — or that the plant is healthy — and then what to
> do about it. It is ResNet50 with transfer learning, 94 % test accuracy, and it
> explains every prediction with a heat-map."

*Point at the three tabs: Detect / How it works / Model & results.*

### 0:15 — First upload (45 s)

> "I'll upload a leaf. **Examiner, would you like to pick one from my samples
> folder, or upload something from your own phone?**"

*If they give their own photo, take it — this scores better than a canned image.
If it is a plant photo, the answer will usually be right. If it is not a plant,
see section D.*

While it processes:

> "Under the hood: the image is resized to 224×224, converted from RGB to BGR,
> and the ImageNet channel means are subtracted — exactly what ResNet50 was
> trained with. Then it goes through the frozen backbone, which produces a
> 7×7×2048 feature map, then global average pooling, a 256-unit dense layer,
> dropout, and a 38-way softmax."

### 1:00 — Read the result out loud (30 s)

> "So: **Tomato — early blight, 96.4 % confidence.** The green bar is the
> confidence. The class listed second is late blight at 2 % — that makes sense,
> the lesions look similar, and we'll come back to that."

*Click the "Class probabilities" bars. Then the Grad-CAM.*

> "This is Grad-CAM. Red means 'this region pushed the score up'. The hotspot is
> on the lesions — so the decision is grounded in the symptoms, not in the
> background. That matters because PlantVillage images all have plain
> backgrounds, so a model can cheat; Grad-CAM is how we check that it doesn't."

### 1:30 — Advice (20 s)

> "Then the class retrieves treatment: remove affected lower leaves, copper or
> chlorothalonil every 7–10 days, mulch to stop splash — plus prevention. This
> is a local knowledge base for all 38 classes, so it works with no internet and
> no API key. There is an optional LLM toggle in the sidebar for extra narrative,
> but the demo does not depend on it."

### 1:50 — Second upload: healthy (20 s)

> "Now a healthy leaf. Predicted **healthy**, green badge, and the advice changes
> to routine care — this is the point: the model distinguishes 'no disease' from
> 'disease', which is the practically important binary."

### 2:10 — Numbers and honesty (30 s)

*Switch to the "Model & results" tab.*

> "24 million parameters total, but only 534 thousand are trainable in phase one,
> because the backbone stays frozen. We train the head first at 1e-4, then
> unfreeze the last 30 layers at 1e-5 and gain about two points. On a held-out
> test split: 94 %. The honest caveat: that is 94 % on PlantVillage, which is
> laboratory imagery — single leaf, plain background. Field photos will be
> harder. That's why we flag low confidence below 55 % and why we ship the
> heat-map."

### 2:40 — Hand over (20 s)

> "Everything here runs locally, offline, on a CPU. The code is six modules —
> preprocessing, model, Grad-CAM, advice, training and evaluation — and we have
> 13 unit tests plus a 9-point setup check. Where would you like to start?"

**Stop. Let them choose. Do not keep talking.**

---

## C. Who speaks when

| Section | Speaker | Backup |
|---|---|---|
| Opening + first upload | Abdul Rehman | Aniq |
| Preprocessing explanation | Uzair | Armghan |
| Architecture + transfer learning | Armghan | Abdul Rehman |
| Grad-CAM | Uzair | Armghan |
| Results, confusion matrix, limitations | Aniq | Abdul Rehman |
| Advice layer + deployment | Aniq | Uzair |

Rule: if a member is asked something outside their area, they answer **one
sentence** and hand over — "that's the preprocessing, Uzair will explain it
precisely." Handing over cleanly looks like a team; guessing looks like four
people who copied the same repository.

---

## D. Contingencies — memorise these three

### 1. The examiner uploads a photo that is not a leaf

The model **will** answer something — it is a closed-set classifier.

> "It answered, because it has to: it can only return one of 38 classes. That is
> the closed-set limitation — the softmax has no 'none of these' option. Notice
> the confidence is only 41 %, and our app flags anything under 55 %. A
> production version needs a 39th 'unknown' class or out-of-distribution
> detection. **Look at the heat-map — it is on the background, not on an object,
> which is exactly the symptom you'd expect.** It's in our limitations list."

This answer is *better* than a lucky correct prediction, because it proves you
understand the model's boundary.

### 2. It predicts the wrong disease for a real leaf

> "It's wrong, and here's why: [the two diseases share lesion shape / the photo
> is low contrast / more than one leaf in frame]. In our confusion matrix the
> cell Corn gray-leaf-spot → Northern leaf blight is the largest off-diagonal —
> biologically similar lesions. That's the class of error we'd attack next with
> more field data."

Never argue with the output. Diagnose it.

### 3. The app will not start

Stay calm, then:

```powershell
# Option 1 — restart cleanly
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8600

# Option 2 — no browser at all: show the CLI
.\.venv\Scripts\python.exe predict_cli.py samples\tomato_early_blight.jpg --gradcam out\cam.png
```

The CLI prints the same top-5 probabilities, the same advice, and saves the same
Grad-CAM image to `out\cam.png`, which you can open in any viewer. **Practice
this path once at home** so it is muscle memory.

Other quick fixes:

| Problem | Fix |
|---|---|
| "No model found" | `python scripts/get_weights.py --from <USB>\resnet_weights.npz` |
| Port already in use | `--server.port 8600` |
| Page won't load | close the terminal, restart `streamlit`, refresh browser |
| Very slow first prediction | normal (TF graph warm-up); do a dummy upload before the examiner arrives |
| Browser zoom broken | `Ctrl` + `+` |

---

## E. Test images to prepare (5 minutes)

Create a `samples/` folder with ~10 images copied from the test split, named by
their true class:

```
samples/
├── tomato_early_blight_01.jpg        ← Tomato___Early_blight
├── tomato_early_blight_02.jpg
├── tomato_healthy_01.jpg             ← Tomato___healthy
├── potato_late_blight_01.jpg         ← Potato___Late_blight
├── potato_early_blight_01.jpg
├── corn_common_rust_01.jpg           ← Corn_(maize)___Common_rust_
├── grape_black_rot_01.jpg            ← Grape___Black_rot
├── apple_scab_01.jpg                 ← Apple___Apple_scab
├── pepper_bacterial_spot_01.jpg      ← Pepper,_bell___Bacterial_spot
└── strawberry_healthy_01.jpg         ← Strawberry___healthy
```

Copy from `data/plantvillage_split/test/<class>/` (pick the first file in each
folder). These are images the model has **never trained on**, so if they work,
that is genuine generalisation and you can say so.

---

## F. Closing line (when the examiner is done)

> "To summarise: ResNet50 transfer learning, 38 classes, 94 % on PlantVillage,
> Grad-CAM for explainability, and a fully offline app. The main limitation is
> that it has only seen laboratory imagery — the next step is field data and an
> 'unknown' class. Thank you."

Then stop, and let them write down the marks.
