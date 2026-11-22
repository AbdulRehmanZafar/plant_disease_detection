# VIVA GUIDE — Plant Disease Detection

> **Read this before 30 November.** The brief says it plainly: *"Any student who
> fails to demonstrate a reasonable understanding of the project during the viva
> will receive ZERO marks."* A working demo is not enough. This document is
> organised so that each of the four members can be quizzed independently.

---

## PART 0 — How this is marked, and what that means for you

| Criterion | What the examiner actually does |
|---|---|
| Live demonstration | Uploads *their own* image and watches what happens |
| Implementation knowledge | Points at a file and says "explain this" |
| Technical understanding | Asks *why*, not *what*: why ResNet50, why freeze, why 224 |
| Viva performance | Asks follow-ups until you either explain or run out |

Three habits that separate a 10 from a 4:

1. **Always answer with a number.** Not "we used a small learning rate" but
   "1e-4 in phase 1, then 1e-5 in phase 2, a factor of ten smaller".
2. **Say what would break.** "If we forgot the RGB→BGR swap, accuracy drops
   because ResNet50 was trained in BGR." That sentence proves you understand the
   code rather than having copied it.
3. **Volunteer one limitation before being asked.** It changes the conversation
   from interrogation to discussion.

---

## PART 1 — The 60-second opening pitch (memorise the structure, not the words)

> "We built a plant disease detector. A user uploads a photo of a leaf and we
> classify it into one of 38 classes covering 14 crops — 26 diseases and 12
> healthy varieties.
>
> We use transfer learning. The backbone is ResNet50, pretrained on ImageNet,
> and it stays frozen; on top of it we add a small head — global average
> pooling, a 256-unit dense layer, dropout, and a 38-way softmax. That head is
> only 534 thousand parameters, so the whole model trains in hours on a CPU, not
> weeks.
>
> The model reaches 94 % on a held-out test split of PlantVillage. Because a
> probability alone is not auditable, we also generate a Grad-CAM heat-map that
> shows which part of the leaf drove the decision — if the hotspot is on the
> lesion, the prediction is grounded.
>
> Finally the predicted class retrieves treatment and prevention advice from a
> local knowledge base, so the whole application runs offline. The interface is
> a Streamlit app: the examiner can upload any photo and get a result in about a
> second."

Deliver it in ~50 seconds, then **stop talking** and let them upload something.

---

## PART 2 — Who owns what (individual assessment)

Everyone must be able to answer anything, but each member should be *unbeatable*
on their assigned area. These are the four natural seams in the codebase.

| Member | Primary area | Files they must know cold | Signature questions |
|---|---|---|---|
| **Abdul Rehman Zafar** | Dataset, splitting, training loop | `prepare_dataset.py`, `train.py` | "Why 70/15/15? Why augmentation? What is EarlyStopping doing?" |
| **Shiekh Armghan Ahmad** | Architecture & transfer learning | `src/model.py`, `src/config.py` | "Why ResNet50? What does freezing do? What a residual connection?" |
| **Uzair Irfan** | Preprocessing, inference, explainability | `src/preprocessing.py`, `src/gradcam.py` | "Why subtract the mean? Why BGR? Explain the Grad-CAM formula." |
| **Aniq Ahmad** | Evaluation, deployment, advice layer | `evaluate.py`, `app.py`, `src/advice.py` | "What is a confusion matrix? What do precision and recall mean here?" |

Suggested drill: each member presents their area for 3 minutes to the other
three, who then ask two hostile questions each. Rotate. Two rounds is enough.

---

## PART 3 — Questions and answers

### A. Problem and dataset

**Q: What is the problem, in one sentence?**
Classify a photo of a plant leaf into one of 38 disease/healthy categories, to
give a farmer a diagnosis without waiting for an expert.

**Q: Why does this problem matter?**
Plant disease causes large yield losses, and expert diagnosis is slow, expensive
and not scalable to millions of small farms. A phone-based classifier is
instant, free and available anywhere.

**Q: What dataset did you use?**
PlantVillage — 54,305 labelled leaf images, 38 classes across 14 crops, released
by Hughes and Salathé in 2015. We use the `color` variant (the download also
contains `grayscale` and `segmented`).

**Q: How many classes and how many are healthy?**
38 classes: 12 healthy (`*___healthy`) and 26 diseased.

**Q: How is the dataset imbalanced, and how did you handle it?**
Class sizes range from 275 images (Apple cedar apple rust) to 5,507 (Orange
HLB) — a factor of 20. We handle it with (a) a stratified split so small classes
are represented in every split, (b) augmentation on the training set only, and
(c) optionally `--class-weights`, which scales the loss by inverse class
frequency so the model cannot win by ignoring rare classes.

**Q: What is the split ratio and why?**
70 % train, 15 % validation, 15 % test — 38,010 / 8,146 / 8,149 images.
Validation is used *during* training for early stopping and learning-rate
decisions; test is touched once, at the end. If you tune on the test set, the
number you report is no longer an honest estimate.

**Q: What is data leakage and how did you avoid it?**
Leakage is when information from the test set influences training — for example
the same photo appearing in both, or augmenting *before* splitting so that two
versions of one leaf land on both sides. We split the **original files first**,
then augment only the training split, in memory, per epoch. Verified: zero file
appears in two splits.

**Q: Why not use the `segmented` variant, where the background is removed?**
It would remove the background-colour shortcut and probably reduce a real-world
failure mode. We kept `color` because the published weights were trained on
`color`, and we wanted a consistent pipeline. This is a good "future work" item.

**Q: What is wrong with PlantVillage as a benchmark?**
Every image is a single detached leaf photographed on a uniform background in
controlled lighting. Real field photos have soil, multiple leaves, shadows,
motion blur and insects. Accuracy on PlantVillage overstates field performance —
this is the best-known criticism of the dataset and you should say it yourself.

---

### B. Preprocessing

**Q: Walk me through what happens to an uploaded image.**
1. Pillow opens the file; EXIF orientation is applied (phone photos are often
   stored rotated); transparency is flattened onto white; it is converted to RGB.
2. It is resized to 224×224 bilinear.
3. Cast to `float32` — *before* any subtraction.
4. RGB channels are reversed into BGR order.
5. The ImageNet means `[103.939, 116.779, 123.68]` are subtracted per channel.
6. A batch dimension is added: `(1, 224, 224, 3)`.

**Q: Why 224×224?**
That is ResNet50's native input size and the size its ImageNet weights were
trained on. The architecture can accept other sizes (it is fully convolutional),
but our weights and the `7×7×2048` feature map assume 224. Changing it would
also change Grad-CAM's spatial resolution.

**Q: Why subtract the mean, and why those exact numbers?**
The mean subtraction centres the input around zero, which keeps activations in
the range the pretrained filters expect. The numbers are the per-channel means
of the ImageNet training set in BGR order. Using the wrong vector — or skipping
it — moves every input far from the distribution the network was trained on and
the accuracy collapses.

**Q: Why not divide by 255 as well?**
Because `preprocess_input` for ResNet50 does not. It only mean-subtracts. If you
divide as well, every activation is ~255× too small and the network sees
essentially a black image.

**Q: Why BGR and not RGB?**
ResNet50 was trained with the Caffe convention, which reads images as BGR.
Keras's `preprocess_input` performs `x[..., ::-1]` for exactly this reason.
Feeding RGB swaps the red and blue detectors — the model still runs and still
gives plausible-looking output, which is why this bug is dangerous.

**Q: How do you know your preprocessing matches Keras?**
`scripts/check_setup.py` generates random images, runs both our NumPy function
and `keras.applications.resnet50.preprocess_input`, and prints the maximum
absolute difference. It must be zero (or ~1e-6). We also unit-test the
white-image and black-image edge cases in `tests/selftest.py`.

**Q: What is the uint8 wrap-around trap?**
`x` is `uint8` in [0,255]. Computing `x - 103.9` in uint8 wraps negative values
around to ~152 (`0 - 104 mod 256`). You must cast to float first. Our test
asserts that the minimum value after preprocessing is negative — if it were not,
the cast happened in the wrong order.

**Q: What augmentation did you use and why those transforms?**
Horizontal flip, rotation ±21°, zoom 10 %, contrast 10 %, brightness 10 %.
These simulate camera angle and lighting variation without changing what the
leaf *is*. We deliberately excluded vertical flips and 90° rotations: leaves
have a natural orientation and PlantVillage is consistently oriented, so those
transforms teach invariance to something that never varies and can hurt.

**Q: Augmentation on the test set?**
Never. Evaluation must measure performance on unmodified images. Our
`make_augmentation()` layer is applied only in the training `.map()`.

**Q: Images are resized to 224 with bilinear — could that lose the lesion?**
Small lesions can lose contrast, which is a genuine limitation. Alternatives:
higher input resolution (costs accuracy-compatible weights), or random crops
from a larger image. We accept it because the baseline needs consistency with
the pretrained weights.

---

### C. Architecture

**Q: What is a CNN, briefly?**
A network of learnable filters. Each convolutional layer slides small kernels
over the image and produces feature maps; stacking them builds a hierarchy —
edges, then textures, then object parts. Pooling/stride reduces spatial size and
increases the receptive field. Fully connected layers at the end combine
features into a decision.

**Q: Why ResNet50?**
Three reasons: (1) it is deep enough (50 layers) to represent lesion texture and
shape yet small enough to fine-tune on a CPU; (2) residual connections make it
actually trainable at that depth; (3) ImageNet-pretrained weights are widely
available and its 7×7×2048 final feature map is a good trade-off for Grad-CAM
(coarse enough to be stable, fine enough to be informative).

**Q: What is a residual connection and why does it matter?**
`output = F(x) + x`. The input is added to the transformed output, so gradients
flow straight back through the identity path. This solves the vanishing-gradient
/ degradation problem that made networks deeper than ~20 layers *worse* than
shallower ones. In ResNet50 the addition happens inside every block, and when
the spatial size changes the shortcut uses a 1×1 convolution to match dimensions.

**Q: What is a bottleneck block?**
ResNet50's block is 1×1 → 3×3 → 1×1 convolutions. The first 1×1 squeezes
channels (e.g. 256 → 64), the 3×3 does the expensive spatial work at low
channel count, the last 1×1 expands back (64 → 256). Same receptive field, a
fraction of the FLOPs.

**Q: What is Batch Normalization?**
It normalises each channel's activations across the batch to zero mean and unit
variance, then applies learnable scale/shift. Benefits: faster convergence,
higher stable learning rates, mild regularisation. Important subtlety: when
fine-tuning a small dataset we keep BN layers **frozen** so their moving
statistics do not drift away from the ImageNet values.

**Q: Why `include_top=False`?**
The top is the ImageNet 1000-class classifier. We want the *features*, not
ImageNet's labels, so we drop it and attach our own head for 38 classes.

**Q: What does the head do, layer by layer?**
* `GlobalAveragePooling2D` — averages each of the 2048 channels over the 7×7
  grid → 2048 numbers. Translation-robust, and only 0 parameters.
* `Dense(256, relu)` — learns combinations of those features that are specific
  to leaves and diseases. 2048×256 + 256 = 524,544 parameters.
* `Dropout(0.4)` — during training, 40 % of those 256 activations are zeroed at
  random, so the network cannot rely on any single unit. A cheap ensemble.
* `Dense(38, softmax)` — 256×38 + 38 = 9,766 parameters. Softmax converts
  scores into probabilities that are positive and sum to 1.

**Q: GAP or Flatten — why GAP?**
Flatten on 7×7×2048 gives 100,352 inputs to the next layer — 50 M+ parameters,
which would overfit 54k images and blow up the model. GAP averages spatially
instead: 2048 inputs, zero parameters, and it keeps translation invariance.
Trade-off: it throws away precise spatial arrangement, which is acceptable for
"texture on a leaf".

**Q: How many parameters in total?**
Backbone 23,587,712 + head 534,310 = **24,122,022**. In phase 1 only the 534,310
head parameters are trainable (2.2 %). `scripts/check_setup.py` prints these
numbers from your machine.

**Q: Why softmax and not sigmoid?**
The classes are mutually exclusive — one leaf is exactly one of the 38 classes.
Softmax enforces exactly that (probabilities sum to 1), sigmoid would treat the
38 classes as 38 independent yes/no questions.

**Q: Why `relu` in the dense layer?**
It is cheap, non-saturating for positive values, and keeps gradients flowing.
Without a non-linearity the two dense layers would collapse into one linear map.

---

### D. Transfer learning and fine-tuning

**Q: What is transfer learning?**
Reusing a network trained on one task (ImageNet, 1.2 M images, 1000 classes) as
the starting point for a different task (leaves, 38 classes). Low- and mid-level
features — edges, colour blobs, textures — are generic and transfer well; only
the final task-specific layers must be relearned.

**Q: Why not train from scratch?**
Two reasons. First, data: 54k images is small for deep learning; ImageNet has
1.2 M. Second, compute: training ResNet50 from scratch takes many GPU-days,
while training the head takes minutes on a CPU. Our own baseline CNN, trained
from scratch on the same data, reached 79 % test accuracy versus 94 % for the
fine-tuned ResNet50 — that is the empirical answer.

**Q: What does "freezing" mean mechanically?**
`layer.trainable = False`. The layer still runs forward, but no gradient is
computed for its weights and the optimizer does not update them. It reduces
trainable parameters from 24 M to 0.53 M.

**Q: Why freeze first and fine-tune later, instead of unfreezing immediately?**
At the start the new head is randomly initialised, so it produces large,
meaningless gradients. If the backbone were trainable, those gradients would
propagate into the pretrained filters and destroy the very features we want to
reuse (catastrophic forgetting). Freeze first, let the head converge, then
unfreeze gently with a 10× smaller learning rate.

**Q: Why is the phase-2 learning rate 10× smaller?**
Fine-tuning is a small correction to already-good weights. A large learning rate
takes big steps away from that good solution; a small one nudges it toward the
new domain. Standard practice: 1e-5 versus 1e-4.

**Q: Why keep BatchNorm frozen during fine-tuning?**
BN's moving statistics (mean/variance) are estimated from data. With a small new
dataset and batch size 32, those statistics become noisy and shift away from
ImageNet's, and the frozen-then-retrained layers disagree with each other — a
well-documented accuracy killer. Keeping BN frozen preserves the calibration.

**Q: How many layers did you unfreeze?**
The last 30 of the ResNet50 backbone, which covers the final stage's blocks —
the layers whose features are most task-specific (ImageNet: "dog fur"; leaves:
"lesion margin").

**Q: What happens if you unfreeze all layers?**
With 54k images and no GPU it is slow, and without a small LR it overfits and
degrades. The typical symptom is validation accuracy dropping while training
accuracy rises.

---

### E. Training loop

**Q: Which optimizer, which loss, and why?**
Adam (adaptive per-parameter learning rates; robust default, fast convergence on
small datasets) and `SparseCategoricalCrossentropy` (multi-class, integer
labels — the dataset yields labels as ints, so no one-hot encoding is needed).

**Q: Which callbacks, and what does each do?**
* `ModelCheckpoint` — saves the model file whenever `val_accuracy` improves.
* `EarlyStopping(patience=4, restore_best_weights=True)` — stops if validation
  accuracy has not improved for 4 epochs, and rolls the weights back to the best
  epoch, so you never keep an overfitted final epoch.
* `ReduceLROnPlateau(factor=0.2, patience=2)` — if validation loss stalls,
  multiply the LR by 0.2, letting the model settle into a minimum.
* `CSVLogger` — writes per-epoch metrics, which is what the training curves are
  drawn from.

**Q: Why `restore_best_weights=True`?**
Because the last epoch is not necessarily the best one. Without it you would
report a worse model than you actually had.

**Q: What is a batch, an epoch, a step?**
Batch = 32 images processed together (one gradient update). Epoch = one full
pass over the training set (~1,188 batches for 38,010 images). Step = one
optimizer update.

**Q: What is overfitting, and how is it prevented here?**
Overfitting is memorising the training set instead of learning generalisable
patterns — training accuracy much higher than validation accuracy. Prevented by:
transfer learning (few trainable parameters), dropout 0.4, augmentation,
early stopping, and a small phase-2 learning rate. Test accuracy (94 %) is close
to training accuracy, so we are not overfitting in practice.

**Q: What is underfitting, and do you see it?**
Training accuracy far below validation — the model is too weak or too
constrained. The frozen-backbone run shows this: 70 % train vs 89 % validation,
because the tiny head cannot fully fit the data while the frozen features
generalise well. Fine-tuning removes it.

**Q: How long does training take?**
Phase 1 (head only, CPU): roughly 25–40 minutes per epoch; 12 epochs plus
fine-tuning is a few hours. On a GPU (Colab T4) it is 10–20× faster. A
`--per-class-limit 300` subset trains in about 30 minutes on CPU.

---

### F. Evaluation and results

**Q: Which metric is primary, and why?**
Accuracy — the classes are roughly balanced in *importance*, so a single number
is meaningful. But because the dataset is imbalanced in *count*, we also report
macro-F1 (the unweighted mean of per-class F1), which treats a rare class as
equal to a common one. If accuracy and macro-F1 disagree strongly, the model is
ignoring rare classes.

**Q: Precision vs recall — define them for this task.**
Precision = of the leaves we *called* "Tomato late blight", how many really
were? Recall = of the real late-blight leaves, how many did we find? In
agriculture, a **false negative** (saying healthy when it is diseased) is usually
costlier than a false positive, so recall on diseased classes matters most.

**Q: What is F1?**
The harmonic mean of precision and recall: `2PR/(P+R)`. It is high only when
both are high, so a model cannot game it by predicting one class for everything.

**Q: Macro vs weighted average?**
Macro averages each class equally (a 275-image class counts as much as a
5,507-image one). Weighted averages by class support (dominated by big classes).
Report both; macro exposes the rare-class problem.

**Q: What does a confusion matrix show?**
An N×N grid: cell (i, j) = number of images whose true class is i and predicted
class is j. The diagonal is correct predictions. The row-normalised version is
per-class recall. Our most frequent off-diagonal cells are the useful talking
point: for example Corn gray leaf spot vs Northern leaf blight (both long grey
lesions) and Tomato early vs late blight — biologically similar lesions, which
is exactly where a CNN struggles.

**Q: What are your actual results?**
Reported reference run: baseline CNN 79 % test, frozen ResNet50 85 %, fine-tuned
ResNet50 **94 %**. Your own numbers are in `reports/metrics.json` and the
confusion matrix in `reports/confusion_matrix.png`. **Quote your own run**, and
say "94 % is the published reference; our run gave X %".

**Q: Why is 94 % not the whole story?**
Because it is accuracy on laboratory images from the same distribution. The
honest claim is "94 % on PlantVillage", not "94 % reliable in a field". Fold
that limitation into your answer before the examiner asks.

**Q: Can it tell me the leaf is not a plant at all?**
Not reliably. It is a closed-set classifier: it must output one of 38 classes.
An image of a shoe still gets a confident-looking softmax. We mitigate this
with a low-confidence warning below 55 %, but a proper solution is an
"unknown/other" class or out-of-distribution detection. Say this — it is a
classic follow-up.

---

### G. Grad-CAM

**Q: What is Grad-CAM for?**
It produces a coarse heat-map showing which image regions most increased the
score of the predicted class. It turns "the model said so" into "the model was
looking at the lesion", which is what makes the result auditable.

**Q: Explain the maths.**
Take the feature maps `A` of the last convolution layer (7×7×2048). Compute the
gradient of the class score `y^c` with respect to `A`. Average that gradient over
the spatial dimensions to get one importance weight `alpha_k` per channel. Take
a weighted sum of the channels, apply ReLU (keep only evidence *for* the class),
normalise to [0,1], upsample to image size, colour-map, and blend with the
original.

**Q: Why the last convolutional layer?**
It is the deepest layer that still has spatial layout. Deeper than that is the
GAP, where spatial information is averaged away. Shallower layers are too
low-level to be interpretable.

**Q: Why ReLU on the heat-map?**
Negative values mean "this region argues *against* the class". We are explaining
why the class was chosen, so we keep only positive contributions.

**Q: Why normalise by the maximum?**
Because the raw scale of the weighted sum is arbitrary; dividing by the max maps
it to [0,1] so the colour map is comparable across images.

**Q: What are the limits of Grad-CAM?**
It is coarse (a 7×7 grid upsampled), it is not a segmentation mask, it is not
guaranteed faithful, and it explains one class at a time. Different runs of the
same class on similar images can produce different maps.

**Q: What would a bad heat-map tell you?**
That the model is using dataset bias — e.g. the heat on the background instead
of the leaf. Because PlantVillage leaves sit on uniform backgrounds, background
colour is genuinely predictive; Grad-CAM is how we detect that.

**Q: Did you implement it or import it?**
Implemented in `src/gradcam.py` with `tf.GradientTape` — a Keras model with two
outputs (feature maps, probabilities), one gradient call, then the weighted sum.
No external Grad-CAM library. The colour mapping uses matplotlib instead of
OpenCV, which removes a heavy dependency and its Windows DLL issues.

---

### H. Application and deployment

**Q: What does the app do, step by step?**
Streamlit renders the page; the uploaded file is read as bytes; our preprocessing
turns it into a `(1,224,224,3)` tensor; `model.predict` returns 38
probabilities; the top class indexes the knowledge base; Grad-CAM runs on the
same tensor; the UI renders verdict, probabilities, heat-map and advice.

**Q: Why Streamlit?**
It turns a Python script into a web app with no front-end code and no server
work — ideal for a live demo, and it runs entirely locally.

**Q: Why is the model cached?**
`@st.cache_resource` keeps the loaded model in memory for the whole server
process. Streamlit re-runs the script on every interaction; without caching we
would reload 92 MB on every slider move.

**Q: How long does one prediction take?**
About 0.3–0.6 s on CPU for the forward pass, plus Grad-CAM (one extra gradient
pass). The first prediction is slower (~5–20 s) because TensorFlow builds and
warms up its graph.

**Q: Why does the app build the backbone with `weights=None`?**
Because we immediately overwrite every weight from our file. Asking Keras for
ImageNet weights would trigger a 95 MB download from Google's servers — which is
exactly what fails when the exhibition Wi-Fi is down. This is a deliberate
offline-first decision and a good thing to mention.

**Q: What happens if the model file is missing?**
The app shows a page with the two fix commands instead of crashing with a
traceback. A demo should fail politely.

**Q: What if the visitor uploads a PNG with transparency, or a rotated phone
photo?**
Transparency is flattened onto white and EXIF rotation is applied before
resizing. Without that, a rotated leaf looks like a different texture and the
prediction degrades for a boring, non-model reason.

**Q: Why did you replace the LLM with a knowledge base?**
The original approach called a hosted Mistral-7B model for care advice. That
makes the demo depend on the network, an API token and a rate limit — three
things that fail under exam pressure. We kept the LLM as an optional toggle
(`src/advice.py::llm_advice`) and made the offline knowledge base the default:
38 entries with pathogen, severity, symptoms, treatment and prevention.

**Q: How would you deploy this for real farmers?**
Export to TensorFlow Lite with post-training quantisation (~24 MB → ~6 MB),
embed in an Android app, run fully offline; add an "unknown" class and
out-of-distribution detection; validate on field photos before claiming field
accuracy.

---

### I. Limitations and future work (say these before they are asked)

1. **Lab-images only.** PlantVillage leaves are detached, single, on plain
   backgrounds. Field photos will be harder. *Fix:* fine-tune on PlantDoc or
   field data, or use segmentation to remove backgrounds.
2. **Closed set.** It cannot say "not a leaf". *Fix:* 39th "unknown" class, or
   an energy/Mahalanobis OOD score with a confidence threshold.
3. **Overconfidence.** Softmax probabilities are poorly calibrated — 70 %
   "confidence" does not mean 70 % of such predictions are right. *Fix:*
   temperature scaling on the validation set.
4. **Co-occurring diseases.** One label per image; real leaves can have two.
   *Fix:* multi-label classification with sigmoid outputs.
5. **Severity is not estimated.** We say *which* disease, not *how bad*.
   *Fix:* ordinal regression or lesion-area segmentation.
6. **Class imbalance** still influences predictions on the rare classes; macro-F1
   is lower than accuracy by a few points, and that gap is the honest measure.
7. **No treatment personalisation** by region, climate or available products.
8. **Privacy / safety**: advice is generic; a wrong diagnosis can cost a farmer
   a season. The app says "decision support, not a substitute for an agronomist".

---

### J. "What if I change X?" — trade-off questions

| Change | Expected effect |
|---|---|
| Remove the mean subtraction | Accuracy collapses — inputs far outside training distribution |
| Feed RGB instead of BGR | Noticeable accuracy drop, especially red/brown lesions |
| Freeze *and* never fine-tune | ~85 % instead of ~94 % |
| Unfreeze everything at LR 1e-3 | Divergence or heavy overfitting |
| Batch size 8 instead of 32 | Noisier gradients, worse BatchNorm statistics |
| Remove dropout | Training accuracy rises, validation accuracy falls |
| Train from scratch (no ImageNet) | Much slower and lower accuracy at this dataset size |
| Use `Flatten` instead of GAP | +50 M parameters, overfits, slower |
| Replace softmax with sigmoid | Probabilities no longer sum to 1; loss/labels must change |
| Increase input to 320×320 | Possible accuracy gain, but pretrained weights and Grad-CAM grid change; slower |
| Add more augmentation (vertical flip) | Little gain, possibly worse — leaves have orientation |
| Delete EarlyStopping | Risk of keeping an overfitted last epoch |

---

## PART 4 — "Explain this line" (be ready to read code aloud)

**1.** `base_model.trainable = False`
→ Freezes every weight in the ResNet50 backbone; gradients are not computed for
them, so the optimizer cannot change them. Trainable parameters drop from 24 M
to 0.53 M.

**2.** `x = L.GlobalAveragePooling2D(name="gap")(x)`
→ Averages each of the 2048 channels over the 7×7 grid → a 2048-vector with no
parameters. Chosen over Flatten to avoid ~50 M extra parameters.

**3.** `model.set_weights(arrays)`
→ Copies raw NumPy arrays into the model's weight tensors, in
`get_weights()` order. We validate every shape first (`compare_shapes`), because
a one-position shift silently corrupts the model.

**4.** `for key in weights` (the original project's line, and why we changed it)
→ It relied on the npz dictionary's iteration order, which the format does not
guarantee. `arr_10` can sort before `arr_2` under a string sort. We sort keys
numerically in `src/npz_utils.py` and verify shapes.

**5.** `with tf.GradientTape() as tape:` … `tape.gradient(loss, conv_output)`
→ Records the forward operations so TensorFlow can differentiate the class score
with respect to the feature maps. That gradient *is* the Grad-CAM signal.

**6.** `pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))`
→ Averages the (1,7,7,2048) gradient over the batch and both spatial axes,
producing one scalar importance per channel.

**7.** `heatmap = tf.maximum(heatmap, 0.0)`
→ ReLU: keep evidence for the class, discard evidence against it.

**8.** `x -= _MEAN_BGR`
→ In-place mean subtraction in BGR order; the core of `preprocess_input`.

**9.** `keras.callbacks.EarlyStopping(..., restore_best_weights=True)`
→ Stops after N epochs without improvement and restores the best epoch's
weights, so the saved model is the best one, not the last one.

**10.** `unfreeze_last_layers(..., keep_bn_frozen=True)`
→ Unfreezes the last 30 backbone layers but keeps BatchNorm frozen so its moving
statistics do not drift on a small dataset.

**11.** `@st.cache_resource`
→ Streamlit reruns the script on every widget change; this decorator keeps one
model instance alive for the whole session.

**12.** `if conf < 0.55: st.warning(...)`
→ An honest uncertainty flag. Closed-set classifiers are overconfident; this is
our cheap mitigation and a talking point for OOD detection.

---

## PART 5 — Numbers cheat sheet (memorise)

| Number | Meaning |
|---|---|
| 38 | classes |
| 14 | crops |
| 54,305 | images in PlantVillage |
| 26 / 12 | diseased / healthy classes |
| 275 … 5,507 | smallest … largest class |
| 70 / 15 / 15 | train / val / test split |
| 38,010 | training images |
| 224 × 224 × 3 | model input |
| 23,587,712 | backbone parameters |
| 534,310 | head parameters |
| 24,122,022 | total parameters |
| 2.2 % | share of trainable parameters in phase 1 |
| 7 × 7 × 2048 | last conv feature-map shape |
| 1e-4 → 1e-5 | phase-1 → phase-2 learning rate |
| 30 | backbone layers unfrozen in phase 2 |
| 32 | batch size |
| 0.4 | dropout rate |
| 0.2 | ReduceLROnPlateau factor |
| 4 | EarlyStopping patience |
| 79 % / 85 % / 94 % | baseline / frozen / fine-tuned test accuracy |
| 96,567,102 bytes | weight file size (~92 MB) |
| 0.3–0.6 s | per-image inference on CPU |

---

## PART 6 — Traps that cost marks

1. **"We used AI to write it."** Never say that. Say *what each file does* and
   that you can reproduce the pipeline from scratch. If asked about tooling,
   answer honestly but redirect to the technical content.
2. **"It is 94 % accurate, so it works everywhere."** Wrong and easily
   dismantled. Say "94 % on PlantVillage, which is laboratory imagery".
3. **Not knowing your own numbers.** If you quote 94 %, know that it is test
   accuracy and that train is 83 % in the fine-tuned run.
4. **Calling the class list arbitrary.** The order is alphabetical and it *must*
   match the weights; if it changes, index 12 means a different disease and the
   model is broken without any error message. This is one of our best answers.
5. **Confusing validation and test.** Validation tunes; test reports, once.
6. **Not being able to run the demo yourself.** Every member must be able to
   start the app from a cold PowerShell window. Practise it.
7. **Silence when the model is wrong.** If a visitor uploads something weird and
   the prediction is nonsense, say why — closed-set classifier, lab-trained,
   Grad-CAM will show what it latched onto. That is a *better* answer than a
   correct prediction.

---

## PART 7 — Five-minute self-test before the viva

Answer these out loud, without notes. If you stumble, reread the matching part.

1. How many classes, crops, and images?
2. Walk through preprocessing in six steps, with the three mean values.
3. Why BGR, and what breaks if you forget?
4. Draw the architecture with shapes at each stage.
5. How many trainable parameters in phase 1, and why so few?
6. Why freeze before fine-tuning, and why a 10× smaller LR?
7. What do EarlyStopping and ReduceLROnPlateau each do?
8. Define precision, recall, macro-F1 for this task.
9. Explain Grad-CAM in five lines of maths.
10. Name three limitations and one fix for each.
11. What happens if the visitor uploads a photo of a shoe?
12. Where does the advice text come from, and does it need internet?

If all twelve come out fluently, you are ready.

---

## Appendix — One-minute file map (say it like this)

> "`app.py` is the UI. `src/preprocessing.py` turns any image into a tensor.
> `src/model.py` builds ResNet50 plus our head and loads the weights — from
> `.keras`, or from the published `.npz` with shape validation.
> `src/gradcam.py` explains a prediction. `src/advice.py` maps the class to
> treatment. `train.py` does the two-phase transfer learning,
> `prepare_dataset.py` makes the stratified split, `evaluate.py` produces the
> metrics and confusion matrix, and `scripts/check_setup.py` is our pre-flight
> test — nine checks that prove the whole pipeline works before the examiner
> arrives."
