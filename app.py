"""
app.py — the live demonstration.

Run with:
    streamlit run app.py

Design decisions that matter on exhibition day
----------------------------------------------
* The model is loaded ONCE and cached (a 92 MB load per visitor would make the
  demo feel broken).
* No internet is required: the image comes from the visitor, the weights come
  from disk, and the care advice comes from a local knowledge base. The LLM box
  is optional and fails gracefully.
* Every failure mode has a human-readable message instead of a traceback: no
  weights, wrong file type, corrupt image, out-of-distribution picture.
* Uploaded images are normalised (EXIF rotation, transparency, greyscale) so a
  phone photo of a real leaf behaves like a PlantVillage image.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Make `import src...` work no matter which folder streamlit was started from.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advice import get_advice, human_readable, llm_advice  # noqa: E402
from src.config import IMG_SIZE, LAST_CONV_LAYER, load_class_names  # noqa: E402
from src.gradcam import overlay_heatmap  # noqa: E402
from src.preprocessing import prepare_upload  # noqa: E402
from src.quality import analyse as assess_image  # noqa: E402

# ---------------------------------------------------------------------------
# Page config + styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1250px; }
      #MainMenu, footer, header { visibility: hidden; }

      .hero {
        background: linear-gradient(120deg, #0f3d2e 0%, #17694d 55%, #2e9e6b 100%);
        color: #fff; padding: 1.4rem 1.7rem; border-radius: 16px; margin-bottom: 1rem;
      }
      .hero h1 { margin: 0 0 .25rem 0; font-size: 1.95rem; }
      .hero p  { margin: 0; opacity: .93; font-size: .98rem; }

      .verdict {
        border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: .7rem;
        border-left: 8px solid #2e9e6b; background: #eefaf3;
      }
      .verdict.sick { border-left-color: #d9534f; background: #fdeeee; }
      .verdict.warn { border-left-color: #e0a800; background: #fdf8e8; }
      .verdict .label { font-size: .74rem; letter-spacing: .09em; text-transform: uppercase;
                        color: #6b7280; margin-bottom: .2rem; }
      .verdict .name  { font-size: 1.5rem; font-weight: 700; color: #111827; line-height: 1.2; }
      .verdict .conf  { font-size: .95rem; color: #374151; margin-top: .35rem; }

      .pill { display:inline-block; padding:.15rem .6rem; border-radius:999px;
              font-size:.76rem; font-weight:600; margin-right:.35rem; }
      .pill.green { background:#d1f2e0; color:#0c6b45; }
      .pill.red   { background:#fadcdb; color:#9b2c2a; }
      .pill.amber { background:#fbeecb; color:#8a6100; }

      .card { border:1px solid #e5e7eb; border-radius:14px; padding:1rem 1.15rem; background:#fff; }
      .card h4 { margin:.1rem 0 .6rem 0; font-size:1rem; }
      small.muted { color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🌿 Plant Disease Detection</h1>
      <p>ResNet50 transfer learning · 38 PlantVillage classes · Grad-CAM explainability</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the ResNet50 model (first time only)…")
def load_everything():
    """Load the model once per server process and keep it in memory."""
    from src.gradcam import GradCAM
    from src.model import describe_model, load_inference_model

    model, path = load_inference_model(verbose=True)
    classes = load_class_names()
    gradcam = GradCAM(model, layer_name=LAST_CONV_LAYER)
    return model, path, classes, gradcam, describe_model(model)


def model_available() -> bool:
    try:
        load_everything()
        return True
    except Exception as exc:                      # noqa: BLE001 - shown to the user
        st.error("### Model weights not found")
        st.markdown(
            "The app needs a trained model in `models/`. Pick **one** of:\n\n"
            "```bash\n"
            "# A) quickest — download the ready-trained 92 MB weights\n"
            "python scripts/get_weights.py\n\n"
            "# B) train it yourself (PlantVillage, ~1-3 h on CPU)\n"
            'python prepare_dataset.py --src "D:/datasets/PlantVillage" '
            "--dst data/plantvillage_split\n"
            "python train.py --data data/plantvillage_split\n"
            "```"
        )
        with st.expander("Technical details"):
            st.code(f"{type(exc).__name__}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------
def predict(model, batch: np.ndarray) -> np.ndarray:
    """Return the 38-class probability vector for one preprocessed image."""
    return np.asarray(model.predict(batch, verbose=0)[0], dtype=np.float32)


def confidence_band(conf: float) -> tuple[str, str]:
    if conf >= 0.80:
        return "green", "High confidence"
    if conf >= 0.55:
        return "amber", "Moderate confidence"
    return "red", "Low confidence"


def report_payload(classes, probs, class_name, conf, quality=None) -> str:
    order = np.argsort(probs)[::-1][:5]
    payload = {
        "prediction": class_name,
        "crop": get_advice(class_name)["crop"],
        "disease": get_advice(class_name)["disease"],
        "confidence": round(float(conf), 4),
        "top5": [
            {"class": classes[int(i)], "probability": round(float(probs[int(i)]), 4)}
            for i in order
        ],
        "advice": get_advice(class_name),
    }
    if quality is not None:
        payload["leaf_check"] = quality
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tab 1: detection
# ---------------------------------------------------------------------------
def render_detect(model, classes, gradcam, *, show_gradcam, cam_alpha, top_k, use_llm) -> None:
    left, right = st.columns([1.05, 1], gap="large")

    with left:
        uploader = st.file_uploader(
            "Upload a leaf photo (JPG, PNG, WebP)",
            type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
            help="A single leaf, filling most of the frame, in good light, "
                 "gives the best result.",
        )

    if uploader is None:
        with left:
            st.info("**Waiting for an image.** Drop a leaf photo above, or take "
                    "one with your phone and upload it.")
            st.markdown(
                """
                **Good photos look like this**
                - one leaf, filling most of the picture
                - even daylight, no deep shadow
                - plain background if possible (the model learned on plain backgrounds)
                - the lesions in focus

                The model knows 38 classes across apple, blueberry, cherry, corn,
                grape, orange, peach, bell pepper, potato, raspberry, soybean,
                squash, strawberry and tomato.
                """
            )
        with right:
            st.markdown(
                '<div class="card"><h4>Pipeline in one line</h4>'
                '<small class="muted">photo → EXIF fix → RGB → 224×224 → BGR mean '
                'subtraction → ResNet50 → 2048-d features → Dense 256 → softmax 38 '
                '→ Grad-CAM → advice</small></div>',
                unsafe_allow_html=True,
            )
        return

    # ---- a file was provided --------------------------------------------
    try:
        display_img, batch = prepare_upload(uploader.getvalue())
    except Exception as exc:                      # noqa: BLE001
        st.error(f"Could not read that file as an image: {exc}")
        return

    probs = predict(model, batch)
    idx = int(np.argmax(probs))
    class_name = classes[idx]
    conf = float(probs[idx])
    advice = get_advice(class_name)
    band, band_msg = confidence_band(conf)
    quality = assess_image(display_img)

    with left:
        st.image(display_img, caption="Your upload", use_container_width=True)
        w, h = display_img.size
        st.caption(f"{w}×{h} px · resized to {IMG_SIZE[0]}×{IMG_SIZE[1]} for the network")

    # ---- leaf guard ------------------------------------------------------
    # The classifier is closed-set: it must return one of 38 classes, so a photo
    # of a desk or a screenshot still gets an answer, often at 99 % confidence.
    # This cheap colour check catches that instead of letting it stand.
    if quality["status"] == "not_leaf":
        st.error(
            "### This does not look like a leaf\n"
            f"{quality['reason'].capitalize()}. The model is a closed-set "
            "classifier — it **must** return one of its 38 leaf classes, so the "
            "result below is **not meaningful**. Upload a close-up of a single "
            "leaf, filling most of the frame."
        )
    elif quality["status"] == "uncertain":
        st.warning(
            f"Only about {quality['score'] * 100:.0f}% of this image is "
            "plant-coloured. If it is not a close-up of a leaf, treat the "
            "prediction with caution."
        )

    # ---- verdict card ----------------------------------------------------
    box = "" if advice["is_healthy"] else "sick"
    if band == "red" or quality["status"] == "not_leaf":
        box = "warn"
    state_pill = ('<span class="pill green">HEALTHY</span>' if advice["is_healthy"]
                  else '<span class="pill red">DISEASED</span>')
    st.markdown(
        f"""
        <div class="verdict {box}">
          <div class="label">Diagnosis</div>
          <div class="name">{advice['crop']} — {advice['disease']}</div>
          <div class="conf">{state_pill}<span class="pill {band}">{band_msg}</span>
            &nbsp; <b>{conf * 100:.1f}%</b> model confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(float(np.clip(conf, 0.0, 1.0)))

    if conf < 0.55:
        st.warning(
            "The model is not confident. Usually that means: the leaf is not one "
            "of the 38 known classes, the photo is blurry or badly lit, or more "
            "than one leaf is in the frame. Try a single, well-lit leaf."
        )

    with right:
        if show_gradcam:
            with st.spinner("Computing Grad-CAM…"):
                heatmap = gradcam.compute(batch, class_index=idx)
                overlay = overlay_heatmap(display_img, heatmap, alpha=cam_alpha)
            st.markdown("**Where the model is looking**")
            st.image(overlay,
                     caption="Red = strong evidence for the predicted class",
                     use_container_width=True)
            with st.expander("Why this matters"):
                st.markdown(
                    "Grad-CAM weights each of the 2048 feature channels of the last "
                    "convolution layer by how much it raised the winning class "
                    "score, then keeps only the positive contributions. A hot region "
                    "on the lesion means the decision is grounded in symptoms; a hot "
                    "region on the background means the model is exploiting dataset "
                    "bias (PlantVillage leaves were photographed on plain backgrounds)."
                )
        else:
            st.markdown("**Grad-CAM disabled** (enable it in the sidebar).")

    st.divider()

    # ---- probabilities + advice -----------------------------------------
    col_a, col_b = st.columns([1, 1.15], gap="large")
    with col_a:
        st.markdown("**Class probabilities**")
        for rank, i in enumerate(np.argsort(probs)[::-1][: int(top_k)], start=1):
            p = float(probs[int(i)])
            st.progress(float(np.clip(p, 0.0, 1.0)),
                        text=f"{rank}. {human_readable(classes[int(i)])} — {p * 100:.2f}%")

    with col_b:
        st.markdown("#### What to do next")
        st.markdown(
            f"**Pathogen:** {advice['pathogen']} &nbsp;|&nbsp; "
            f"**Severity:** {advice['severity']}",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Symptoms to confirm**")
            for s in advice["symptoms"]:
                st.markdown(f"- {s}")
        with c2:
            st.markdown("**Treatment now**")
            for n, t in enumerate(advice["treatment"], start=1):
                st.markdown(f"{n}. {t}")
        with c3:
            st.markdown("**Prevention**")
            for p in advice["prevention"]:
                st.markdown(f"- {p}")

    st.divider()

    # ---- optional LLM layer ---------------------------------------------
    if use_llm:
        st.markdown("#### AI narrative (optional)")
        if st.button("Generate Mistral-7B advice", type="secondary"):
            with st.spinner("Calling the Hugging Face Inference API…"):
                text = llm_advice(class_name)
            if text:
                st.success(text)
            else:
                st.info(
                    "The LLM layer is unavailable (no token or no internet). The "
                    "structured advice above came from the offline knowledge base "
                    "and is fully sufficient."
                )

    st.download_button(
        "⬇️ Download this result (JSON)",
        data=report_payload(classes, probs, class_name, conf, quality),
        file_name=f"diagnosis_{class_name.replace('___', '_')}.json",
        mime="application/json",
    )
    st.caption(
        "Research prototype trained on PlantVillage (laboratory-condition images). "
        "A decision-support demo, not a substitute for an agronomist."
    )


# ---------------------------------------------------------------------------
# Tab 2: how it works
# ---------------------------------------------------------------------------
def render_how() -> None:
    st.markdown(
        """
        ### The five stages

        **1. Preprocessing** — `src/preprocessing.py`
        The upload is opened with Pillow, rotated according to its EXIF tag,
        flattened to RGB, resized to 224×224 with bilinear interpolation,
        converted to float32, re-ordered from **RGB to BGR**, and the ImageNet
        channel means (103.939 / 116.779 / 123.68) are subtracted. That is exactly
        what `keras.applications.resnet50.preprocess_input` does; ResNet50 was
        trained with this convention, so we must reproduce it bit for bit. We do
        **not** divide by 255 — the network expects the raw mean-subtracted scale.

        **2. Feature extraction** — the ResNet50 backbone
        ResNet50 is a 50-layer convolutional network with residual ("skip")
        connections. Its last block outputs a 7×7×2048 tensor: 2048 different
        feature detectors sampled on a 7×7 grid. Those detectors already respond to
        edges, textures, colour gradients and blob-like structures — everything a
        leaf lesion has — because they were learned on 1.2 million ImageNet photos.
        We keep them frozen.

        **3. Classification head** — 3 small layers
        `GlobalAveragePooling2D` collapses 7×7×2048 to 2048 numbers (one average
        per detector; far fewer parameters than Flatten, and less overfitting).
        `Dense(256, relu)` learns plant-specific combinations of those features.
        `Dropout(0.4)` switches 40 % of them off at random during training — a
        cheap ensemble that fights overfitting.
        `Dense(38, softmax)` turns the 256 numbers into 38 probabilities summing
        to 1.

        **4. Grad-CAM** — `src/gradcam.py`
        Instead of trusting a number, we ask the network *where* it looked.

        **5. Advice** — `src/advice.py`
        The predicted class indexes a local knowledge base with pathogen,
        severity, symptoms, treatment and prevention. The LLM layer is optional.

        ---

        ### Numbers to quote

        | Item | Value |
        |---|---|
        | Backbone parameters | 23,587,712 |
        | Head parameters | 534,310 |
        | Total parameters | 24,122,022 |
        | Input tensor | 224 × 224 × 3 |
        | Output | 38 probabilities |
        | Weight file size | ~92 MB (float32) |
        | Classes / crops | 38 / 14 |
        | Grad-CAM target | `conv5_block3_out` (7×7×2048) |
        """
    )


# ---------------------------------------------------------------------------
# Tab 3: model + results
# ---------------------------------------------------------------------------
def render_eval(description: str) -> None:
    st.markdown("### Loaded architecture")
    st.code(description, language="text")

    st.markdown(
        """
        ### Reported results (from the reference run)

        | Model | Train | Val | Test |
        |---|---|---|---|
        | Baseline CNN (from scratch) | 83 % | 77 % | 79 % |
        | ResNet50 frozen backbone | 70 % | 89 % | 85 % |
        | ResNet50 fine-tuned (last 30 layers) | 83 % | 94 % | **94 %** |

        The frozen-backbone run trains faster but converges lower; unfreezing the
        last 30 layers at a 10× smaller learning rate adds about 2 points of
        validation accuracy. The train/test gap is small, so the model is not
        overfitting — PlantVillage is an "easy" dataset because every image is a
        single leaf on a uniform background.
        """
    )

    metric_file = PROJECT_ROOT / "reports" / "metrics.json"
    if metric_file.exists():
        st.markdown("### Your local run")
        data = json.loads(metric_file.read_text(encoding="utf-8"))
        cols = st.columns(4)
        for col, (label, key) in zip(
            cols,
            [("Test accuracy", "accuracy"), ("Macro F1", "macro_f1"),
             ("Images", "images"), ("Classes", "classes")],
        ):
            if key in data:
                value = data[key]
                col.metric(label, f"{value * 100:.2f}%" if key in ("accuracy", "macro_f1")
                           else f"{value:,}")
        with st.expander("Full metrics.json"):
            st.json(data)
        for name in ("training_curves.png", "confusion_matrix.png"):
            p = PROJECT_ROOT / "reports" / name
            if p.exists():
                st.image(str(p), caption=name)
    else:
        st.info(
            "No local metrics yet. After training or downloading the weights, run:\n\n"
            "`python evaluate.py --data data/plantvillage_split`\n\n"
            "and this tab shows your own accuracy, confusion matrix and curves."
        )


# ---------------------------------------------------------------------------
# Layout: sidebar first, then the three tabs
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Session control")
    st.caption("Everything runs locally — no internet needed for the demo.")

    show_gradcam = st.checkbox("Show Grad-CAM heat-map", value=True)
    cam_alpha = st.slider("Heat-map intensity", 0.15, 0.85, 0.45, 0.05)
    top_k = st.slider("How many classes to show", 3, 10, 5)

    st.divider()
    st.markdown("### Optional AI narrative")
    use_llm = st.checkbox(
        "Add Mistral-7B advice (needs internet + HF token)",
        value=False,
        help="The offline knowledge base is enough for the demo. This only adds "
             "an extra paragraph.",
    )

    st.divider()
    st.markdown("### About")
    st.markdown(
        "- **Backbone:** ResNet50 (ImageNet)\n"
        "- **Head:** GAP → Dense 256 → Dropout 0.4 → Softmax 38\n"
        f"- **Input:** {IMG_SIZE[0]}×{IMG_SIZE[1]} RGB, BGR mean-subtracted\n"
        f"- **Explainability:** Grad-CAM @ `{LAST_CONV_LAYER}`\n"
        "- **Dataset:** PlantVillage, 38 classes"
    )

if not model_available():
    st.stop()

model, model_path, classes, gradcam, description = load_everything()

with st.sidebar:
    st.divider()
    st.markdown("### Loaded model")
    st.caption(f"`{model_path.name}`")
    st.code(description, language="text")

tab_detect, tab_how, tab_eval = st.tabs(
    ["🔍 Detect", "⚙️ How it works", "📊 Model & results"]
)

with tab_detect:
    render_detect(
        model, classes, gradcam,
        show_gradcam=show_gradcam, cam_alpha=cam_alpha, top_k=top_k, use_llm=use_llm,
    )

with tab_how:
    render_how()

with tab_eval:
    render_eval(description)
