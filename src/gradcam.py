"""
gradcam.py — Grad-CAM: "which part of the leaf made the model decide that?"

Why we need it (marks reasoning, not decoration):
    A softmax probability alone is not auditable. If the model says "Tomato late
    blight" for a picture of a shoe, the number looks confident and meaningless.
    Grad-CAM shows the *evidence*: a heat-map of the image regions whose
    activations pushed the winning class score up. If the red blob sits on a
    lesion, the prediction is grounded; if it sits on the background, we know the
    model is cheating (a well-known PlantVillage bias — the dataset was shot on
    uniform backgrounds, so background colour alone is partly predictive).

THE MATH (5 lines, worth memorising):

    A      = feature maps of the last conv layer          (7, 7, 2048)
    y^c    = score of class c before softmax
    alpha_k = (1/Z) * sum_{i,j} d y^c / d A^k_{ij}        (importance of channel k)
    L^c_{ij} = ReLU( sum_k alpha_k * A^k_{ij} )           (weighted sum, keep positives)
    L^c      = L^c / max(L^c)                              (normalise to 0..1)

    Then upsample 7x7 -> image size and colour-map it.

Why the last conv layer: it keeps spatial information (7x7) while being
semantically deep. Earlier layers are too low-level; after GAP the spatial
layout is gone.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .config import LAST_CONV_LAYER


def find_layer(model, name: str):
    """
    Find a layer by name, including layers *inside* a nested backbone.

    Subtle but important: in Keras, ``model.get_layer("conv5_block3_out")`` only
    searches the model's top-level layers. Our ResNet50 backbone is nested as a
    single layer called "resnet50", so the last conv block is *not* a top-level
    layer and a plain ``get_layer`` raises ValueError. This helper walks one
    level down into nested models, which makes the code work whether you build
    the graph with ``input_tensor=`` (nested backbone) or by iterating layers.
    """
    try:
        return model.get_layer(name)
    except (ValueError, AttributeError):
        pass

    for layer in model.layers:
        if hasattr(layer, "layers") and layer is not model:
            try:
                return layer.get_layer(name)
            except (ValueError, AttributeError):
                continue

    available = sorted(
        l.name for l in model.layers
        if "conv" in l.name or "predictions" in l.name
    )
    nested = [
        f"{l.name}: {[x.name for x in l.layers][-3:]}"
        for l in model.layers if hasattr(l, "layers")
    ]
    raise ValueError(
        f"Layer '{name}' not found.\n"
        f"Top-level conv/output layers: {available}\n"
        f"Nested models (last 3 layer names): {nested}"
    )


class GradCAM:
    """
    Reusable Grad-CAM explainer.

    The second Keras model (``grad_model``) is built once and cached: rebuilding
    it on every upload is slow and leaks memory over a whole exhibition day.
    """

    def __init__(self, model, layer_name: str = LAST_CONV_LAYER):
        self.model = model
        self.layer_name = layer_name
        self._grad_model = None
        self._layer = None

    # -- internal ---------------------------------------------------------
    def _build(self):
        from tensorflow import keras

        self._layer = find_layer(self.model, self.layer_name)

        # Same input as the full model, but two outputs: the feature maps and
        # the class probabilities.
        # NOTE: pass `model.input` (a single tensor) rather than `model.inputs`
        # (a list). Keras 3 warns "the structure of inputs doesn't match the
        # expected structure" when a single-input model is called with a list.
        model_input = self.model.input
        if isinstance(model_input, (list, tuple)) and len(model_input) == 1:
            model_input = model_input[0]
        self._grad_model = keras.Model(
            inputs=model_input,
            outputs=[self._layer.output, self.model.output],
        )

    def _ensure(self):
        if self._grad_model is None:
            self._build()

    # -- public -----------------------------------------------------------
    def compute(self, batch: np.ndarray, class_index: int | None = None) -> np.ndarray:
        """
        Parameters
        ----------
        batch : (1, 224, 224, 3) preprocessed float32 array
        class_index : which class to explain; None = the predicted one

        Returns
        -------
        heatmap : (7, 7) float32 in [0, 1]
        """
        import tensorflow as tf

        self._ensure()
        batch = np.asarray(batch, dtype=np.float32)

        with tf.GradientTape() as tape:
            conv_output, predictions = self._grad_model(batch, training=False)
            if class_index is None:
                class_index = int(tf.argmax(predictions[0]))
            # Score of the class we are explaining.
            loss = predictions[:, class_index]

        # d(score) / d(feature maps)  ->  (1, 7, 7, 2048)
        grads = tape.gradient(loss, conv_output)
        if grads is None:
            raise RuntimeError(
                "Grad-CAM got no gradients. The target layer is probably not "
                "connected to the model output."
            )

        # Global-average the gradients over space -> one weight per channel.
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))      # (2048,)

        conv = conv_output[0]                                     # (7,7,2048)
        heatmap = tf.reduce_sum(conv * pooled_grads, axis=-1)     # (7,7)

        # ReLU: we only care about evidence FOR the class, not against it.
        heatmap = tf.maximum(heatmap, 0.0)

        max_val = tf.reduce_max(heatmap)
        if float(max_val) > 0:
            heatmap = heatmap / max_val

        return np.asarray(heatmap, dtype=np.float32)


def overlay_heatmap(
    display_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> Image.Image:
    """
    Blend a (H, W) heat-map over the original photo and return a Pillow image.

    Implemented with matplotlib + Pillow only — no OpenCV, so the project installs
    without the usual Windows DLL headaches.
    """
    from matplotlib import colormaps

    display_img = display_img.convert("RGB")
    width, height = display_img.size

    heat = np.asarray(heatmap, dtype=np.float32)
    heat = np.maximum(heat, 0.0)
    peak = float(heat.max())
    if peak > 0:
        heat = heat / peak

    heat_img = Image.fromarray(np.uint8(255 * heat), mode="L")
    heat_img = heat_img.resize((width, height), Image.BILINEAR)

    # 0..1 heat -> RGB through the colour map
    coloured = colormaps[colormap](np.asarray(heat_img, dtype=np.float32) / 255.0)[..., :3]
    coloured_img = Image.fromarray(np.uint8(255 * coloured), mode="RGB")

    return Image.blend(display_img, coloured_img, alpha=float(np.clip(alpha, 0.0, 1.0)))
