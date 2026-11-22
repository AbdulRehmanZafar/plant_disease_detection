"""
src package for the Plant Disease Detection project.

Sub-modules
-----------
config        : paths, constants, the 38 class names (canonical order)
preprocessing : any uploaded image -> (1, 224, 224, 3) tensor   [NumPy only]
npz_utils     : reading resnet_weights.npz in the right order    [NumPy only]
model         : architecture + weight loading                   [needs TensorFlow]
gradcam       : class activation heat-maps (explainability)     [needs TensorFlow]
advice        : offline treatment knowledge base + optional LLM [pure Python]

Keeping preprocessing/npz_utils/advice free of TensorFlow means they can be
unit-tested in seconds, without installing a 600 MB framework.
"""
