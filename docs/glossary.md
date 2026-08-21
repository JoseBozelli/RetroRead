# Glossary

> This is a living document. Terms are added or revised as the project progresses. If a term used elsewhere in the docs isn't defined here, that's a gap to fix, not something the reader should have to infer.

---

**Abstention**
The system's ability to decline to give a reading rather than return a low-quality one. Triggered when confidence falls below a chosen threshold. Central to RetroRead's design because a wrong high-confidence reading is treated as more dangerous than an honest "I don't know."

**Classical Computer Vision (CV) Baseline (Hough Transform)**
The non-learned comparison point for the deep learning model. "Classical CV" is the general category — computer vision done with hand-designed rules and geometry rather than a trained neural network. In RetroRead specifically, the classical baseline is built using the **Hough Transform** (see below): Hough Circle Transform to find the gauge face, edge detection plus line-fitting to find the needle, then geometry to convert the needle angle into a reading. First mention in any document should use the full name "Classical CV Baseline (Hough Transform)"; later mentions in the same document can shorten to "the classical baseline."

**Confidence threshold**
A cutoff value: readings above it are accepted, readings below it are abstained on. Raising the threshold generally reduces bad readings but increases the abstention rate — this tradeoff is evaluated directly (see coverage).

**Coverage**
The percentage of input images for which the system produces a reading at all (i.e., does not abstain). Reported alongside accuracy at each confidence threshold, since coverage and accuracy trade off against each other.

**Domain transfer / domain shift**
The gap between the conditions a model was trained on and the conditions it's actually used in. For RetroRead, this specifically means: trained mostly on **synthetic** (computer-rendered) gauge images, evaluated on **real photographs**. Performance dropping when moving from synthetic to real images is domain shift, and how much it drops is one of the project's three core hypotheses (H3).

**Ground truth**
The true, correct value for something the model is trying to predict — here, the actual gauge reading a human would record, used to check whether the model's output is right or wrong.

**Hough Transform**
A classical (non-learned) computer vision algorithm for detecting simple shapes — circles or lines — in an image, even when the image is noisy or the shape is partially obscured. It works by having candidate edge pixels "vote" for shapes they could belong to; the shape with the most votes wins. Used in RetroRead's classical baseline: Hough Circle Transform locates the gauge face, and an edge/line-based variant locates the needle.

**Keypoint detection**
A deep learning task where the model predicts the pixel locations of specific, meaningful points in an image — rather than classifying the whole image or drawing a bounding box. RetroRead's deep learning model predicts four keypoints: the scale-minimum tick, scale-maximum tick, pointer center, and pointer tip. The numeric reading is then computed deterministically from those four points using geometry, not predicted directly by the network.

**MAE (Mean Absolute Error)**
The average size of the model's prediction errors, ignoring whether each error was too high or too low. Reported in gauge units (e.g., PSI) to describe typical reading accuracy.

**Pretrained backbone**
A neural network (or part of one) that has already been trained on a large, general image dataset before being adapted to this project's specific task. Reusing a pretrained backbone (rather than training from scratch) is called **transfer learning**, and is generally faster and needs less task-specific data — important given the CPU-only training constraint on this project.

**Product Hypothesis**
The specific business/user claim this portfolio project is testing — see `docs/PRODUCT_HYPOTHESIS.md`. Distinguished deliberately from a "project hypothesis": this document evaluates whether RetroRead would be useful and viable as a product, not just whether the technical approach works.

**Robustness matrix**
A table showing how model performance degrades under controlled image perturbations (brightness, contrast, blur, rotation), used to compare the classical baseline and the deep learning model side by side (H2).

**Transfer learning**
See *pretrained backbone*.
