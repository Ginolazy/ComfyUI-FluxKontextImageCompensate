# ComfyUI-FluxKontextImageCompensation

A focused ComfyUI plugin for restoring spatial consistency when working with the Flux Kontext model during image expansion and outpainting.

When generating on an expanded canvas, Flux Kontext may redistribute image content in a way that leads to subtle but consistent spatial distortion. This project provides a practical, reversible solution by introducing a **pre-compensation and post-restoration workflow** that preserves the original composition and alignment.

The approach is content-aware and resolution-independent, avoiding blind pixel resizing or fixed-ratio corrections.

---

## Nodes

### 1. Flux Kontext Image Compensate

**Function**  
Prepares an image for Flux Kontext generation by expanding the canvas to accommodate content redistribution without distorting the original composition.

**Usage**  
Connect this node *before* passing the image into a Flux Kontext / KSampler workflow.

**Outputs**
- `IMAGE`  
  The expanded image used for generation.
- `data`  
  Compensation metadata required for accurate restoration.

---

### 2. Flux Kontext Image Restore

**Function**  
Restores the generated image back to its original spatial composition using the compensation metadata.

**Key Features**
- **Content-Aligned Restoration**  
  Restores spatial consistency without relying on fixed pixel ratios.
- **Auto Alignment (Optional)**  
  When a `reference_image` is provided, the node performs automatic X/Y alignment to precisely match the original composition, even if minor pixel shifts occurred during generation.
- **Clean Cropping**  
  Excess regions introduced during compensation are removed to recover the original framing.

---

## Installation

Clone this repository into your `ComfyUI/custom_nodes/` directory:

```bash
git clone https://github.com/Ginolazy/ComfyUI-FluxKontextImageCompensation.git
