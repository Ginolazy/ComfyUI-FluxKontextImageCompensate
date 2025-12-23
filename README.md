# ComfyUI-FluxKontextImageCompensate

A focused ComfyUI plugin to handle the **vertical stretching** issue introduced by the Flux Kontext model.

## Nodes

### 1. Flux Kontext Image Compensate
- **Function**: Expands the canvas height (and optionally width) to compensate for the approximate **5.2% vertical stretch** caused by the Kontext model during sampling.
- **Usage**: Connect your source image to this node *before* passing it to the KSampler/Kontext workflow.
- **Outputs**: 
    - `IMAGE`: The padded/expanded image ready for generation.
    - `data`: Compensation metadata required for restoration.

### 2. Flux Kontext Image Restore
- **Function**: Restores the processed image to its original aspect ratio and composition by squeezing and cropping based on the compensation data.
- **Key Feature**:
    - **Auto-Alignment**: If you provide the original `reference_image`, this node performs pixel-perfect X/Y auto-alignment to ensure the generated content matches the original composition exactly, even if some pixel shifting occurred during generation.

## Installation
Clone this repository into your `ComfyUI/custom_nodes/` directory.
