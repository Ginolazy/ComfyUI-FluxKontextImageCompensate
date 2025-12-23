from .nodes import FluxKontextImageCompensate, FluxKontextImageRestore

NODE_CLASS_MAPPINGS = {
    "FluxKontextImageCompensate": FluxKontextImageCompensate,
    "FluxKontextImageRestore": FluxKontextImageRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKontextImageCompensate": "FluxKontextImageCompensate",
    "FluxKontextImageRestore": "FluxKontextImageRestore",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
