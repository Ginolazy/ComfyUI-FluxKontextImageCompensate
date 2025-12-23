from .nodes import FluxKontextImageCompensate, FluxKontextImageRestore

NODE_CLASS_MAPPINGS = {
    "FluxKontextImageCompensate": FluxKontextImageCompensate,
    "FluxKontextImageRestore": FluxKontextImageRestore,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxKontextImageCompensate": "Flux Kontext Compensate",
    "FluxKontextImageRestore": "Flux Kontext Restore",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
