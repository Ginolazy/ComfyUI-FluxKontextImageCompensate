
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import re
import os
import folder_paths
import kornia
from kornia.feature import LoFTR
from kornia.geometry.transform import warp_perspective 
from PIL import ImageColor
from typing import Tuple

def parse_color(color_str: str) -> Tuple[int, int, int, int]:
    try:
        s = str(color_str).strip().lower()
        if s.startswith("#"): # Hex colors
            hex_str = s[1:]
            if len(hex_str) in (3, 4):
                hex_str = "".join(c*2 for c in hex_str)
            comps = [int(hex_str[i:i+2], 16) for i in range(0, len(hex_str), 2)]
            return tuple(comps + [255]) if len(comps) == 3 else tuple(comps)
        match = re.findall(r"[\d.]+%?", s)
        if match:
            comps = []
            for p in match:
                if "%" in p:
                    comps.append(int(float(p.strip("%")) * 2.55))
                else:
                    comps.append(int(float(p) * (255 if "." in p else 1)))
            return tuple(comps + [255]) if len(comps) == 3 else tuple(comps)
        rgb = ImageColor.getrgb(s)
        return (*rgb, 255)
    except Exception as e:
        raise ValueError(f"[parse_color] Invalid color '{color_str}': {e}")

class FluxKontextImageCompensate:
    """
    Flux Kontext Stretch Compensation Node
    The Kontext model introduces vertical stretching during sampling.
    This node expands the canvas height (Padding) in the Y direction (and optionally X), allowing AI to generate on a larger canvas.
    Combined with the Restore node later to squeeze it back to the original size, counteracting the stretch and maintaining correct aspect ratio.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "comp_mode": (["Mirror", "Replicate", "Solid Color"], {"default": "Mirror"}),
            },
            "optional": {
                "solid_color": ("STRING", {"default": "#FFFFFF"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "COMPENSATION_DATA", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "comp_data", "width", "height")
    FUNCTION = "compensate"
    CATEGORY = "CCNotes/Process & Restore"

    def compensate(self, image, comp_mode, solid_color="#FFFFFF"):
        k_factor = 1.0521
        # image shape: [B, H, W, C]
        img = image.permute(0, 3, 1, 2)  # [B, C, H, W]
        # Map nice names to internal pyTorch modes
        mode_map = {
            "Mirror": "reflect",
            "Replicate": "replicate", 
            "Solid Color": "constant"
        }
        pt_pad_mode = mode_map.get(comp_mode, "reflect")
        
        old_h, old_w = img.shape[2], img.shape[3]
        
        # Expand canvas Y
        new_h = int(round(old_h * k_factor / 16)) * 16
        pad_total_y = new_h - old_h
        pad_top = pad_total_y // 2
        pad_bottom = pad_total_y - pad_top
        
        # Expand canvas X
        new_w = int(round(old_w * k_factor / 16)) * 16
        pad_total_x = new_w - old_w
        pad_left = pad_total_x // 2
        pad_right = pad_total_x - pad_left
        
        # Pad (left, right, top, bottom)
        try:
            if comp_mode == "Solid Color":
                # Manual padding for constant color
                r, g, b, a = parse_color(solid_color) # returns ints 0-255
                b_sz, c, _, _ = img.shape
                
                # Create canvas
                canvas = torch.zeros((b_sz, c, new_h, new_w), dtype=img.dtype, device=img.device)
                
                # Fill color
                c_data = [r/255.0, g/255.0, b/255.0, a/255.0]
                for i in range(min(c, 4)):
                    canvas[:, i, :, :] = c_data[i]
                
                # Paste original image
                # pad_left, pad_top are starting indices
                canvas[:, :, pad_top:pad_top+old_h, pad_left:pad_left+old_w] = img
                img_out = canvas
            else:
                img_out = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode=pt_pad_mode)
        except: 
            img_out = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=1.0)
        
        actual_k_h = new_h / old_h if old_h > 0 else 1.0
        actual_k_w = new_w / old_w if old_w > 0 else 1.0
        
        data = {
            "orig_h": old_h, 
            "orig_w": old_w, 
            "new_h": new_h,
            "new_w": new_w,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
        }

        
        output_image = img_out.permute(0, 2, 3, 1)
        return (output_image, data, new_w, new_h)

class FluxKontextImageRestore:
    """
    Restores image to original aspect ratio using Kornia LoFTR feature matching.
    Effective for all image types including low-texture backgrounds.
    """
    _loftr_matcher = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"image": ("IMAGE",)},
            "optional": {
                "reference_image": ("IMAGE",), 
                "comp_data": ("COMPENSATION_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "CCNotes/Process & Restore"

    @classmethod
    def get_loftr_matcher(cls, device):
        """Load LoFTR matcher from ComfyUI/models/loftr/loftr_outdoor.ckpt"""
        if cls._loftr_matcher is None:
            loftr_model_dir = os.path.join(folder_paths.models_dir, "loftr")
            os.makedirs(loftr_model_dir, exist_ok=True)
            model_path = os.path.join(loftr_model_dir, "loftr_outdoor.ckpt")
            
            if not os.path.exists(model_path):
                import urllib.request
                url = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"
                print(f"[CCNotes] Downloading LoFTR model to {model_path}...")
                urllib.request.urlretrieve(url, model_path)
                print(f"[CCNotes] LoFTR model downloaded.")
            cls._loftr_matcher = LoFTR(pretrained=model_path)
        return cls._loftr_matcher.to(device).eval()

    def align_image_kornia(self, generated_img: torch.Tensor, reference_img: torch.Tensor, orig_h: int, orig_w: int):
        """Align generated image to reference using LoFTR feature matching + homography."""
        device = generated_img.device
        gen_h, gen_w = generated_img.shape[2], generated_img.shape[3]
        
        ref_bchw = reference_img.permute(0, 3, 1, 2).to(device)
        ref_resized = F.interpolate(ref_bchw, size=(gen_h, gen_w), mode='bilinear', align_corners=False)
        
        gen_gray = kornia.color.rgb_to_grayscale(generated_img)
        ref_gray = kornia.color.rgb_to_grayscale(ref_resized)
        
        matcher = self.get_loftr_matcher(device)
        with torch.no_grad():
            correspondences = matcher({"image0": gen_gray, "image1": ref_gray})
        
        kpts0 = correspondences['keypoints0']
        kpts1 = correspondences['keypoints1']
        confidence = correspondences['confidence']
        
        mask = confidence > 0.5
        kpts0_filtered = kpts0[mask]
        kpts1_filtered = kpts1[mask]
        
        if len(kpts0_filtered) < 4:
            return None
        
        H, inliers = cv2.findHomography(
            kpts0_filtered.cpu().numpy(), kpts1_filtered.cpu().numpy(), cv2.RANSAC, 5.0
        )
        
        if H is None or np.sum(inliers) < 4:
            return None
        
        H_tensor = torch.from_numpy(H).float().to(device).unsqueeze(0)
        aligned = warp_perspective(generated_img, H_tensor, (gen_h, gen_w), mode='bilinear', padding_mode='border')
        return F.interpolate(aligned, size=(orig_h, orig_w), mode='bicubic', align_corners=False)

    def restore(self, image, comp_data, reference_image=None):
        img = image.permute(0, 3, 1, 2)
        orig_h, orig_w = comp_data["orig_h"], comp_data["orig_w"]
        
        # Try Kornia alignment
        if reference_image is not None:
            try:
                aligned = self.align_image_kornia(img, reference_image, orig_h, orig_w)
                if aligned is not None:
                    return (aligned.permute(0, 2, 3, 1),)
            except Exception:
                pass
        
        # Math fallback
        new_h = comp_data.get("new_h", img.shape[2])
        pad_total_y = comp_data.get("pad_top", 0) + comp_data.get("pad_bottom", 0)
        
        if new_h < 1: new_h = 1 
        squeeze_s_y = orig_h / new_h
        crop_h_squeezed = orig_h - (pad_total_y * squeeze_s_y)
        
        if crop_h_squeezed > 0:
            zoom_f_y = orig_h / crop_h_squeezed
            final_scale_y = squeeze_s_y * zoom_f_y 
            final_offset_y = comp_data.get("pad_top", 0) * final_scale_y
        else:
            final_scale_y = orig_h / img.shape[2]
            final_offset_y = 0

        if img.shape[3] < orig_w:
            final_scale_x = orig_w / img.shape[3]
            final_offset_x = 0
        else:
            final_scale_x = 1.0
            final_offset_x = comp_data.get("pad_left", (img.shape[3] - orig_w) // 2)

        target_h = max(1, int(img.shape[2] * final_scale_y))
        target_w = max(1, int(img.shape[3] * final_scale_x))
        img_scaled = F.interpolate(img, size=(target_h, target_w), mode='bicubic', align_corners=False)
        
        y_start, x_start = int(final_offset_y), int(final_offset_x)
        y_end, x_end = y_start + orig_h, x_start + orig_w
        
        pad_l, pad_r, pad_t, pad_b = 0, 0, 0, 0
        if y_start < 0:
            pad_t = -y_start
            y_start = 0
            y_end += pad_t
        if y_end > img_scaled.shape[2]:
            pad_b = y_end - img_scaled.shape[2]
        if x_start < 0:
            pad_l = -x_start
            x_start = 0
            x_end += pad_l
        if x_end > img_scaled.shape[3]:
            pad_r = x_end - img_scaled.shape[3]
            
        if any([pad_l, pad_r, pad_t, pad_b]):
            img_scaled = F.pad(img_scaled, (pad_l, pad_r, pad_t, pad_b), mode='replicate')
            
        img_out = img_scaled[:, :, y_start:y_end, x_start:x_end]
        
        if img_out.shape[2] != orig_h or img_out.shape[3] != orig_w:
            img_out = F.interpolate(img_out, size=(orig_h, orig_w), mode='bicubic')
        
        return (img_out.permute(0, 2, 3, 1),)
