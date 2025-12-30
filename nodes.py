
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
    Flux Kontext Stretch Compensation Node.
    Expands canvas height to counteract 5.2% vertical stretch during sampling.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "comp_mode": (["Mirror", "Replicate", "Solid Color"], {"default": "Solid Color"}),
                "solid_color": ("STRING", {"default": "#FFFFFF"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "COMPENSATION_DATA", "INT", "INT")
    RETURN_NAMES = ("IMAGE", "comp_data", "width", "height")
    FUNCTION = "compensate"
    CATEGORY = "CCNotes/Process & Restore"

    def compensate(self, image, comp_mode, solid_color="#FFFFFF"):
        k_factor = 1.0521
        img = image.permute(0, 3, 1, 2)
        mode_map = {"Mirror": "reflect", "Replicate": "replicate", "Solid Color": "constant"}
        pt_pad_mode = mode_map.get(comp_mode, "reflect")
        
        old_h, old_w = img.shape[2], img.shape[3]
        
        new_h = int(round(old_h * k_factor / 16)) * 16
        pad_total_y = new_h - old_h
        pad_top = pad_total_y // 2
        pad_bottom = pad_total_y - pad_top
        
        new_w = int(round(old_w * k_factor / 16)) * 16
        pad_total_x = new_w - old_w
        pad_left = pad_total_x // 2
        pad_right = pad_total_x - pad_left
        
        try:
            if comp_mode == "Solid Color":
                r, g, b, a = parse_color(solid_color)
                b_sz, c, _, _ = img.shape
                canvas = torch.zeros((b_sz, c, new_h, new_w), dtype=img.dtype, device=img.device)
                c_data = [r/255.0, g/255.0, b/255.0, a/255.0]
                for i in range(min(c, 4)):
                    canvas[:, i, :, :] = c_data[i]
                canvas[:, :, pad_top:pad_top+old_h, pad_left:pad_left+old_w] = img
                img_out = canvas
            else:
                img_out = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode=pt_pad_mode)
        except: 
            img_out = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=1.0)
        
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
    Flux Kontext Stretch Restore Node.
    Restores image aspect ratio/composition using Kornia LoFTR or Template Matching alignment.
    """
    
    # LoFTR matcher singleton (lazy loaded)
    _loftr_matcher = None
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
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
        """
        Lazy load LoFTR matcher from ComfyUI models directory.
        Model will be stored in: ComfyUI/models/loftr/loftr_outdoor.ckpt
        """
        if cls._loftr_matcher is None:
            # Set up model path in ComfyUI models directory
            loftr_model_dir = os.path.join(folder_paths.models_dir, "loftr")
            os.makedirs(loftr_model_dir, exist_ok=True)
            
            model_path = os.path.join(loftr_model_dir, "loftr_outdoor.ckpt")
            
            if os.path.exists(model_path):
                # Load from local path
                cls._loftr_matcher = LoFTR(pretrained=model_path)
            else:
                # Download and save to local path
                import urllib.request
                url = "http://cmp.felk.cvut.cz/~mishkdmy/models/loftr_outdoor.ckpt"
                print(f"[CCNotes] Downloading LoFTR model to {model_path}...")
                urllib.request.urlretrieve(url, model_path)
                print(f"[CCNotes] LoFTR model downloaded successfully.")
                cls._loftr_matcher = LoFTR(pretrained=model_path)
        
        return cls._loftr_matcher.to(device).eval()

    def align_image_kornia(self, generated_img: torch.Tensor, reference_img: torch.Tensor, orig_h: int, orig_w: int):
        """Kornia LoFTR alignment."""
        device = generated_img.device
        ref_bchw = reference_img.permute(0, 3, 1, 2).to(device)
        
        gen_h, gen_w = generated_img.shape[2], generated_img.shape[3]
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
        
        kpts0_np = kpts0_filtered.cpu().numpy()
        kpts1_np = kpts1_filtered.cpu().numpy()
        
        H, inliers = cv2.findHomography(kpts0_np, kpts1_np, cv2.RANSAC, 5.0)
        
        if H is None or np.sum(inliers) < 4:
            return None
        
        H_tensor = torch.from_numpy(H).float().to(device).unsqueeze(0)
        aligned = warp_perspective(generated_img, H_tensor, (gen_h, gen_w), mode='bilinear', padding_mode='border')
        
        aligned_final = F.interpolate(aligned, size=(orig_h, orig_w), mode='bicubic', align_corners=False)
        return aligned_final

    def align_image_template(self, main_img_np, ref_img_np):
        """Legacy template matching alignment."""
        main_gray = cv2.cvtColor(main_img_np, cv2.COLOR_RGB2GRAY)
        ref_gray = cv2.cvtColor(ref_img_np, cv2.COLOR_RGB2GRAY)
        
        ref_h, ref_w = ref_gray.shape[:2]
        main_h, main_w = main_gray.shape[:2]
        
        best_score = -1
        best_scale = 1.0
        best_y = 0
        best_x = 0
        
        base_ratio = ref_h / main_h
        search_scales = np.linspace(base_ratio * 0.9, 1.1, 40)
        
        crop_h = int(ref_h * 0.5)
        crop_w = int(ref_w * 0.5)
        crop_y = (ref_h - crop_h) // 2
        crop_x = (ref_w - crop_w) // 2
        
        ref_template = ref_gray[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        
        for s in search_scales:
            target_h = int(main_h * s)
            if target_h < crop_h: continue
            
            resized_main = cv2.resize(main_gray, (main_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            res = cv2.matchTemplate(resized_main, ref_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            if max_val > best_score:
                best_score = max_val
                best_scale = s
                found_x, found_y = max_loc
                best_y = found_y - crop_y
                best_x = found_x - crop_x
                
        return best_scale, best_y, best_x

    def restore(self, image, comp_data, reference_image):
        img = image.permute(0, 3, 1, 2)
        orig_h, orig_w = comp_data["orig_h"], comp_data["orig_w"]
        
        try:
            aligned = self.align_image_kornia(img, reference_image, orig_h, orig_w)
            if aligned is not None:
                return (aligned.permute(0, 2, 3, 1),)
        except Exception:
            pass
        
        final_scale_y = orig_h / img.shape[2] 
        final_scale_x = orig_w / img.shape[3]
        final_offset_y = 0
        final_offset_x = 0
        
        try:
            ref_np = (reference_image[0].cpu().numpy() * 255).astype(np.uint8)
            main_np = (img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            
            scale_found, off_y, off_x = self.align_image_template(main_np, ref_np)
            
            final_scale_y = scale_found
            final_scale_x = 1.0
            final_offset_y = off_y
            final_offset_x = off_x
        except Exception:
            pass

        target_h = int(img.shape[2] * final_scale_y)
        target_w = int(img.shape[3] * final_scale_x)
        if target_h < 1: target_h = 1
        if target_w < 1: target_w = 1
        
        img_scaled = F.interpolate(img, size=(target_h, target_w), mode='bicubic', align_corners=False)
        
        y_start = int(final_offset_y)
        x_start = int(final_offset_x)
        y_end = y_start + orig_h
        x_end = x_start + orig_w
        
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
