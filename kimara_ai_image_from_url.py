import urllib.request
import urllib.error
from PIL import Image, ImageOps
import torch
from io import BytesIO
import numpy as np
from urllib.parse import unquote, urlparse

ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.avif', '.webp', '.heic', '.heif')
MAX_IMAGE_SIZE = 10 * 1024 * 1024

class KimaraAIImageFromURL:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"multiline": False, "default": "", "lazy": False}),
                "megapixels": ("FLOAT", {"default": 1, "min": 0, "max": 20, "step": 0.1}),
                "user_agent": ("STRING", {"multiline": False, "default": "ComfyUI Image Downloader/1.0", "lazy": False})
            }
        }
        

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "Kimara.ai"

    def execute(self, url, megapixels, user_agent):
        self.validate_url(url)
        try:
            headers = {"User-Agent": user_agent}
            http_request = urllib.request.Request(url, headers = headers)
            with urllib.request.urlopen(http_request, timeout=10) as response:
                content = response.read()
            if self.is_valid_image(content):
                img = Image.open(BytesIO(content))
            image_tensor, mask_tensor = self.process_image(img, megapixels)
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise ValueError("403 Forbidden:{url} has blocked this request. Try changing the User_Agent.")
            elif e.code == 404:
                raise ValueError("404 Not found: The images url is incorrect or no longer available.")
            else:
                raise ValueError(f"Error loading image from '{url}': {e}")
        except (urllib.error.URLError, IOError) as e:
            raise ValueError(f"Error loading image from '{url}': {e}")
        return image_tensor, mask_tensor

    def validate_url(self, url):
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL: {url}. The URL must have a valid scheme (http/https) and domain.")
        if not url.lower().endswith(ALLOWED_EXTENSIONS):
            raise ValueError(f"Invalid image file extension in URL: {url}. Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed.")
        url = unquote(url)
        parsed_url = urlparse(url)
        if not parsed_url.path or not parsed_url.netloc:
            raise ValueError(f"Invalid URL. The URL is improperly formed after unquoting.")
        return url

    def is_valid_image(self, content):
        if len(content) > MAX_IMAGE_SIZE:
            raise ValueError(f"Image too large. Maximum size is {MAX_IMAGE_SIZE // 1024 // 1024} MB.")
        try:
            with Image.open(BytesIO(content)) as img:
                img.verify()
            return True
        except (IOError, SyntaxError):
            raise ValueError(f"URL does not point to a valid image.")

    def process_image(self, img, megapixels):
        img = ImageOps.exif_transpose(img)
        if megapixels != 0:
            img = self.downscale_to_megapixels(img, megapixels)
        # Normalize 16-bit images (mode 'I') to [0, 1] range
        if img.mode == 'I':
            img = img.point(lambda i: i * (1 / 255))
        # Process alpha channel if present
        if 'A' in img.getbands():
            mask = np.array(img.getchannel('A'), dtype=np.float32) / 255.0
            mask_tensor = 1.0 - torch.from_numpy(mask).unsqueeze(0)
        else:
            mask_tensor = torch.zeros((1, img.size[1], img.size[0]), dtype=torch.float32)
        # Convert to RGB since mask is handled separately
        img = img.convert("RGB")
        image_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
        return image_tensor, mask_tensor

    def downscale_to_megapixels(self, img, target_megapixels):
        width, height = img.size
        current_megapixels = (width * height) / 1000000
        if current_megapixels <= target_megapixels:
            return img
        scaling_factor = (target_megapixels / current_megapixels) ** 0.5
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

NODE_CLASS_MAPPINGS = {
    "KimaraAIImageFromURL": KimaraAIImageFromURL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KimaraAIImageFromURL": "Kimara.ai Image From URL"
}
