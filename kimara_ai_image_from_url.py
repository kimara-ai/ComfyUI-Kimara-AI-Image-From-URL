import requests
from PIL import Image, ImageOps
import torch
from io import BytesIO
import numpy as np
from urllib.parse import unquote, urlparse

ALLOWED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')
# Limit image size to prevent excessively large images from causing issues
# 10 * 1024 * 1024 = 10 Mb
MAX_IMAGE_SIZE = 10 * 1024 * 1024

class KimaraAIImageFromURL:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"multiline": False, "default": "", "lazy": False}),
                "megapixels": ("FLOAT", {"default": 1, "min": 0, "max": 20, "step": 0.1})         
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "execute"
    CATEGORY = "Kimara.ai"

    def execute(self, url, megapixels):
        # Decode and validate received URL
        self.validate_url(url)

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            if self.is_valid_image(response.content):
                img = Image.open(BytesIO(response.content))

            image_tensor, mask_tensor = self.process_image(img, megapixels)      

        except (requests.RequestException, IOError) as e:
            raise ValueError(f"Error loading image from '{url}': {e}")

        return image_tensor, mask_tensor

    def validate_url(self, url):
        parsed_url = urlparse(url)

        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError(f"Invalid URL: {url}. The URL must have a valid scheme (http/https) and domain.")

        if not url.lower().endswith(ALLOWED_EXTENSIONS):
            raise ValueError(f"Invalid image file extension in URL: {url}. Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed.")

        # Unquote after validation
        url = unquote(url)
        # Revalidate the URL after unquoting to catch any unusual cases
        parsed_url = urlparse(url)

        if not parsed_url.path or not parsed_url.netloc:
            raise ValueError(f"Invalid URL. The URL is improperly formed after unquoting.")

        return url

    def is_valid_image(self, content):
        # Verify that the response content is a valid image
        if len(content) > MAX_IMAGE_SIZE:
            raise ValueError(f"Image too large. Maximum size is {MAX_IMAGE_SIZE // 1024 // 1024} MB.")

        try:
            with Image.open(BytesIO(content)) as img:
                img.verify()
            return True

        except (IOError, SyntaxError):
            raise ValueError(f"URL does not point to a valid image.")

    def process_image(self, img, megapixels):
        
        # Ensure correct orientation using EXIF data
        img = ImageOps.exif_transpose(img)

        # Use megapixels 0 as a no downscaling toggle
        if megapixels != 0:
            img = self.downscale_to_megapixels(img, megapixels)

        # Normalize 16-bit images (mode 'I') to [0, 1] range
        if img.mode == 'I':
            img = img.point(lambda i: i * (1 / 255))

        # Process alpha channel if present
        if 'A' in img.getbands():
            mask = np.array(img.getchannel('A'), dtype=np.float32) / 255.0
            # Invert mask values and convert to Tensor. Remove the '1.0 -' or make toggleable if causes mask handling issues
            mask_tensor = 1.0 - torch.from_numpy(mask).unsqueeze(0)
        else:
            # Default mask: zeros with same height and width
            mask_tensor = torch.zeros((1, img.size[1], img.size[0]), dtype=torch.float32)

        # Convert to RBG since mask is handled separately
        img = img.convert("RGB")
        # Convert from PIL to Tensor
        image_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

        return image_tensor, mask_tensor

    def downscale_to_megapixels(self, img, target_megapixels):

        width, height = img.size

        current_megapixels = (width * height) / 1000000

        if current_megapixels <= target_megapixels:
            return img

        # Calculate the scaling factor
        scaling_factor = (target_megapixels / current_megapixels) ** 0.5
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)

        # Resize the image while maintaining quality
        return img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    

NODE_CLASS_MAPPINGS = {
    "KimaraAIImageFromURL": KimaraAIImageFromURL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KimaraAIImageFromURL": "Kimara.ai Image From URL"
}