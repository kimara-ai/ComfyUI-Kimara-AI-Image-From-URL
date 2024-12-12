"""
ComfyUI Custom Nodes for Kimara.ai
"""

try:
    from .kimara_ai_image_from_url import KimaraAIImageFromURL
except ImportError as e:
    print(f"Error importing nodes: {e}")

# Maps internal node names to their class implementations
NODE_CLASS_MAPPINGS = {
    "KimaraAIImageFromURL": KimaraAIImageFromURL
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KimaraAIImageFromURL": "Kimara.ai Image From URL"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
__version__ = "1.0.0"
