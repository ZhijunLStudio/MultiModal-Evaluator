import base64
from io import BytesIO
from PIL import Image

class ImageProcessor:
    @staticmethod
    def encode_image(image_path: str) -> str:
        """将图像编码为Base64字符串"""
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format="PNG")
                base64_data = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                return f"data:image/png;base64,{base64_data}"
        except Exception as e:
            raise Exception(f"图像处理错误: {str(e)}")
