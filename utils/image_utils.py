from io import BytesIO
from PIL import Image

def load_image(uploaded_file):
    return Image.open(BytesIO(uploaded_file.getvalue())).convert('RGB')
