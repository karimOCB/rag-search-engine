from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name, model_kwargs={"use_fast": True})

    def embed_image(self, path):
        image = Image.open(path)
        image_embedding = self.model.encode([image])
        return image_embedding[0]

def verify_image_embedding_command(img_path):
    multimodal_search = MultimodalSearch()
    embedded_img = multimodal_search.embed_image(img_path)
    print(f"Embedding shape: {embedded_img.shape[0]} dimensions")