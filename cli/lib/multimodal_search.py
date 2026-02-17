import numpy as np
import os
from PIL import Image
from sentence_transformers import SentenceTransformer
from .search_utils import cosine_similarity, load_movies, CACHE_DIR

class MultimodalSearch():
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name, model_kwargs={"use_fast": True})
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.text_embeddings = []
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings2.npy")

    def embed_image(self, img_path):    
        image = Image.open(img_path)
        image_embedding = self.model.encode([image])
        return image_embedding[0]

    def build_embeddings(self):
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar = True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(self.embeddings_path, self.text_embeddings)
        return self.text_embeddings

    def load_or_create_embeddings(self):
        if os.path.exists(self.embeddings_path):
            self.text_embeddings = np.load(self.embeddings_path)
            if len(self.text_embeddings) == len(self.documents):
                return self.text_embeddings
         
        return self.build_embeddings()

    def search_with_image(self, img_path):
        image_embedding = self.embed_image(img_path)
        cosine_similarities = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity_score = cosine_similarity(text_embedding, image_embedding)
            cosine_similarities.append((similarity_score, self.documents[i]))
        sorted_cosine_similarities = sorted(cosine_similarities, key=lambda x: x[0], reverse=True)
        results = []
        for _, result in enumerate(sorted_cosine_similarities[:5]):
            results.append({
                "similarity_score": result[0],
                "id": result[1]["id"],
                "title": result[1]["title"],
                "description": result[1]["description"],
            })
        return results

def verify_image_embedding_command(img_path):
    multimodal_search = MultimodalSearch()
    embedded_img = multimodal_search.embed_image(img_path)
    print(f"Embedding shape: {embedded_img.shape[0]} dimensions")

def image_search_command(img_path):
    movies = load_movies()
    multimodal_search = MultimodalSearch(movies)
    multimodal_search.load_or_create_embeddings()
    return multimodal_search.search_with_image(img_path)
    