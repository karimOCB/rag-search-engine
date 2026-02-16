import os

from dotenv import load_dotenv
from google import genai

from .hybrid_search import rrf_search_command

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"

def rag_command(query):
    response = rrf_search_command(query, limit=5)
    results = response["results"]

    formatted_ranking = []
    titles = []
    for i, ranking in enumerate(results):
        formatted_ranking.append(f'{i}. Title: {ranking['document']['title']}  Description": {ranking['document']['description']}')
        titles.append(f'  - Title: {ranking['document']['title']}')
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Documents:
            {formatted_ranking}

            Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    
    return (titles, corrected)
