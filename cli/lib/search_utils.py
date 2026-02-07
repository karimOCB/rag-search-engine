import json, os

from dotenv import load_dotenv
from google import genai

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_path = os.path.join(ROOT_DIR, "data", "movies.json")
stop_words_path = os.path.join(ROOT_DIR, "data", "stopwords.txt")
CACHE_DIR = os.path.join(ROOT_DIR, "cache")

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
SCORE_PRECISION = 3

DEFAULT_CHUNK_LIMIT = 200
DEFAULT_CHUNK_OVERLAP = 2
DEFAULT_ALPHA_HYBRID = 0.5
RRF_K1 = 60

def load_movies():
    with open(data_path, 'r') as f:
        data = json.load(f)
    return data["movies"]

def load_stop_words():
    with open(stop_words_path, 'r') as f:
        stop_words = f.read()
        stop_words = stop_words.splitlines()
    return stop_words

def single_token(term):
    if isinstance(term, list):
        if len(term) != 1:
            raise ValueError("Term must be a single token")

def get_enhanced_query(query, choice):
    if choice == "spell":
        prompt = f"""Fix any spelling errors in this movie search query.

                Only correct obvious typos. Don't change correctly spelled words.

                Query: "{query}"

                If no errors, return the original query.
                Corrected:"""
    elif choice == "rewrite":
        prompt= f"""Rewrite this movie search query to be more specific and searchable.

                Original: "{query}"

                Consider:
                - Common movie knowledge (famous actors, popular films)
                - Genre conventions (horror = scary, animation = cartoon)
                - Keep it concise (under 10 words)
                - It should be a google style search query that's very specific
                - Don't use boolean logic

                Examples:

                - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                Rewritten query:"""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"Using key {api_key[:6]}...")
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.5-flash', contents=prompt
    )
    return response.text
    # print(response.text)
    # print(f"Prompt Tokens: {response.usage_metadata.prompt_token_count}")
    # print(f"Response Tokens: {response.usage_metadata.candidates_token_count}")