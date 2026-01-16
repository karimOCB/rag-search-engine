import string
from .search_utils import load_movies, load_stop_words, DEFAULT_SEARCH_LIMIT

def keyword_search(query):
    movies = load_movies()
    result = []
    query_tokens = tokenize_text(query) 

    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            result.append(movie)
            if len(result) >= DEFAULT_SEARCH_LIMIT:
                break
    return result

def has_matching_token(query_tokens, title_tokens):
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def remove_stop_words(tokens):
    stop_words = load_stop_words()
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def tokenize_text(text): 
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = [token for token in tokens if token != ""]
    good_tokens = remove_stop_words(valid_tokens)
    return good_tokens