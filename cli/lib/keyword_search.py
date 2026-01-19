import string
from nltk.stem import PorterStemmer
from .search_utils import load_movies, load_stop_words, DEFAULT_SEARCH_LIMIT

def keyword_search(query, inv_idx):
    inv_idx.load()
    result = []
    query_tokens = tokenize_text(query)
    seen_ids = set()

    for q_token in query_tokens:
        if len(result) >= DEFAULT_SEARCH_LIMIT:
                break
        if q_token not in inv_idx.index:
            continue
        ids_set = inv_idx.get_documents(q_token)
        for doc_id in ids_set:
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            movie = inv_idx.docmap[doc_id]
            result.append((movie["id"], movie["title"]))
            if len(result) >= DEFAULT_SEARCH_LIMIT:
                break
    return result

def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()

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
    stemmer = PorterStemmer() 
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = [token for token in tokens if token != ""]
    filtered_tokens = remove_stop_words(valid_tokens)
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    return stemmed_tokens

