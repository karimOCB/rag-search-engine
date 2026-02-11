import os, time, json

from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY_2")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash-lite"

def individual_rerank(results, query):
    new_results = results
    for i, result in enumerate(results):
        if i > 0:
            time.sleep(60)
        prompt = f"""Rate how well this movie matches the search query.
            
                Query: "{query}"
                Movie: {result.get("document", {}).get("title", "")} - {result.get("document", "")}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Give me ONLY the number in your response, no other text or explanation.

                Score:"""

        response = client.models.generate_content(model=model, contents=prompt)
        corrected = (response.text or "").strip().strip('"')
        new_results[i]["llm_rank"] = int(corrected)
    
    return sorted(new_results, key=lambda item: item["llm_rank"], reverse=True)   


def batch_rerank(results, query):
    documents = [result["document"] for result in results]
    documents_map = {}
    for result in results:
        documents_map[result["document"]["id"]] = result
    prompt = f"""Rank these movies by relevance to the search query.

            Query: "{query}"

            Movies: {documents}

            Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

            [75, 12, 34, 2, 1]
            """
    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    data = json.loads(corrected)
    print(f"\n\n {data}")
    reranked_results = []
    for i, id in enumerate(data):
        documents_map[id]["llm_rank"] = i + 1
        reranked_results.append(documents_map[id])
    
    return reranked_results


def cross_encoder(results, query):
    pairs = []
    cross_encode = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
    for result in results:
        pairs.append([query, f"{result.get('document', '')} - {result["document"].get('title', '')}"])
    scores = cross_encode.predict(pairs)
    for i, score in enumerate(scores):
        results[i]["llm_rank"] = score
    sorted_results = sorted(results, key=lambda item: item["llm_rank"], reverse=True)
    return sorted_results

def rerank_result(results, query, method):
    match method:
        case "individual":
            return individual_rerank(results, query)
        case "batch":
            return batch_rerank(results, query)
        case "cross_encoder":
            return cross_encoder(results, query)
        case _:
            return query