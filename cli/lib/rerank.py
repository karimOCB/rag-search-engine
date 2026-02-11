import os, time

from dotenv import load_dotenv
from google import genai

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
        new_results[i]["llm_score"] = int(corrected)
    
    return sorted(new_results, key=lambda item: item["llm_score"], reverse=True)   

def rerank_result(results, query, method):
    match method:
        case "individual":
            return individual_rerank(results, query)
        case _:
            return query