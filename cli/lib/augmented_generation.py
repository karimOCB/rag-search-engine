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

    titles, formatted_ranking = formatter_for_llm(results)

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Documents:
            {formatted_ranking}

            Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    
    return (titles, corrected)


def summarize_command(query, limit):
    response = rrf_search_command(query, limit = limit)
    results = response["results"]

    titles, formatted_ranking = formatter_for_llm(results)

    prompt = f"""
            Provide information useful to this query by synthesizing information from multiple search results in detail.
            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
            This should be tailored to Hoopla users. Hoopla is a movie streaming service.
            Query: {query}
            Search Results:
            {formatted_ranking}
            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
            """

    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return (titles, corrected)


def citations_command(query, limit):
    response = rrf_search_command(query, limit = limit)
    results = response["results"]

    titles, formatted_ranking = formatter_for_llm(results)

    prompt = f"""Answer the question or provide information based on the provided documents.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

            Query: {query}

            Documents:
            {formatted_ranking}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources using [1], [2], etc. format when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""

    response = client.models.generate_content(model=model, contents=prompt)
    corrected = (response.text or "").strip().strip('"')
    return (titles, corrected)


def formatter_for_llm(results):
    formatted_ranking = []
    titles = []
    for i, ranking in enumerate(results):
        formatted_ranking.append(f'{i}. Title: {ranking['document']['title']}  Description": {ranking['document']['description']}')
        titles.append(f'  - Title: {ranking['document']['title']}')

    return (titles, formatted_ranking)