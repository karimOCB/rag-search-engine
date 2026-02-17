import argparse, mimetypes, os

from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("gemini_api_key")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--image", type=str, help="The path to the image file")
    parser.add_argument("--query", type=str, help="The query to rewrite based on the image")

    args = parser.parse_args()

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, mode="rb") as f:
        img = f.read()

    system_prompt = f"""
            Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
            - Synthesize visual and textual information
            - Focus on movie-specific details (actors, scenes, style, etc.)
            - Return only the rewritten query, without any additional commentary
            """

    parts = [
        system_prompt,
        genai.types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    response = client.models.generate_content(model=model, contents=parts)
    corrected = (response.text or "").strip().strip('"')
    
    print(f"Rewritten query: {corrected}")
    if response.usage_metadata is not None:
        print(f"Total tokens: {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()