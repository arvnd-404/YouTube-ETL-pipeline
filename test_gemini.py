import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 1. Load variables from .env
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please check your configuration.")

# 2. Initialize Gemini Client
client = genai.Client(api_key=gemini_api_key)

print(" Testing Gemini API connection...")

# 3. Sample batch of video titles
sample_videos = [
    {"id": "v1", "title": "I Built a $10,000 Secret Gaming Room Underground!"},
    {"id": "v2", "title": "RBI Announces New Monetary Policy Rules for 2026"},
    {"id": "v3", "title": "The Brutal Truth About Becoming a Software Engineer"}
]

prompt = f"""
Analyze the following YouTube video titles. For each video, provide:
1. 'ai_category': A concise, granular category (e.g., Gaming, Finance, Tech & Career, Entertainment).
2. 'sentiment': Positive, Neutral, or Negative.
3. 'clickbait_score': An integer rating from 1 to 10 on how exaggerated the title is.
4. 'summary_tag': A 1-sentence analytical takeaway.

Input Videos:
{json.dumps(sample_videos, indent=2)}

Return the output strictly in valid JSON format matching this schema:
[
  {{
    "id": "video_id",
    "ai_category": "string",
    "sentiment": "string",
    "clickbait_score": integer,
    "summary_tag": "string"
  }}
]
"""

# 4. Request Structured JSON from Gemini using the current model endpoint
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
    config=types.GenerateContentConfig(
        response_mime_type="application/json"
    )
)

print(" Successfully received response from Gemini API!\n")
print(response.text)