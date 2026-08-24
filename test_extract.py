import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

# 1. Load variables from .env
load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")

if not api_key:
    raise ValueError("YOUTUBE_API_KEY not found in .env file. Please check your configuration.")

# 2. Initialize the YouTube API client
youtube = build("youtube", "v3", developerKey=api_key)

print(" Connecting to YouTube Data API...")

# 3. Request the top 50 trending videos (e.g., regionCode='IN' or 'US')
request = youtube.videos().list(
    part="snippet,statistics,contentDetails",
    chart="mostPopular",
    regionCode="IN",  # Change to 'US' or your preferred region
    maxResults=50
)

response = request.execute()
items = response.get("items", [])

print(f" Successfully fetched {len(items)} trending videos!\n")

# 4. Print a sample of the first 3 videos to inspect data fields
print("--- Sample Trending Records ---")
for idx, item in enumerate(items[:3], start=1):
    title = item["snippet"]["title"]
    channel = item["snippet"]["channelTitle"]
    views = item["statistics"].get("viewCount", "0")
    likes = item["statistics"].get("likeCount", "0")
    published_at = item["snippet"]["publishedAt"]
    
    print(f"[{idx}] {title}")
    print(f"    Channel: {channel} | Views: {views} | Likes: {likes}")
    print(f"    Published: {published_at}")
    print("-" * 50)