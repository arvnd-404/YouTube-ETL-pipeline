import os
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

# Modern Data Processing Engine
import polars as pl

# Cloud & AI SDKs
from googleapiclient.discovery import build
from google import genai
from google.genai import types
from google.cloud import bigquery
from google.oauth2 import service_account

# 1. Environment & Credentials
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
BIGQUERY_DATASET = os.getenv("BIGQUERY_DATASET", "youtube_data")
KEY_PATH = os.path.abspath(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./gcp-key.json"))


def extract_trending_videos(api_key: str, region_code: str = "IN", max_results: int = 50) -> list:
    """Extract trending videos from YouTube Data API v3 (1 API quota unit)."""
    print("🚀 [1/4] Extracting trending videos from YouTube API...")
    youtube = build("youtube", "v3", developerKey=api_key)
    
    request = youtube.videos().list(
        part="snippet,statistics",
        chart="mostPopular",
        regionCode=region_code,
        maxResults=max_results
    )
    response = request.execute()
    items = response.get("items", [])
    
    extracted_data = []
    for item in items:
        extracted_data.append({
            "video_id": item["id"],
            "title": item["snippet"]["title"],
            "channel_title": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"],
            "view_count": int(item["statistics"].get("viewCount", 0)),
            "like_count": int(item["statistics"].get("likeCount", 0)),
            "comment_count": int(item["statistics"].get("commentCount", 0)),
        })
    
    print(f"       ✅ Extracted {len(extracted_data)} video records.")
    return extracted_data


def enrich_with_gemini(api_key: str, videos: list) -> list:
    """Enrich video titles in a single batch prompt using Gemini 3.6 Flash."""
    print("🤖 [2/4] Enriching video metadata with Gemini API...")
    client = genai.Client(api_key=api_key)
    
    batch_input = [{"id": v["video_id"], "title": v["title"]} for v in videos]
    
    prompt = f"""
    Analyze the following YouTube video titles. For each video, provide:
    1. 'ai_category': Granular genre (e.g., Entertainment, Tech, Finance, Gaming, Politics, Music).
    2. 'sentiment': Positive, Neutral, or Negative.
    3. 'clickbait_score': Rating from 1 (completely factual) to 10 (extreme clickbait).
    4. 'summary_tag': A 1-sentence analytical takeaway.

    Input:
    {json.dumps(batch_input)}

    Return strictly a JSON array matching:
    [
      {{
        "id": "video_id",
        "ai_category": "string",
        "sentiment": "string",
        "clickbait_score": 5,
        "summary_tag": "string"
      }}
    ]
    """
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json")
    )
    
    ai_results = json.loads(response.text)
    ai_lookup = {item["id"]: item for item in ai_results}
    
    now_iso = datetime.now(timezone.utc).isoformat()
    enriched_records = []
    for v in videos:
        ai_meta = ai_lookup.get(v["video_id"], {})
        enriched_records.append({
            **v,
            "ai_category": ai_meta.get("ai_category", "General"),
            "sentiment": ai_meta.get("sentiment", "Neutral"),
            "clickbait_score": int(ai_meta.get("clickbait_score", 5)),
            "summary_tag": ai_meta.get("summary_tag", "Trending topic."),
            "extracted_at": now_iso
        })
    
    print("       ✅ AI categorization and sentiment enrichment completed.")
    return enriched_records


def transform_with_polars(records: list) -> pl.DataFrame:
    """Clean and transform data using multi-threaded Polars expressions."""
    print("⚡ [3/4] Transforming metrics with Polars Engine...")
    
    # Ingest into Polars DataFrame
    df = pl.DataFrame(records)
    
    # Apply vectorized transformations
    transformed_df = df.with_columns([
        # Parse ISO timestamps to UTC and make them timezone-naive to avoid Windows tz bugs
        pl.col("published_at").str.to_datetime(time_zone="UTC").dt.replace_time_zone(None),
        pl.col("extracted_at").str.to_datetime(time_zone="UTC").dt.replace_time_zone(None),
        
        # Calculate engagement rate: ((likes + comments) / views) * 100
        pl.when(pl.col("view_count") > 0)
        .then(
            ((pl.col("like_count") + pl.col("comment_count")) / pl.col("view_count") * 100).round(2)
        )
        .otherwise(0.0)
        .alias("engagement_rate"),
        
        # Ensure strict integer casting
        pl.col("clickbait_score").cast(pl.Int64),
        pl.col("view_count").cast(pl.Int64),
        pl.col("like_count").cast(pl.Int64),
        pl.col("comment_count").cast(pl.Int64),
    ])
    
    print("\n       --- Polars Schema & Preview ---")
    print(transformed_df.schema)
    print(transformed_df.head(2))
    return transformed_df


def load_to_bigquery(df: pl.DataFrame, project_id: str, dataset_id: str, key_path: str):
    """Load transformed Polars DataFrame directly into BigQuery via Batch Load."""
    print("\n☁️ [4/4] Loading batch into BigQuery...")
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = bigquery.Client(credentials=credentials, project=project_id)
    table_ref = f"{project_id}.{dataset_id}.trending_enriched"
    
    # Export Polars records directly to Python dictionaries with ISO timestamps
    records = []
    for row in df.iter_rows(named=True):
        row["published_at"] = row["published_at"].isoformat() if row.get("published_at") else None
        row["extracted_at"] = row["extracted_at"].isoformat() if row.get("extracted_at") else None
        records.append(row)
    
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND
    )
    
    job = client.load_table_from_json(records, table_ref, job_config=job_config)
    job.result()  # Wait for batch completion
    print(f"\n🎉 Pipeline executed successfully! {len(records)} rows committed to {table_ref}.")

def run_pipeline():
    raw_videos = extract_trending_videos(YOUTUBE_API_KEY)
    enriched_data = enrich_with_gemini(GEMINI_API_KEY, raw_videos)
    transformed_df = transform_with_polars(enriched_data)
    load_to_bigquery(transformed_df, GCP_PROJECT_ID, BIGQUERY_DATASET, KEY_PATH)


if __name__ == "__main__":
    run_pipeline()