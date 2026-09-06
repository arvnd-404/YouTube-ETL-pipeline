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


import time
from google.genai.errors import APIError

def enrich_with_gemini(api_key: str, raw_videos: list) -> list:
    print("🤖 [2/4] Enriching video metadata with Gemini API...")
    client = genai.Client(api_key=api_key)
    
    # Prepare prompt and payload
    prompt = "..."  # your existing prompt logic
    
    max_retries = 5
    delay = 5  # Start with a 5-second wait

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",  # or your configured model
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json"
                ),
            )
            # If successful, parse response and return
            return parse_gemini_response(response.text, raw_videos)

        except APIError as e:
            if e.code in (503, 429) and attempt < max_retries:
                print(f"⚠️ Gemini API busy ({e.code}). Retrying in {delay}s... (Attempt {attempt}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff: 5s, 10s, 20s, 40s
            else:
                print(f"❌ Gemini enrichment failed: {e}")
                # Fallback: return raw videos with default values so BigQuery ingestion doesn't fail
                return fallback_enrichment(raw_videos)


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