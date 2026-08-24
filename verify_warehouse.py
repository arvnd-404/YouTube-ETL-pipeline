import os
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("BIGQUERY_DATASET", "youtube_data")
KEY_PATH = os.path.abspath(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./gcp-key.json"))

credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=credentials, project=PROJECT_ID)

query = f"""
SELECT 
    video_id,
    title,
    channel_title,
    view_count,
    engagement_rate,
    ai_category,
    sentiment,
    clickbait_score,
    summary_tag
FROM `{PROJECT_ID}.{DATASET_ID}.trending_enriched`
ORDER BY extracted_at DESC
LIMIT 5
"""

print(f"📊 Querying latest enriched rows from BigQuery...\n")
query_job = client.query(query)
results = query_job.result()

for i, row in enumerate(results, 1):
    print(f"[{i}] {row.title}")
    print(f"    Channel: {row.channel_title} | Views: {row.view_count:,} | Engagement: {row.engagement_rate}%")
    print(f"    AI Category: {row.ai_category} | Sentiment: {row.sentiment} | Clickbait Score: {row.clickbait_score}/10")
    print(f"    Takeaway: {row.summary_tag}")
    print("-" * 80)