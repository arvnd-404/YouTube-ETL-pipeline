import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from google.cloud import bigquery
from google.oauth2 import service_account

# 1. Load environment variables
load_dotenv()
project_id = os.getenv("GCP_PROJECT_ID")
dataset_id = os.getenv("BIGQUERY_DATASET", "youtube_data")
key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./gcp-key.json")

absolute_key_path = os.path.abspath(key_path)

# 2. Authenticate explicitly
credentials = service_account.Credentials.from_service_account_file(absolute_key_path)
client = bigquery.Client(credentials=credentials, project=project_id)

table_ref = f"{project_id}.{dataset_id}.trending_enriched"
print(f" Connecting to BigQuery table via Batch Load: {table_ref}...")

# 3. Create test records
test_rows = [
    {
        "video_id": "test_001",
        "title": "Test Pipeline Run",
        "channel_title": "OmniTrend Data Team",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "view_count": 50000,
        "like_count": 4200,
        "comment_count": 350,
        "engagement_rate": 9.1,
        "ai_category": "Tech & Engineering",
        "sentiment": "Positive",
        "clickbait_score": 2,
        "summary_tag": "Initial connectivity test verifying free-tier batch load pipeline.",
        "extracted_at": datetime.now(timezone.utc).isoformat()
    }
]

# 4. Configure Free Tier Batch Load Job
job_config = bigquery.LoadJobConfig(
    write_disposition=bigquery.WriteDisposition.WRITE_APPEND
)

try:
    load_job = client.load_table_from_json(test_rows, table_ref, job_config=job_config)
    print("⏳ Submitting load job...")
    load_job.result()  # Wait for the batch load to complete

    print(" Successfully inserted test record into Google BigQuery using Free Tier Batch Load!")

except Exception as e:
    print(f"\n❌ Execution failed with error:\n{type(e).__name__}: {e}")