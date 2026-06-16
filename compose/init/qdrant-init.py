import time
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PayloadSchemaType

def wait_for_qdrant(client, retries=30, delay=2):
    for _ in range(retries):
        try:
            client.get_collections()
            return
        except Exception:
            time.sleep(delay)
    print("Qdrant not reachable after retries", file=sys.stderr)
    sys.exit(1)

def ensure_collection(client, name, vector_size=768):
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"Created collection: {name}")
    else:
        print(f"Collection already exists: {name}")

def create_indexes(client, name, keyword_fields, integer_fields):
    for field in keyword_fields:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except Exception as exc:
            print(f"Index {name}.{field} already exists or could not be created: {exc}")
    for field in integer_fields:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=PayloadSchemaType.INTEGER,
            )
        except Exception as exc:
            print(f"Index {name}.{field} already exists or could not be created: {exc}")
    print(f"Indexes created for collection: {name}")

def main():
    client = QdrantClient(host="qdrant", port=6333)
    wait_for_qdrant(client)

    # chunks collection — one vector per video chunk (shot or fixed window)
    ensure_collection(client, "chunks")
    create_indexes(client, "chunks",
        keyword_fields=[
            "user_id", "tenant_id", "video_id", "chunk_id", "profile",
            "tags", "objects", "main_activity", "scene", "motion", "source_file",
        ],
        integer_fields=["t_start_ms", "t_end_ms", "duration_ms", "width", "height"],
    )

    # frames collection — one vector per sampled frame inside a chunk
    ensure_collection(client, "frames")
    create_indexes(client, "frames",
        keyword_fields=[
            "user_id", "tenant_id", "video_id", "chunk_id", "frame_id", "profile",
            "tags", "objects", "main_activity", "scene", "motion", "source_file",
        ],
        integer_fields=["t_ms", "duration_ms", "width", "height"],
    )

    print("Qdrant collections ready.")

if __name__ == "__main__":
    main()
