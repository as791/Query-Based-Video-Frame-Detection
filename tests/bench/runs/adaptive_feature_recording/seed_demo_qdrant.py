import json
import time
import urllib.request
import uuid

QDRANT = "http://qdrant:6333"
SOURCE_USER_ID = "00000000-0000-0000-0000-000000000003"
TARGET_USER_ID = "7b36da09-9ce6-4819-99ab-a68f7136e097"
DOMAIN_ID = "cctv-adaptive-demo"


def request_json(path, body, method="POST"):
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        QDRANT + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    request_json(
        "/collections/frames/points/delete?wait=true",
        {
            "filter": {
                "must": [
                    {"key": "user_id", "match": {"value": TARGET_USER_ID}},
                    {"key": "domain_id", "match": {"value": DOMAIN_ID}},
                    {"key": "demo_seed", "match": {"value": True}},
                ]
            }
        },
    )

    source = request_json(
        "/collections/frames/points/scroll",
        {
            "filter": {
                "must": [
                    {"key": "user_id", "match": {"value": SOURCE_USER_ID}},
                    {"key": "source_file", "match": {"text": "fall"}},
                ]
            },
            "limit": 10,
            "with_payload": True,
            "with_vector": True,
        },
    )["result"]["points"]

    points = []
    for index, point in enumerate(source):
        payload = dict(point["payload"])
        source_file = payload.get("source_file", f"demo_fall_{index}.mp4")
        payload.update(
            {
                "user_id": TARGET_USER_ID,
                "tenant_id": "default",
                "domain_id": DOMAIN_ID,
                "benchmark_run_id": "",
                "demo_seed": True,
                "source_file": f"demo_seed_{source_file}",
                "action_labels": ["fall", "lyingdown", "motion_posture"],
                "action_top": "fall",
                "action_scores": {"fall": 1.0, "lyingdown": 0.72, "motion_posture": 0.66},
                "action_confidence": 0.96,
                "tags": sorted(set([*payload.get("tags", []), "fall", "domain-demo", "feedback-demo"])),
            }
        )
        points.append(
            {
                "id": str(uuid.uuid4()),
                "vector": point["vector"],
                "payload": payload,
            }
        )

    request_json("/collections/frames/points?wait=true", {"points": points}, method="PUT")
    print(json.dumps({"seeded": len(points), "domainId": DOMAIN_ID, "targetUserId": TARGET_USER_ID}))


if __name__ == "__main__":
    main()
