import json
import mimetypes
import pathlib
import sys
import urllib.request

API = "http://api:8080"
GOOGLE_SUB = "106618487278266707487"
EMAIL = "aryamansinha123@gmail.com"


def main():
    clip_path = pathlib.Path(sys.argv[1])
    boundary = "----videovault-demo-boundary"
    content_type = mimetypes.guess_type(clip_path.name)[0] or "video/mp4"
    file_bytes = clip_path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{clip_path.name}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            file_bytes,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
    )
    url = f"{API}/v1/video/upload?profile=cctv&fewShotLabel=fall&domainId=cctv-adaptive-demo"
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
            "X-Benchmark-Google-Sub": GOOGLE_SUB,
            "X-Benchmark-Email": EMAIL,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        print(response.read().decode("utf-8"))


if __name__ == "__main__":
    main()
