import requests
import os

API_URL = "https://resq-server.onrender.com/api/violence-detected/"

def send_violence_event(
    image_paths,
    confidence_score,
    beacon_id,
    device_id="AI-VISION-SURVEILLANCE-01",
    description="Violent activity detected"
):
    if not image_paths or len(image_paths) > 3:
        raise ValueError("Must send 1–3 images only")

    files = []
    for img_path in image_paths:
        with open(img_path, "rb") as f:
            files.append(
                (
                    "images",  # SAME field name as endpoint
                    (
                        os.path.basename(img_path),
                        f.read(),              # RAW IMAGE BYTES
                        "image/jpeg"
                    )
                )
            )

    data = {
        "beacon_id": beacon_id,
        "confidence_score": str(confidence_score),
        "description": description,
        "device_id": device_id,
    }

    response = requests.post(
        API_URL,
        data=data,
        files=files,
        timeout=10
    )

    response.raise_for_status()
    print("✅ API response:", response.json())
    return response.json()
