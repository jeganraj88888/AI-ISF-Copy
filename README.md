AI-ISF PROFESSIONAL V13

This version replaces the previous single-class LBPH identity check with OpenCV YuNet + SFace embeddings. An unknown face is compared against the enrolled owner embeddings instead of being automatically assigned to the owner class.

On first startup with internet access, the app downloads the official OpenCV Zoo YuNet and SFace ONNX models into data/face/. After that, inference is local.

Run:
  python -m pip install -r requirements.txt
  python app.py
Open: http://127.0.0.1:5000/

Enrollment is owner-only and requires the configured owner credentials. Re-enroll the owner once after upgrading from V12 so the SFace embeddings are created.

The OpenCV SFace reference implementation uses cosine similarity for identity matching; its documented default cosine threshold is 0.363. This project uses an owner-specific calibrated threshold with a conservative floor and multi-frame evidence.
