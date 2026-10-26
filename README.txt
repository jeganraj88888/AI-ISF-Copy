AI-ISF V4 — Owner Face Identity Authentication

1. Remove conflicting OpenCV packages:
   pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
2. Install:
   pip install -r requirements.txt
3. Run:
   python app.py
4. Open:
   http://127.0.0.1:5000/
5. First run: enter owner email/password and click Enroll My Face. Capture 15 live samples.
6. Future login: password is checked first, then the camera compares the face against the enrolled owner model. A non-matching face is denied.

Security note: this uses OpenCV LBPH recognition for a local academic/prototype security project. It is not equivalent to Windows Hello/Apple Face ID and does not provide certified anti-spoof/liveness guarantees.
