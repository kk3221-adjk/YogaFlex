🧘‍♂️ YogaFlex

YogaFlex is a real-time yoga posture detection and feedback system designed to help users improve their alignment and stability while performing yoga poses. It leverages computer vision and pose estimation techniques to analyze body posture and provide instant corrective feedback.

Unlike basic pose detection systems, YogaFlex focuses on angle-based analysis and real-time feedback delivery, making it more interactive and user-centric.

📌 Overview

YogaFlex works by capturing live video input from a webcam and processing each frame to detect human body landmarks. These landmarks are then used to calculate joint angles and compare them with predefined ideal pose configurations.

The system provides:

Live visual feedback
Similarity scoring
Pose correction suggestions

This makes it useful for beginners and intermediate users practicing yoga without a trainer.

✨ Key Features
Real-Time Pose Detection
Processes live webcam feed using OpenCV
Angle-Based Pose Evaluation
Calculates joint angles and compares with ideal values
Multiple Pose Support
T Pose
Triangle Pose
Tree Pose
Mountain Pose
Crescent Lunge Pose
Warrior Pose
Instant Feedback System
Displays similarity score
Highlights incorrect posture
Suggests corrections
WebSocket Communication
Enables real-time interaction between backend and frontend
🛠 Tech Stack
Language: Python
Backend: FastAPI
Computer Vision: OpenCV
Pose Estimation: MediaPipe
Data Processing: NumPy
Frontend: HTML, CSS, JavaScript
📁 Project Structure
YogaFlex/
├── api/
│   └── main.py              # FastAPI backend & WebSocket handling
├── logic/
│   ├── T_pose.py
│   ├── triangle_pose.py
│   ├── Tree_pose.py
│   ├── Crescent_lunge_pose.py
│   ├── warrior_pose.py
│   └── mountain_pose.py
├── tests/
│   ├── index.html           # Frontend interface
│   ├── script.js
│   └── style.css
└── README.md
⚙️ How It Works
Video Capture
The system accesses the webcam using OpenCV.
Pose Detection
MediaPipe extracts body landmarks from each frame.
Angle Calculation
Joint angles are computed using geometric formulas.
Comparison with Ideal Pose
Angles are compared with predefined thresholds.
Feedback Generation
Similarity score is calculated
Corrections are generated
Annotated frame is returned
Real-Time Communication
Data is sent to the frontend via WebSockets
🔌 API Endpoint
WebSocket
URL: /ws/{client_id}
Sends:
Processed video frames (base64 encoded)
Feedback data:
Similarity score
Pose correction text
Joint-level insights
▶️ Running the Project
1. Clone Repository
git clone https://github.com/your-username/YogaFlex.git
cd YogaFlex
2. Install Dependencies
pip install fastapi uvicorn opencv-python mediapipe numpy
3. Run Backend
uvicorn api.main:app --reload
4. Open Frontend
Open tests/index.html in your browser
⚠️ Important Notes
Requires a local machine with webcam
Not suitable for cloud deployment without hardware access
Works best in good lighting conditions
🚀 Future Improvements
Add more yoga poses
Improve UI with React
Add voice-based feedback
Optimize performance using multithreading
Deploy using client-side camera processing
🎯 Contribution

This project focuses on:

Designing pose evaluation logic
Implementing angle-based comparison
Building real-time feedback system
📜 License

This project is intended for educational purposes.
