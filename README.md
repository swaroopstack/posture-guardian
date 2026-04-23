# Posture Guardian

A real-time desktop posture assistant built with **Python**, **OpenCV**, and **MediaPipe Pose**.

Posture Guardian uses your webcam to track your upper-body landmarks, estimates a posture angle in real time, and gives instant visual feedback to help you maintain healthier sitting habits.

---

## Features

- **Real-time posture monitoring** using webcam input.
- **Personalized calibration** based on your own neutral sitting posture.
- **Live angle estimation** from ear → shoulder → hip landmarks.
- **On-screen posture status** (`Good Posture` / `Bad Posture!`).
- **Quick recalibration** with a single key press.
- **Optional audio alert support** (already scaffolded in the code).

---

## How It Works

1. The app opens your default webcam.
2. MediaPipe Pose detects body landmarks each frame.
3. It calculates the angle formed by:
   - left ear,
   - left shoulder,
   - left hip.
4. During initial calibration, you sit upright for 5 seconds.
5. That baseline angle is stored as your “good posture” reference.
6. If the live angle drops below the baseline by a threshold (10°), posture is flagged as bad.

---

## Project Structure

```text
posture-guardian/
├── src/
│   └── main.py          # Core posture tracking application
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Requirements

- Python **3.9+** (recommended)
- A webcam
- OS support for OpenCV display windows (Linux/macOS/Windows)

Python packages (in `requirements.txt`):

- `opencv-python`
- `mediapipe`
- `numpy`
- `pygame`

---

## Installation

```bash
# 1) Clone the repository
git clone <your-repo-url>
cd posture-guardian

# 2) (Optional but recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

---

## Usage

Run the app:

```bash
python src/main.py
```

### In-app controls

- **`q`** → Quit the application
- **`c`** → Recalibrate posture baseline

### First run behavior

On first launch, the app asks you to sit straight for 5 seconds while it calibrates your neutral posture angle.

---

## Optional Audio Alerts

Audio hooks are already present in `src/main.py` using `pygame`.

To enable:

1. Add an alert sound file (e.g., `alert.mp3`) to the project root.
2. Uncomment the relevant lines:
   - `pygame.mixer.music.load("alert.mp3")`
   - `pygame.mixer.music.play()` inside the bad-posture condition.

---

## Notes & Limitations

- Posture estimation uses **2D camera landmarks**, so depth and camera placement can affect accuracy.
- Works best when:
  - camera is at or slightly above shoulder height,
  - upper body is clearly visible,
  - lighting is consistent.
- Current logic tracks **left-side landmarks** only.

---

## Troubleshooting

- **Camera not opening**: ensure no other app is using the webcam.
- **No pose detected**: improve lighting and step fully into frame.
- **Pygame audio issues**: verify system audio drivers and file path to `alert.mp3`.

---

## Future Improvements

- Configurable posture thresholds.
- Smoothing/filtering to reduce jitter.
- Session statistics (time in good/bad posture).
- Cross-platform desktop packaging.
- Optional notifications and scheduled posture breaks.

---


## Acknowledgments

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://developers.google.com/mediapipe)
- [Pygame](https://www.pygame.org/)
