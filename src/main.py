import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

# Initialize mediapipe pose and pygame for sound
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pygame.mixer.init()

# Optional sound alert
# pygame.mixer.music.load("alert.mp3")

calibrated_angle = None
last_calibration_time = 0


def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


def calibrate_posture(angle):
    """Set the reference angle for good posture."""
    global calibrated_angle
    print("Sit straight for 5 seconds to calibrate your good posture...")
    time.sleep(5)
    calibrated_angle = angle
    print(f"✅ Calibration complete! Your good posture angle: {calibrated_angle:.2f}°")


# Start video capture
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            angle = calculate_angle(ear, shoulder, hip)

            # Run calibration if not done
            if calibrated_angle is None:
                calibrate_posture(angle)

            # Compare current angle with calibrated
            if angle < calibrated_angle - 10:
                posture = "Bad Posture!"
                color = (0, 0, 255)
                # pygame.mixer.music.play()
            else:
                posture = "Good Posture"
                color = (0, 255, 0)

            # Display info
            cv2.putText(image, f'Angle: {int(angle)} deg', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(image, posture, (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3, cv2.LINE_AA)
            cv2.putText(image, "Press 'C' to recalibrate", (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        except:
            pass

        cv2.imshow('Posture Guardian', image)

        # Key controls
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("🔄 Recalibration triggered.")
            calibrate_posture(angle)

cap.release()
cv2.destroyAllWindows()
