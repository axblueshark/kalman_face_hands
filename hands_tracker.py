"""
Hands detector

run:
    python3 hands_tracker.py
    python3 hands_tracker.py --camera 1

controls:
    q  quit
    r  reset Kalman trackers
"""

import argparse
import csv
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


OUTPUT_FILE = "tracking_data.csv"

# fixed size of the displayed camera frame
WIDTH = 1280
HEIGHT = 720

# MediaPipe hand landmarker model file
MODEL_PATH = "hand_landmarker.task"


class KalmanPointTracker:
    """
    Kalman tracker for one moving 2D point.

    The tracked point is the center of one detected hand.
    The state contains position and velocity: [x, y, vx, vy]
    """

    def __init__(self):
        # 4 state variables, 2 measured variables
        self.kf = cv2.KalmanFilter(4, 2)

        # transition matrix
        # x_new = x + vx, y_new = y + vy
        # vx_new = vx, vy_new = vy
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]],
            dtype=np.float32,
        )

        # measurement matrix
        # the detector only measures position [x, y]
        # velocity is hidden and estimated by the Kalman filter
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float32,
        )

        # process noise: how much to trust the motion model
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

        # measurement noise: how noisy the detector measurements are
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 12.0

        # initial uncertainty
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1000

        self.initialized = False

    def predict(self):
        if not self.initialized:
            return None

        return self.kf.predict()

    def correct(self, x, y):
        # convert the measured point to the format expected by OpenCV
        measurement = np.array(
            [[np.float32(x)], [np.float32(y)]],
            dtype=np.float32,
        )

        # first measurement initializes the state directly
        if not self.initialized:
            self.kf.statePost = np.array(
                [[np.float32(x)], [np.float32(y)], [0.0], [0.0]],
                dtype=np.float32,
            )
            self.initialized = True

        # later measurements correct the predicted state
        else:
            self.kf.correct(measurement)


def open_camera(camera_index):
    """
    Open the selected camera index.

    The default index is 1, but can be changed from the command line,
    for example --camera 0.
    """

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        cap.release()
        return None

    # test whether the camera can actually provide a frame
    ok, frame = cap.read()

    if not ok or frame is None:
        cap.release()
        return None

    print(f"Using camera index {camera_index}")
    return cap


def create_hand_landmarker():
    """
    Create the MediaPipe hand detector.

    The newer MediaPipe API uses a separate .task model file.
    The detector runs in VIDEO mode, so each frame must have a timestamp.
    """

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(options)


def get_hand_center(hand_landmarks):
    """
    Compute one representative point for a detected hand.

    MediaPipe returns 21 landmarks for each hand.
    Here we take the average of all landmark coordinates and use it as the
    measured hand center.
    """

    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    # MediaPipe coordinates are normalized to the interval [0, 1]
    # therefore they must be converted to pixel coordinates
    cx = int(np.mean(xs) * WIDTH)
    cy = int(np.mean(ys) * HEIGHT)

    return cx, cy


def draw_hand_skeleton(frame, hand_landmarks, color):
    """
    Draw the hand skeleton manually.
    """

    points = []

    # convert normalized landmark coordinates to pixel coordinates
    for lm in hand_landmarks:
        x = int(lm.x * WIDTH)
        y = int(lm.y * HEIGHT)
        points.append((x, y))

    # landmark connections for fingers and palm
    connections = [
        # thumb
        (0, 1), (1, 2), (2, 3), (3, 4),

        # index finger
        (0, 5), (5, 6), (6, 7), (7, 8),

        # middle finger
        (0, 9), (9, 10), (10, 11), (11, 12),

        # ring finger
        (0, 13), (13, 14), (14, 15), (15, 16),

        # little finger
        (0, 17), (17, 18), (18, 19), (19, 20),

        # palm connections
        (5, 9), (9, 13), (13, 17),
    ]

    # draw connections between landmarks
    for a, b in connections:
        cv2.line(frame, points[a], points[b], color, 2)

    # draw each landmark point
    for p in points:
        cv2.circle(frame, p, 3, color, -1)


def draw_and_log_hands(frame, hand_result, trackers, writer, timestamp):
    """
    Process detected hands.

    For each detected hand:
    - compute its center,
    - assign left/right label,
    - predict its position using Kalman filter,
    - correct the filter with the current measurement,
    - draw the hand skeleton,
    - write measurement and prediction to csv.
    """

    hands = []

    for hand_landmarks in hand_result.hand_landmarks:
        cx, cy = get_hand_center(hand_landmarks)
        hands.append((cx, cy, hand_landmarks))

    # sort hands by x coordinate
    hands.sort(key=lambda item: item[0])

    # assign labels
    if len(hands) == 1:
        labels = ["hand_left" if hands[0][0] < WIDTH / 2 else "hand_right"]
    else:
        labels = ["hand_left", "hand_right"][:len(hands)]

    # predict both trackers before correction
    predictions = {
        label: tracker.predict()
        for label, tracker in trackers.items()
    }

    for label, (cx, cy, hand_landmarks) in zip(labels, hands):
        pred = predictions[label]

        # update the tracker using the measured hand center
        trackers[label].correct(cx, cy)

        pred_x = ""
        pred_y = ""

        if pred is not None:
            pred_x = int(pred[0, 0])
            pred_y = int(pred[1, 0])
            cv2.circle(frame, (pred_x, pred_y), 5, (0, 0, 255), -1)

        writer.writerow([timestamp, label, cx, cy, pred_x, pred_y])

        color = (0, 255, 0) if label == "hand_left" else (0, 255, 255)
        text = "left hand" if label == "hand_left" else "right hand"

        draw_hand_skeleton(frame, hand_landmarks, color)
        cv2.circle(frame, (cx, cy), 6, color, -1)

        cv2.putText(
            frame, text,
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )


def draw_stats_panel(frame, hand_count):
    """
    Draw a small status panel in the top-left corner.
    """

    # black background rectangle
    cv2.rectangle(frame, (10, 10), (330, 65), (0, 0, 0), -1)

    # number of detected hands
    cv2.putText(
        frame,
        f"hands detected: {hand_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
    )


def create_trackers():
    """
    Create one Kalman tracker for each hand.
    """

    return {
        "hand_left": KalmanPointTracker(),
        "hand_right": KalmanPointTracker(),
    }


def run_detector(camera_index):
    """
    Main detector loop.

    This function:
    - opens the camera,
    - loads the MediaPipe hand landmarker,
    - reads frames in a loop,
    - detects hands,
    - writes measurements to csv,
    - displays the annotated video.
    """

    cap = open_camera(camera_index)

    if cap is None:
        print("Error: no working camera stream found")
        return

    trackers = create_trackers()

    # create display window
    window_name = "Hand measurements"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIDTH, HEIGHT)

    try:
        # create hand detector and output csv file
        with create_hand_landmarker() as hand_landmarker, open(
            OUTPUT_FILE,
            "w",
            newline="",
        ) as csv_file:
            writer = csv.writer(csv_file)

            # header expected by the notebook
            writer.writerow([
                "timestamp", "obj_type",
                "meas_x", "meas_y",
                "pred_x", "pred_y"])

            print("Detector started")
            print("Controls: q=quit, r=reset Kalman trackers")

            # MediaPipe video mode requires increasing timestamps
            # in milliseconds
            start_time = time.time()

            while True:
                ok, frame = cap.read()

                if not ok:
                    print("Error: could not read webcam frame")
                    break

                # mirror the image so the preview behaves 
                # like a normal webcam mirror
                frame = cv2.flip(frame, 1)

                # resize to fixed size for consistent coordinates
                frame = cv2.resize(frame, (WIDTH, HEIGHT))

                timestamp = datetime.now().strftime("%H:%M:%S.%f")

                # MediaPipe expects rgb image, while OpenCV uses bgr
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # array memory layout must be compatible with MediaPipe
                rgb = np.ascontiguousarray(rgb)

                # convert numpy array to MediaPipe image object
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb,
                )

                # timestamp in milliseconds for MediaPipe video mode
                timestamp_ms = int((time.time() - start_time) * 1000)

                # run hand detection
                hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)

                # draw hands and log their measurements
                draw_and_log_hands(frame, hand_result, trackers, writer, timestamp)

                # draw status panel
                draw_stats_panel(frame, len(hand_result.hand_landmarks))

                # show annotated frame
                cv2.imshow(window_name, frame)

                key = cv2.waitKey(1) & 0xFF

                # quit program
                if key == ord("q"):
                    break

                # reset Kalman filters
                if key == ord("r"):
                    trackers = create_trackers()
                    print("Kalman trackers reset")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Saved measurements to {OUTPUT_FILE}")


def parse_args():
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_detector(args.camera)
