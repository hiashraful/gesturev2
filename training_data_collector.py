import cv2
import mediapipe as mp
import csv
import copy
import itertools
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# Gesture mappings
GESTURE_NAMES = {
    0: "👍 Thumbs Up",
    1: "✌️ Peace Sign", 
    2: "🖕 Middle Finger",
    3: "🤘 Heavy Metal",
    4: "🤟 Our Sign"
}

def calc_landmark_list(image, landmarks):
    """Calculate landmark list from MediaPipe hand landmarks"""
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    """Preprocess landmark coordinates for ML model"""
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, landmark_list):
    """Save training data to CSV file"""
    if 0 <= number <= 4:  # 0-4 for your 5 gestures
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number] + landmark_list)

def count_samples():
    """Count existing samples for each gesture"""
    csv_path = 'model/keypoint_classifier/keypoint.csv'
    counts = {i: 0 for i in range(5)}
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].isdigit():
                    gesture_id = int(row[0])
                    if 0 <= gesture_id <= 4:
                        counts[gesture_id] += 1
    except FileNotFoundError:
        pass
    
    return counts

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    mode = 0  # 0: Normal, 1: Data Collection
    current_gesture = -1
    samples_collected_this_session = 0

    print("=== GESTURE TRAINING DATA COLLECTOR ===")
    print("Instructions:")
    print("1. Press 'k' to enter data collection mode")
    print("2. Press 0-4 to collect data for gestures:")
    print("   0: 👍 Thumbs Up")
    print("   1: ✌️ Peace Sign")
    print("   2: 🖕 Middle Finger") 
    print("   3: 🤘 Heavy Metal")
    print("   4: 🤟 Our Sign")
    print("3. Hold gesture steady and press number repeatedly")
    print("4. Collect 200+ samples per gesture for best results")
    print("5. Press ESC to quit")
    print("=" * 50)

    while True:
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        
        # Set mode for data collection
        if key == ord('k'):
            mode = 1 if mode == 0 else 0
            print(f"Mode changed to: {'Data Collection' if mode == 1 else 'Normal'}")
        
        # Gesture selection (0-4 for your 5 gestures)
        number = -1
        if 48 <= key <= 52:  # 0-4
            number = key - 48
            current_gesture = number
            if mode == 1:
                samples_collected_this_session += 1
                print(f"Collected sample #{samples_collected_this_session} for {GESTURE_NAMES[number]}")

        ret, image = cap.read()
        if not ret:
            break
        
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Calculate landmark list
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                # Preprocess landmarks
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Save training data
                if mode == 1 and number != -1:
                    logging_csv(number, pre_processed_landmark_list)

                # Draw landmarks
                mp_draw.draw_landmarks(
                    debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        # Get current sample counts
        sample_counts = count_samples()
        
        # Display information
        h, w, _ = debug_image.shape
        
        # Mode indicator
        mode_color = (0, 255, 0) if mode == 1 else (255, 255, 255)
        cv2.putText(debug_image, f"MODE: {'Data Collection' if mode == 1 else 'Normal'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Current gesture
        if current_gesture != -1:
            cv2.putText(debug_image, f"Current: {GESTURE_NAMES[current_gesture]}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Sample counts
        y_offset = 110
        cv2.putText(debug_image, "Sample Counts:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i in range(5):
            color = (0, 255, 0) if sample_counts[i] >= 200 else (0, 255, 255) if sample_counts[i] >= 100 else (255, 255, 255)
            cv2.putText(debug_image, f"{i}: {sample_counts[i]} samples", 
                       (10, y_offset + 25 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Instructions
        if mode == 1:
            cv2.putText(debug_image, "Hold gesture steady and press 0-4", (10, h - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(debug_image, "Aim for 200+ samples per gesture", (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            cv2.putText(debug_image, "Press 'k' for data collection mode", (10, h - 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(debug_image, "ESC to quit", (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Gesture Training Data Collection', debug_image)

    cap.release()
    cv2.destroyAllWindows()
    
    # Final summary
    final_counts = count_samples()
    print("\n=== FINAL SUMMARY ===")
    for i in range(5):
        status = "✅ Ready" if final_counts[i] >= 200 else "⚠️ Need more" if final_counts[i] >= 100 else "❌ Too few"
        print(f"{GESTURE_NAMES[i]}: {final_counts[i]} samples {status}")
    print("=" * 50)

if __name__ == '__main__':
    main()