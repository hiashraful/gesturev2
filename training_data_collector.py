import cv2
import mediapipe as mp
import csv
import copy
import itertools
import os

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

GESTURE_NAMES = {
    0: "👍 Thumbs Up",
    1: "✌️ Peace", 
    2: "🖕 Middle Finger",
    3: "🤘 Rock On",
    4: "❤️ Heart",
    5: "👊 Fist Bump",
    6: "🔲 Cuadro",
    7: "👋 Waving"
}

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value if max_value != 0 else 0

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, landmark_list):
    if 0 <= number <= 7:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number] + landmark_list)

def count_samples():
    csv_path = 'model/keypoint_classifier/keypoint.csv'
    counts = {i: 0 for i in range(8)}
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0].isdigit():
                    gesture_id = int(row[0])
                    if 0 <= gesture_id <= 7:
                        counts[gesture_id] += 1
    except FileNotFoundError:
        pass
    
    return counts

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    mode = 0
    current_gesture = -1
    samples_collected_this_session = 0

    print("=== GESTURE TRAINING DATA COLLECTOR ===")
    print("Instructions:")
    print("1. Press 'k' to enter data collection mode")
    print("2. Press 0-7 to collect data for gestures:")
    for i, name in GESTURE_NAMES.items():
        print(f"   {i}: {name}")
    print("3. Hold gesture steady and press number repeatedly")
    print("4. Collect 200+ samples per gesture for best results")
    print("5. Press ESC to quit")
    print("=" * 50)

    while True:
        key = cv2.waitKey(10)
        if key == 27:
            break
        
        if key == ord('k'):
            mode = 1 if mode == 0 else 0
            print(f"Mode changed to: {'Data Collection' if mode == 1 else 'Normal'}")
        
        number = -1
        if 48 <= key <= 55:
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

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                if mode == 1 and number != -1:
                    logging_csv(number, pre_processed_landmark_list)

                mp_draw.draw_landmarks(
                    debug_image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

        sample_counts = count_samples()
        
        h, w, _ = debug_image.shape
        
        mode_color = (0, 255, 0) if mode == 1 else (255, 255, 255)
        cv2.putText(debug_image, f"MODE: {'Data Collection' if mode == 1 else 'Normal'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        if current_gesture != -1:
            cv2.putText(debug_image, f"Current: {GESTURE_NAMES[current_gesture]}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        y_offset = 110
        cv2.putText(debug_image, "Sample Counts:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for i in range(8):
            color = (0, 255, 0) if sample_counts[i] >= 200 else (0, 255, 255) if sample_counts[i] >= 100 else (255, 255, 255)
            cv2.putText(debug_image, f"{i}: {sample_counts[i]} samples", 
                       (10, y_offset + 25 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        if mode == 1:
            cv2.putText(debug_image, "Hold gesture steady and press 0-7", (10, h - 60), 
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
    
    final_counts = count_samples()
    print("\n=== FINAL SUMMARY ===")
    for i in range(8):
        status = "✅ Ready" if final_counts[i] >= 200 else "⚠️ Need more" if final_counts[i] >= 100 else "❌ Too few"
        print(f"{GESTURE_NAMES[i]}: {final_counts[i]} samples {status}")
    print("=" * 50)

if __name__ == '__main__':
    main()