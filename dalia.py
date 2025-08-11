import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
import json
import threading
import time
import logging
from collections import Counter, deque
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

connected_clients = set()
data_lock = threading.Lock()
current_data = {
    "person_count": 0,
    "gesture": "unknown",
    "timestamp": time.time(),
    "status": "starting",
    "detection_triggered": False,
    "last_trigger_time": 0
}

MIN_DISTANCE = 0.5
MAX_DISTANCE = 2.0
CENTER_THRESHOLD = 0.3
COOLDOWN_TIME = 10

mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

def load_keypoint_classifier():
    try:
        keypoint_classifier = KeyPointClassifier()
        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = [row[0] for row in [line.strip().split(',') for line in f]]
        return keypoint_classifier, keypoint_classifier_labels
    except Exception as e:
        logger.error(f"Error loading keypoint classifier: {e}")
        return None, ["unknown"]

def load_point_history_classifier():
    try:
        point_history_classifier = PointHistoryClassifier()
        with open('model/point_history_classifier/point_history_classifier_label.csv', encoding='utf-8-sig') as f:
            point_history_classifier_labels = [row[0] for row in [line.strip().split(',') for line in f]]
        return point_history_classifier, point_history_classifier_labels
    except Exception as e:
        logger.error(f"Error loading point history classifier: {e}")
        return None, ["unknown"]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = landmark_list.copy()
    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(np.array(temp_landmark_list).flatten())
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def pre_process_point_history(image, point_history):
    image_width, image_height = image.shape[1], image.shape[0]
    temp_point_history = point_history.copy()
    base_x, base_y = temp_point_history[0][0], temp_point_history[0][1]
    for index, point in enumerate(temp_point_history):
        temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
    temp_point_history = list(np.array(temp_point_history).flatten())
    return temp_point_history

def detect_person(pose_landmarks):
    required_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]
    
    visible_count = 0
    for landmark_id in required_landmarks:
        landmark = pose_landmarks.landmark[landmark_id]
        if landmark.visibility > 0.5:
            visible_count += 1
    
    return visible_count >= 3

def check_detection_conditions(pose_landmarks, frame_width, frame_height):
    left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    
    shoulder_width = abs(right_shoulder.x - left_shoulder.x)
    torso_height = abs((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2)
    
    distance_estimate = 1.0 / (shoulder_width + 0.001)
    height_pixels = torso_height * frame_height
    
    center_x = (left_shoulder.x + right_shoulder.x) / 2
    distance_from_center = abs(center_x - 0.5)
    
    in_distance_range = MIN_DISTANCE <= distance_estimate <= MAX_DISTANCE
    in_center = distance_from_center <= CENTER_THRESHOLD
    
    return in_distance_range and in_center, distance_estimate, height_pixels

def should_trigger_detection():
    with data_lock:
        time_since_last = time.time() - current_data["last_trigger_time"]
        return time_since_last >= COOLDOWN_TIME

async def websocket_handler(websocket, path=""):
    client_ip = "unknown"
    try:
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        logger.info(f"Client connected from {client_ip}")
        connected_clients.add(websocket)
        
        welcome_msg = {
            "type": "connection",
            "message": "Connected to person and gesture detection",
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(welcome_msg))
        
        async for message in websocket:
            try:
                data = json.loads(message)
                if data.get("type") == "ping":
                    await websocket.send(json.dumps({
                        "type": "pong", 
                        "timestamp": time.time()
                    }))
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON from {client_ip}")
                continue
            except Exception as e:
                logger.error(f"Error processing message from {client_ip}: {e}")
                break
        
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_ip} disconnected normally")
    except websockets.exceptions.InvalidMessage as e:
        logger.info(f"Invalid WebSocket handshake from {client_ip}: {e}")
    except websockets.exceptions.ConnectionClosedError:
        logger.info(f"Connection closed unexpectedly for {client_ip}")
    except Exception as e:
        logger.error(f"Unexpected error for client {client_ip}: {e}")
    finally:
        connected_clients.discard(websocket)
        logger.info(f"Client {client_ip} cleaned up")

async def broadcast_data():
    while True:
        if connected_clients:
            with data_lock:
                message = json.dumps(current_data)
            
            disconnected_clients = set()
            for client in connected_clients.copy():
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception:
                    disconnected_clients.add(client)
            
            for client in disconnected_clients:
                connected_clients.discard(client)
        
        await asyncio.sleep(1/30)

def start_websocket_server():
    async def server():
        try:
            server = await websockets.serve(
                websocket_handler, 
                "localhost", 
                8765,
                ping_interval=20,
                ping_timeout=10,
                max_size=2**16,
                compression=None,
                process_request=None
            )
            
            logger.info("WebSocket server started on localhost:8765")
            broadcast_task = asyncio.create_task(broadcast_data())
            
            try:
                await server.wait_closed()
            except KeyboardInterrupt:
                logger.info("Shutting down WebSocket server...")
                broadcast_task.cancel()
                server.close()
                await server.wait_closed()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server())
    except Exception as e:
        logger.error(f"Failed to start WebSocket server: {e}")

def main():
    global current_data
    
    logger.info("Starting WebSocket server...")
    websocket_thread = threading.Thread(target=start_websocket_server, daemon=True)
    websocket_thread.start()
    
    time.sleep(2)
    
    logger.info("Loading ML models...")
    keypoint_classifier, keypoint_classifier_labels = load_keypoint_classifier()
    point_history_classifier, point_history_classifier_labels = load_point_history_classifier()
    
    if not keypoint_classifier:
        logger.error("Failed to load keypoint classifier")
        return
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 420)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        logger.error("Cannot open camera")
        return
    
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        model_complexity=0
    )
    
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    
    show_video = True
    
    logger.info("Starting main detection loop...")
    
    try:
        while True:
            ret, img = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
            
            img = cv2.flip(img, 1)
            debug_img = img.copy()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb.flags.writeable = False
            
            frame_height, frame_width = img.shape[:2]
            
            hand_result = hands.process(img_rgb)
            pose_result = pose.process(img_rgb)
            
            img_rgb.flags.writeable = True
            
            person_count = 0
            gesture = "unknown"
            detection_triggered = False
            
            if pose_result.pose_landmarks:
                if detect_person(pose_result.pose_landmarks):
                    person_count = 1
                    
                    in_range, distance, height_pixels = check_detection_conditions(
                        pose_result.pose_landmarks, frame_width, frame_height
                    )
                    
                    if in_range and should_trigger_detection():
                        detection_triggered = True
                        with data_lock:
                            current_data["last_trigger_time"] = time.time()
            
            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    landmark_list = calc_landmark_list(img, hand_landmarks)
                    pre_processed_landmark_list = pre_process_landmark(landmark_list)
                    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                    
                    if hand_sign_id == 2:
                        point_history.append(landmark_list[8])
                    else:
                        point_history.append([0, 0])
                    
                    finger_gesture_id = 0
                    if len(point_history) == history_length:
                        pre_processed_point_history_list = pre_process_point_history(img, point_history)
                        finger_gesture_id = point_history_classifier(pre_processed_point_history_list)
                    
                    finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(finger_gesture_history).most_common()
                    
                    if hand_sign_id < len(keypoint_classifier_labels):
                        gesture = keypoint_classifier_labels[hand_sign_id]
                        break
            else:
                point_history.append([0, 0])
            
            with data_lock:
                current_data.update({
                    "person_count": person_count,
                    "gesture": gesture,
                    "timestamp": time.time(),
                    "status": "running",
                    "detection_triggered": detection_triggered
                })
            
            if show_video:
                display_img = debug_img.copy()
                
                status_color = (0, 255, 0) if person_count > 0 else (0, 0, 255)
                # cv2.putText(display_img, f"Person: {person_count}", (10, 30), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                # cv2.putText(display_img, f"Gesture: {gesture}", (10, 70), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                # cv2.putText(display_img, f"Clients: {len(connected_clients)}", (10, 110), 
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                
                if hand_result.multi_hand_landmarks:
                    for hand_landmarks in hand_result.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            display_img, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                            mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2)
                        )
                
                if pose_result.pose_landmarks:
                    mp_draw.draw_landmarks(
                        display_img, pose_result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2)
                    )
                
                cv2.imshow("Enhanced Person & Gesture Detection", display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_video = not show_video
                if not show_video:
                    cv2.destroyAllWindows()
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
    finally:
        with data_lock:
            current_data["status"] = "stopped"
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        pose.close()
        logger.info("Cleanup completed")

if __name__ == "__main__":
    try:
        import websockets
        main()
    except ImportError:
        logger.error("websockets library not installed")
        exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)