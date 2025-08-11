# Hand Gesture Recognition System

A comprehensive real-time hand gesture recognition system using MediaPipe and TensorFlow, featuring a web interface and WebSocket communication for interactive applications.

## 📁 Project Structure

```
GESTUREV3/
├── .vscode/                              # VS Code configuration
├── model/                                # ML models and data
│   ├── keypoint_classifier/
│   │   ├── __init__.py
│   │   ├── keypoint_classifier.py        # TensorFlow Lite model class
│   │   ├── keypoint_classifier.tflite    # Trained model (generated)
│   │   ├── keypoint_classifier.keras     # Keras model (generated)
│   │   ├── keypoint.csv                  # Training data
│   │   └── keypoint_classifier_label.csv # Gesture labels
│   └── point_history_classifier/
│       ├── point_history_classifier.py   # Dynamic gesture classifier
│       ├── point_history_classifier.tflite # Model for movement patterns
│       └── point_history_classifier_label.csv # Movement labels
├── utils/
│   ├── __init__.py
│   └── cvfpscalc.py                      # FPS calculation utility
├── app.py                                # Basic gesture recognition app
├── dalia.py                              # Enhanced app with WebSocket support
├── digital_comm_interface.html           # Web interface (intro video)
├── gesture_recognition.html              # Web interface (gesture reactive)
├── keypoint_classification.ipynb         # Model training notebook
├── training_data_collector.py            # Data collection tool
├── dalia_intro.mp4                       # Intro video
├── default_idle3.webm                    # Idle animation
├── inteactive_dalia_THUMBSUP_1.mp4      # Thumbs up reaction video
└── interactive_dalia_PEACE_1.mp4        # Peace sign reaction video
```

## 🚀 Features

- **Real-time Hand Gesture Recognition**: Recognizes 5 different gestures
- **Machine Learning Based**: Uses TensorFlow Lite for efficient inference
- **Web Interface**: Interactive web-based interface with video responses
- **WebSocket Communication**: Real-time data streaming
- **Training Pipeline**: Complete workflow for collecting data and training models
- **Multiple Interfaces**: Basic OpenCV interface and advanced web interface

## 📋 Supported Gestures

1. 👍 **Thumbs Up**
2. ✌️ **Peace Sign**
3. 🖕 **Middle Finger**
4. 🤘 **Heavy Metal**
5. 🤟 **Our Sign**

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Webcam/Camera
- Web browser (for web interface)

### Required Libraries

Install the required packages using pip:

```bash
pip install opencv-python
pip install mediapipe
pip install tensorflow
pip install numpy
pip install websockets
pip install pandas
pip install scikit-learn
pip install seaborn
pip install matplotlib
pip install jupyter
```

Or install all at once:

```bash
pip install opencv-python mediapipe tensorflow numpy websockets pandas scikit-learn seaborn matplotlib jupyter
```

### Alternative Installation with requirements.txt

Create a `requirements.txt` file with the following content:

```txt
opencv-python>=4.5.0
mediapipe>=0.10.0
tensorflow>=2.10.0
numpy>=1.21.0
websockets>=10.0
pandas>=1.3.0
scikit-learn>=1.0.0
seaborn>=0.11.0
matplotlib>=3.5.0
jupyter>=1.0.0
```

Then install:

```bash
pip install -r requirements.txt
```

## 📖 Step-by-Step Usage Guide

### Step 1: Collect Training Data

Before using the gesture recognition system, you need to collect training data for your gestures.

1. **Run the data collector**:
   ```bash
   python training_data_collector.py
   ```

2. **Follow the instructions**:
   - Press `k` to enter data collection mode
   - Hold your hand in the desired gesture position
   - Press the corresponding number key (0-4) repeatedly
   - Collect **200+ samples** per gesture for best results

3. **Gesture mapping**:
   - Press `0`: 👍 Thumbs Up
   - Press `1`: ✌️ Peace Sign
   - Press `2`: 🖕 Middle Finger
   - Press `3`: 🤘 Heavy Metal
   - Press `4`: 🤟 Our Sign

4. **Tips for good training data**:
   - Vary hand positions slightly
   - Try different lighting conditions
   - Use both left and right hands
   - Keep gestures clear and distinct

### Step 2: Train the Model

1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook keypoint_classification.ipynb
   ```

2. **Run all cells** in the notebook to:
   - Load and preprocess the training data
   - Train the neural network model
   - Evaluate model performance
   - Convert to TensorFlow Lite format
   - Save the trained model

3. **Check model performance**:
   - Look at the confusion matrix
   - Aim for >95% accuracy
   - If accuracy is low, collect more training data

### Step 3: Test Basic Recognition

Run the basic gesture recognition application:

```bash
python app.py
```

**Controls**:
- **ESC**: Quit the application
- **n**: Normal mode
- **k**: Keypoint logging mode
- **h**: Point history logging mode

### Step 4: Run Enhanced Application

For the full-featured application with WebSocket support:

```bash
python dalia.py
```

**Features**:
- Real-time gesture recognition
- Person detection and distance estimation
- WebSocket server for web interface communication
- Enhanced UI with detection status

### Step 5: Launch Web Interface

1. **Open a web browser**
2. **Open either HTML file**:
   - `digital_comm_interface.html` - Shows intro video when person detected
   - `gesture_recognition.html` - Responds to specific gestures

3. **Make sure** `dalia.py` is running for WebSocket communication

## ⚙️ Configuration Options

### Model Parameters

Edit the model parameters in the respective files:

**Detection Confidence** (in `dalia.py`):
```python
hands = mp_hands.Hands(
    min_detection_confidence=0.8,  # Adjust detection sensitivity
    min_tracking_confidence=0.7    # Adjust tracking sensitivity
)
```

**Distance Parameters**:
```python
MIN_DISTANCE = 0.5  # meters
MAX_DISTANCE = 2.0  # meters
CENTER_THRESHOLD = 0.3  # 30% from center
COOLDOWN_TIME = 10  # seconds
```

### Training Parameters

Modify training parameters in `keypoint_classification.ipynb`:

```python
# Model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])
```

## 🔧 Troubleshooting

### Common Issues

1. **Camera not working**:
   ```python
   # Try different camera indices
   cap = cv2.VideoCapture(1)  # Instead of 0
   ```

2. **Model not found error**:
   - Make sure you've trained the model using the Jupyter notebook
   - Check if `.tflite` files exist in the model directories

3. **Poor gesture recognition**:
   - Collect more training data (aim for 300+ samples per gesture)
   - Ensure good lighting conditions
   - Make gestures more distinct

4. **WebSocket connection failed**:
   - Make sure `dalia.py` is running
   - Check if port 8765 is available
   - Try refreshing the web page

### Performance Optimization

1. **Reduce video resolution**:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

2. **Adjust MediaPipe model complexity**:
   ```python
   hands = mp_hands.Hands(
       model_complexity=0,  # 0 for faster, 1 for more accurate
       max_num_hands=1
   )
   ```

## 📝 Adding New Gestures

1. **Update gesture mappings** in `training_data_collector.py`:
   ```python
   GESTURE_NAMES = {
       0: "👍 Thumbs Up",
       1: "✌️ Peace Sign", 
       2: "🖕 Middle Finger",
       3: "🤘 Heavy Metal",
       4: "🤟 Our Sign",
       5: "🆕 Your New Gesture"  # Add new gesture
   }
   ```

2. **Increase NUM_CLASSES** in the training notebook:
   ```python
   NUM_CLASSES = 6  # Update from 5 to 6
   ```

3. **Collect training data** for the new gesture
4. **Retrain the model** using the notebook
5. **Update label files** with the new gesture name

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- [MediaPipe](https://google.github.io/mediapipe/) for hand tracking
- [TensorFlow](https://tensorflow.org/) for machine learning
- Original gesture recognition concepts from various computer vision research

---

**Happy Gesture Recognition! 👋**