# Gesture-Controlled Gaming Using Computer Vision

A comprehensive system for real-time hand gesture recognition designed to control games and applications using Computer Vision. This project leverages **MediaPipe** for hand landmark extraction and compares **CNN (1D-Convolutional Neural Network)** and **MLP (Multi-Layer Perceptron)** models for high-accuracy gesture classification.

##  Features

- **Real-Time Recognition**: Low-latency gesture detection using MediaPipe.
- **Dual Model Support**: compare performance between an MLP and a 1D-CNN.
- **Custom Data Collection**: Easily record your own hand gestures to expand the dataset.
- **Interactive Game Controller**: Map detected gestures to keyboard keys (Space, Arrows, etc.) in real-time.
- **Dynamic Key Mapping**: Interactive "Mapping Mode" to change key bindings on the fly.
- **Visualization**: Detailed training history, confusion matrices, and classification reports included in the analysis.

##  Tech Stack

- **Computer Vision**: OpenCV, MediaPipe
- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Data Handling**: Pandas, Numpy, Pickle
- **Automation**: PyAutoGUI (for keyboard simulation)

##  Project Structure

```text
├── Game /                      # Real-time game control scripts
│   ├── gesture_game_controller.py  # Main gaming utility
│   └── gesture_mapping.json        # Saved gesture-to-key mappings
├── classification models/      # Trained models and artifacts
│   ├── CNN model/                  # CNN weights and inference script
│   ├── MLP model/                  # MLP weights and inference script
│   ├── labels.pkl                  # Label Encoder
│   └── scaler.pkl                  # Data Scaler
├── dataset/                    # Raw CSV landmark data
├── collect_data.py             # Script to record new gestures
└── gesture_classification.ipynb # Training and analysis notebook
```

##  Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/chanpanha126/Gesture-Controlled-Gaming-Using-Computer-Vision.git
   cd Gesture-Controlled-Gaming-Using-Computer-Vision
   ```

2. **Install Dependencies**:
   ```bash
   pip install opencv-python mediapipe tensorflow pandas numpy pyautogui scikit-learn matplotlib seaborn
   ```

##  How to Use

### 1. Collect Data (Optional)
If you want to train for new gestures, run:
```bash
python collect_data.py
```
- Enter the gesture name.
- Press **'s'** to start recording and **'q'** to save and quit.

### 2. Train the Models
Open `gesture_classification.ipynb` in Jupyter Notebook or Google Colab and run all cells. This will preprocess the data, train both MLP and CNN models, and export them to the `classification models/` folder.

### 3. Run the Game Controller
To start controlling games with your hands:
```bash
python "Game /gesture_game_controller.py"
```

**Controls:**
- **'m'**: Toggle **Mapping Mode**. In this mode, perform a gesture and press any key on your keyboard to link them.
- **'q'**: Quit the controller.

##  Performance

The project includes a detailed evaluation between the two architectures:
- **CNN**: Excellent at capturing spatial relationships between hand landmarks.
- **MLP**: Lightweight and fast, providing high accuracy for distinct gestures.

Detailed accuracy, loss curves, and confusion matrices can be generated directly within the training notebook.
