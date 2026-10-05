# 🔬 Image Analytics Lab

An interactive **Image & Video Analytics laboratory** built with **Streamlit**, demonstrating classical and deep-learning-based image analysis techniques through visual experiments.

**Pipeline:** Input → Preprocessing → Algorithm → Visualization → Result → Explanation

## 🚀 Live Application

> **Streamlit App:https://image-analytics-lab-qgbzputfcewxfkw6zcmm7t.streamlit.app/

## 📌 About

Image Analytics Lab is an educational computer vision application that visualizes not only final results, but also important intermediate processing stages.

### Experiments

- 🎯 **Template Matching** — locate a template inside a larger image.
- 👤 **Viola–Jones** — classical Haar-cascade face detection.
- 🧠 **DeepFace** — age, gender, emotion and race/ethnicity predictions.
- 🔗 **FaceNet** — face embeddings and verification using cosine similarity.

## 🎯 Template Matching

```text
Input Image
    ↓
Grayscale Conversion
    ↓
Template Matching
    ↓
Response / Heatmap
    ↓
Best Match Detection
    ↓
Result Visualization
```

Displays the source image, template, grayscale images, matching heatmap, similarity score, location and bounding region.

## 👤 Viola–Jones

Uses Haar-like features, integral-image concepts and a cascade classifier for face detection.

```text
Input Image
    ↓
Grayscale Conversion
    ↓
Haar Cascade
    ↓
Face Detection
    ↓
Bounding Boxes
    ↓
Final Result
```

## 🧠 DeepFace

Performs facial attribute analysis using the DeepFace framework.

- Age estimation
- Gender prediction
- Emotion prediction
- Race/ethnicity prediction

> ⚠️ These are model predictions and should not be treated as definitive real-world facts.

## 🔗 FaceNet

Generates face embeddings and compares two faces using cosine similarity.

```text
Face Image 1 → Face Detection → Face Embedding
                                      │
                                      ↓
                              Cosine Similarity
                                      ↑
                                      │
Face Image 2 → Face Detection → Face Embedding
                                      │
                                      ↓
                              Verification Result
```

## 🏗️ Project Structure

```text
image_analytics_lab/
│
├── app.py
├── algorithms/
│   ├── __init__.py
│   ├── template_matching.py
│   ├── viola_jones.py
│   ├── deepface_analysis.py
│   └── facenet.py
├── utils/
│   ├── __init__.py
│   └── image_utils.py
├── sample_images/
│   └── .gitkeep
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── runtime.txt
├── README.md
└── .gitignore
```

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Streamlit | Interactive web application |
| OpenCV | Image processing and computer vision |
| NumPy | Numerical computation |
| Pandas | Data handling |
| Matplotlib | Visualization |
| Scikit-learn | Machine learning utilities |
| DeepFace | Facial analysis |
| TensorFlow | Deep learning backend |
| Keras-FaceNet | FaceNet implementation |
| Pillow | Image handling |

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/image-analytics-lab.git
cd image-analytics-lab
```

### 2. Create a virtual environment

Python **3.11** is recommended.

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows:**

```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
streamlit run app.py
```

## 📊 Experiment Comparison

| Experiment | Technique | Main Output |
|---|---|---|
| Template Matching | OpenCV Template Matching | Location + similarity |
| Viola–Jones | Haar Cascade | Face detection |
| DeepFace | Deep Learning | Facial attribute predictions |
| FaceNet | Face Embeddings | Face similarity / verification |

## 🎓 Academic Concepts

This project demonstrates:

- Image representation
- Image preprocessing
- Grayscale conversion
- Template matching
- Similarity measurement
- Haar-like features
- Integral images
- Cascade classifiers
- Face detection
- Face embeddings
- Cosine similarity
- Deep learning-based facial analysis
- Image visualization
- Computer vision pipelines

## 🔬 Experimental Workflow

Every experiment follows the same structure:

1. **Input** — upload one or more images.
2. **Preprocessing** — prepare the image for the algorithm.
3. **Algorithm** — execute the selected technique.
4. **Intermediate Visualization** — inspect important processing stages.
5. **Final Result** — view the detection, matching or analysis result.
6. **Explanation** — understand what the result means.

## 🔐 Privacy & Security

This is an educational image-analysis application. Avoid uploading sensitive or personally identifiable images when using a publicly deployed instance.

## ⚠️ Limitations

- Results depend on image quality and input conditions.
- Classical detection methods may fail on difficult images.
- Deep learning predictions are model-dependent.
- Facial attribute predictions may be inaccurate.
- FaceNet thresholds are experimental.
- Performance depends on available CPU/RAM resources.

## 🔮 Future Enhancements

- Video upload and frame-by-frame processing
- Real-time webcam analysis
- Additional image filtering techniques
- Edge detection experiments
- Object detection
- Image segmentation
- Feature extraction visualization
- Additional face recognition models
- Experiment history
- Downloadable analysis reports
- Algorithm performance comparison



## 👩‍💻 Author

**Padma Priya**  
B.Tech – Artificial Intelligence & Data Science

## 📜 License

This project is intended primarily for educational and academic purposes.
