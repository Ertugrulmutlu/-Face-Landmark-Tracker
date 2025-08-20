# 🎯 Face Landmarks Detection (OpenCV DNN + Facemark)

This project detects faces in a video, extracts **68 facial landmarks**, and saves them into:

* an **annotated output video**
* an optional **CSV file** containing all landmarks per frame

Built with **OpenCV (DNN + Facemark LBF)** and includes smoothing for stable tracking.

---

## ⚡ Features

* Face detection with **OpenCV DNN** (ResNet SSD Caffe model)
* 68-point landmark detection with **Facemark LBF**
* Optional CSV export (`x0..x67, y0..y67`)
* **EMA smoothing filter** to stabilize jitter
* Configurable input/output paths & frame skipping

---

## 🚀 Setup

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/face-landmarks-detection.git
   cd face-landmarks-detection
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   or manually:

   ```bash
   pip install opencv-contrib-python numpy
   ```

3. Download the required model files:

   * **Facial landmark model (LBF):**
     [lbfmodel.yaml](https://github.com/kurnianggoro/GSOC2017/blob/master/data/lbfmodel.yaml)

   * **Face detector weights:**
     [res10\_300x300\_ssd\_iter\_140000\_fp16.caffemodel](https://github.com/mostofashakib/Image-Analysis-and-Real-Time-Face-Recognition-system/blob/master/res10_300x300_ssd_iter_140000_fp16.caffemodel)

   * **Face detector config:**
     [deploy.prototxt](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)

   Save them under a `model/` directory:

   ```
   model/
   ├── lbfmodel.yaml
   ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
   └── deploy.prototxt
   ```

---

## ▶️ Usage

Put your input video inside a `video/` folder, e.g.:

```
video/example.mp4
```

Run the script:

```bash
python main.py
```

Outputs:

* Annotated video → `./output/annotated.mp4`
* Landmark CSV (if enabled) → `./output/landmarks.csv`

---

## 📂 Project Structure

```
.
├── main.py
├── Utils/
│   ├── detector.py
│   ├── landmarks.py
│   └── helpers.py
├── model/
│   ├── deploy.prototxt
│   ├── res10_300x300_ssd_iter_140000_fp16.caffemodel
│   └── lbfmodel.yaml
├── video/
│   └── example.mp4
└── output/
    ├── annotated.mp4
    └── landmarks.csv
```

---

## 📝 Notes

* Works best with **frontal faces**.
* Adjust `SKIP` parameter in `main.py` to balance performance vs. accuracy.
* EMA smoothing is on by default (`alpha = 0.7`).

---

## 📜 License

MIT License

---

## 🔗 Further Reading

A full technical breakdown is available in the accompanying blog post: \[BLOG\_URL\_PLACEHOLDER]
