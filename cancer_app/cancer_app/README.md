# ViT Cancer Detection System — Localhost Setup

## Files
```
cancer_app/
├── app.py              ← Flask backend
├── requirements.txt    ← Python dependencies
└── templates/
    └── index.html      ← Full UI
```

## Setup & Run

### Step 1 — Install dependencies
```bash
cd cancer_app
pip install -r requirements.txt
```

### Step 2 — Start the server
```bash
python app.py
```

### Step 3 — Open browser
```
http://localhost:5000
```

---

## Features
- **Tab 1 · Single Image**: Upload a histopathology image → get predicted class + confidence bars.  
  If **MALIGNANT** → cancer cell regions are automatically circled on the image using H&E staining density analysis.
- **Tab 2 · Bulk Upload**: Upload many images at once → batch results with per-image verdicts.
- **Tab 3 · Model Comparison**: ViT-B/16 vs ViT-B/32 stats, per-class accuracy, and confusion matrices from the notebook.
- **Download Reports**: All 3 tabs have a downloadable `.txt` report.

## Connect Real Model (Optional)
Replace `simulate_prediction()` in `app.py` with actual TensorFlow inference:

```python
import tensorflow as tf
model = tf.keras.models.load_model('your_vit_model.h5')

def real_predict(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).resize((224, 224))
    arr = np.array(img) / 255.0
    arr = arr.reshape((1, 224, 224, 3))
    probs = model.predict(arr)[0]
    top_idx = np.argmax(probs)
    return probs.tolist(), top_idx, CLASSES[top_idx]
```

## Classes (from notebook)
Blood-Benign, Blood-Malignant, Breast-Benign, Breast-Malignant,
Colon-Benign, Colon-Malignant, Lung-Benign, Lung-Malignant
