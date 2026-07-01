# Video Car Counter (HOG + SVM)

Classical computer-vision + machine-learning project that counts cars in videos.

This project uses:
- **Motion detection** (background subtraction) to find moving objects
- **HOG (Histogram of Oriented Gradients)** to describe each candidate object
- **SVM (Support Vector Machine)** to classify candidates as **CAR** vs **NON-CAR**
- **Simple object tracking** to avoid double-counting

The repository is organized around two Jupyter notebooks:
- **Training**: `Car_Classifier_SVM.ipynb`
- **Inference / Counting**: `Video_Car_Counter.ipynb`

---

## Repository Contents

Top-level items you’ll see in the workspace:

- `Car_Classifier_SVM.ipynb` — Builds the dataset (from YOLO labels), extracts HOG features, trains an SVM classifier, evaluates it, and saves artifacts.
- `Video_Car_Counter.ipynb` — Loads the trained model + config and runs a car-counting pipeline over video(s).
- `car_classifier_svm.joblib` — Serialized scikit-learn SVM model (exported from training notebook).
- `car_classifier_config.joblib` — Serialized configuration and metrics needed to reproduce the exact feature extraction at inference time.
- `dataset/`
  - `dataset/car/` — Cropped car images (64×64) used for training.
  - `dataset/non_car/` — Cropped non-car images (64×64) used for training.
- `videos/` — Input videos (e.g. `1.mp4` … `7.mp4`).
- `output/` — Output videos with bounding boxes, labels, and a running car counter.
- `.venv/` — Local virtual environment (optional; depends how you set up Python).

> Note: In the current workspace snapshot, the `Traffic Dataset/` folder exists but is empty. The training notebook expects a YOLO-format dataset under `Traffic Dataset/images/train` and `Traffic Dataset/labels/train`. If you don’t have those folders, you can still train using the existing `dataset/` crops (if present), and you can always run video inference using the existing `.joblib` artifacts.

---

## How the System Works (End-to-End)

### 1) Training Pipeline (`Car_Classifier_SVM.ipynb`)

**Goal:** learn a binary classifier that predicts `car` vs `non-car` given a cropped image patch.

Pipeline stages:

1. **(Optional) Build cropped training set from YOLO labels**
   - Reads label files in YOLO format: `<class_id> <x_center> <y_center> <width> <height>`
   - Converts normalized coordinates to pixel coordinates
   - Crops each labeled object from the original image and resizes it to a fixed size
   - Saves crops into:
     - `dataset/car/` if class is `car`
     - `dataset/non_car/` for all other classes (motorcycle, truck, bus, bicycle, pedestrian)

2. **Dataset sanity checks**
   - Counts images in each class folder
   - Prints class distribution and warns if heavily imbalanced
   - Visualizes sample crops

3. **HOG feature extraction**
   - Converts crop to grayscale
   - Extracts HOG features using fixed parameters (see “Key Parameters” below)

4. **Train/test split**
   - Stratified split (default 80% train / 20% test)

5. **SVM training**
   - Linear-kernel SVM (`sklearn.svm.SVC(kernel='linear', probability=True)`)

6. **Evaluation**
   - Accuracy, precision, recall, F1
   - Confusion matrix and classification report

7. **Model export**
   - Saves the model to `car_classifier_svm.joblib`
   - Saves the inference-critical configuration to `car_classifier_config.joblib`

### 2) Video Car Counting Pipeline (`Video_Car_Counter.ipynb`)

**Goal:** count unique cars in a video by detecting moving objects, classifying them, tracking them, and counting line-crossings.

Pipeline stages:

1. **Load artifacts**
   - Loads `car_classifier_svm.joblib`
   - Loads `car_classifier_config.joblib` and extracts:
     - crop size
     - HOG parameters

2. **Motion-based detection**
   - Background subtraction (`MOG2`)
   - Shadow removal via thresholding
   - Morphology cleanup (close/open/dilate)
   - Contour detection

3. **Candidate filtering**
   - Filters contours by area (`min_area` / `max_area`)
   - Filters by aspect ratio (roughly matching expected vehicle shapes)

4. **Classification**
   - Crops each candidate region from the frame
   - Resizes to the same `CROP_SIZE` used in training
   - Extracts HOG features using the same parameters
   - Predicts car/non-car using the SVM
   - Applies a confidence threshold (when probability estimates are available)

5. **Tracking**
   - Simple Euclidean distance tracker assigns IDs frame-to-frame

6. **Counting**
   - A horizontal counting line is placed at `line_position` fraction of the frame height
   - When a tracked object’s center crosses the line (within tolerance), it increments the counter once per tracked ID

7. **Output**
   - Writes an annotated video to `output/` with bounding boxes and the counter overlay

---

## Key Parameters You Can Tune

### Training (HOG/SVM)

These must remain consistent between training and inference:

- **Crop size**: `CROP_SIZE = (64, 64)`
- **HOG**:
  - `orientations = 9`
  - `pixels_per_cell = (8, 8)`
  - `cells_per_block = (2, 2)`
- **SVM**:
  - `kernel = 'linear'`
  - `C = 1.0`
  - `probability = True` (enables `predict_proba`, used as “confidence” in the video pipeline)

### Video Counting

Main knobs in `count_cars_in_video(...)`:

- `line_position` — where the counting line sits vertically (0–1). Example: `0.6` means 60% down from the top.
- `min_area`, `max_area` — contour area thresholds to ignore tiny noise and huge blobs.
- `confidence_threshold` — minimum probability required to accept a detection as “car”.
- Tracker `max_distance` — larger values handle faster movement but may increase ID switches.

---

## Setup (Windows)

### 1) Create / activate a virtual environment (recommended)

From PowerShell in the project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install numpy opencv-python scikit-image scikit-learn matplotlib seaborn tqdm joblib ipython
```

> If you already have a working environment (e.g., `.venv/` exists), you may only need to install missing packages.

### 3) Open notebooks

Open and run cells in:
- `Car_Classifier_SVM.ipynb`
- `Video_Car_Counter.ipynb`

---

## How to Run

### A) Run Video Counting (recommended quickest path)

You already have trained artifacts in this workspace:
- `car_classifier_svm.joblib`
- `car_classifier_config.joblib`

Steps:

1. Open `Video_Car_Counter.ipynb`
2. Run all cells up to the “Run Car Counting Pipeline” section
3. For a single video, set an input video path.

**Important:** the sample config uses `INPUT_VIDEO = "1.mp4"`, but your videos are located in `videos/`. Use:

- `INPUT_VIDEO = "videos/1.mp4"` (or `videos/2.mp4`, etc.)

4. Run the pipeline cell. Output will be saved under `output/`.

### B) Batch process all 7 videos

The notebook includes a batch section that processes:
- `videos/1.mp4` through `videos/7.mp4`

Outputs:
- `output/car_count_1.mp4` … `output/car_count_7.mp4`

### C) Train / Retrain the Classifier

1. Ensure you have a YOLO-format dataset located at:
   - `Traffic Dataset/images/train`
   - `Traffic Dataset/labels/train`

2. Open `Car_Classifier_SVM.ipynb`
3. Run the notebook from top to bottom.

This will regenerate crops into `dataset/`, retrain the model, and overwrite:
- `car_classifier_svm.joblib`
- `car_classifier_config.joblib`

---

## Output and Results

- Annotated videos are saved to `output/`
- Each output frame can include:
  - green boxes for confident car detections
  - red boxes for rejected detections (optional/visual)
  - a running “CARS: N” overlay
  - the counting line

---

## Common Issues / Troubleshooting

### 1) “Video file not found”

Your input videos are stored in `videos/`. Use `videos/1.mp4` rather than `1.mp4` unless you’ve copied the file into the project root.

### 2) Training notebook cannot find YOLO dataset

If `Traffic Dataset/` is empty or missing required subfolders, the YOLO-to-crops extraction step will fail.

Options:
- Restore the YOLO dataset structure under `Traffic Dataset/images/train` and `Traffic Dataset/labels/train`
- Or skip crop generation and train from the existing `dataset/` crops (if they’re already populated)

### 3) Low accuracy / too many false positives

Try:
- Increasing `confidence_threshold` (e.g., 0.7–0.85)
- Tightening `min_area` and aspect-ratio thresholds
- Improving motion segmentation parameters (MOG2 history/threshold, morphology kernel)
- Adding more diverse negatives in `dataset/non_car/`

### 4) Missed cars / false negatives

Try:
- Lowering `confidence_threshold`
- Loosening `min_area` (but beware noise)
- Adjusting `line_position` and the line-crossing `offset`

### 5) Double counting

Common causes:
- Tracker ID switches (object occlusion, abrupt motion)

Try:
- Increasing tracker `max_distance`
- Using a more robust tracker (SORT/DeepSORT) if you expand the project

---

## Suggested Improvements (Optional)

If you want to take this further:

- Add **feature scaling** (e.g., `StandardScaler`) + a pipeline (`sklearn.pipeline.Pipeline`) for cleaner training
- Add **hard negative mining** (collect false positives as negatives)
- Replace the simple tracker with SORT/DeepSORT for crowded scenes
- Add unit-tested utilities (dataset IO, HOG extraction, model load/save)
- Add a CLI script (e.g., `python run_counter.py --video videos/1.mp4 --out output/out.mp4`)

---

## License / Attribution

This repository contains notebooks and model artifacts created for educational/research purposes.
If you use an external dataset (e.g., a Kaggle YOLO dataset), follow that dataset’s license and attribution requirements.

```
ML project - car counter
├─ car_classifier_config.joblib
├─ Car_Classifier_SVM.ipynb
├─ car_classifier_svm.joblib
├─ dataset
│  ├─ car
│  └─ non_car
│     
├─ output
│  ├─ car_count_1.mp4
│  ├─ car_count_2.mp4
│  ├─ car_count_3.mp4
│  ├─ car_count_4.mp4
│  ├─ car_count_5.mp4
│  ├─ car_count_6.mp4
│  ├─ car_count_7.mp4
│  └─ car_count_output.mp4
├─ README.md
├─ videos
│  ├─ 1.mp4
│  ├─ 2.mp4
│  ├─ 3.mp4
│  ├─ 4.mp4
│  ├─ 5.mp4
│  ├─ 6.mp4
│  └─ 7.mp4
└─ Video_Car_Counter.ipynb

```