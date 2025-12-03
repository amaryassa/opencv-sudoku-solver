# Sudoku Solver - Automatic Resolution with Local Model

## 🎥 Quick Demo

### 📹 Video Demonstration

Watch the full demonstration of the Sudoku solver in action:

<p align="center">
  <img src="demo.gif" alt="Sudoku Solver Demo">
</p>

> **Note**: If the demo doesn't display above, you can view it directly here : [demo.gif](https://github.com/amaryassa/opencv-sudoku-solver/blob/main/demo.gif)

### 🔍 Try It Yourself

Want to see how it works? Choose your preferred method:

| Method                      | Description                                                                         | Command                                                     |
| --------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 📓 **Interactive Notebook** | Visual step-by-step walkthrough with detailed explanations of each processing stage | [Open sudoku_image.ipynb](sudoku-solver/sudoku_image.ipynb) |
| 🎥 **Live Webcam**          | Real-time Sudoku detection and solving using your webcam                            | `cd sudoku-solver && python sudoku_video.py`                |

#### 📓 Notebook Demo Features:

- 🖼️ Visual output at each pipeline step
- 🔍 Grid detection and contour analysis
- 🧠 Digit recognition predictions
- ✅ Solution overlay on original image

#### 🎥 Webcam Demo Features:

- ⚡ Real-time processing (<100ms per frame)
- 🎯 Automatic grid detection
- 💚 Green overlay for solved digits

## 📂 Project Structure

The project is organized into two main modules:

### 🎯 **sudoku-solver/** - Main Application

This is the application that uses the trained model to solve Sudokus:

- **`sudoku_video.py`** : Real-time resolution via webcam
- **`sudoku_image.ipynb`** : Resolution from static images (see detailed process overview)
- **`model/model_trained.keras`** : Pre-trained CNN model for digit recognition
- **`libs/sudukoSolver.py`** : Backtracking resolution algorithm
- **`utils/helpers.py`** : Utility functions for image processing (grid detection, cell extraction, etc.)
- **`images/`** : Test images

**Processing Pipeline**:

1. Grid detection in the image (OpenCV contours)
2. Perspective transformation to straighten the grid
3. Extraction of the 81 individual cells
4. Digit recognition with the CNN model
5. Sudoku resolution with backtracking algorithm
6. Solution overlay on the original image

### 🔢 **digit-training/** - Recognition Model Training (optional)

**If you want to recreate the model yourself**, this folder contains the entire training pipeline. Otherwise, you don't need it to run the sudoku solver - the pre-trained model is already included.

This module includes:

- **`generate_digit.ipynb`** : Automatically generates over 40,000 digit images (0-9) with different fonts and variations (rotation, noise, size)
- **`digit_cnn_trainning.ipynb`** : Trains a convolutional neural network (CNN) on the generated images
- **`digit_cnn_test.ipynb`** : Tests and evaluates the model's performance
- **`model_trained.keras`** : Trained model (to copy to `sudoku-solver/model/`)
- **`data/`** : Generated training images
- **`fonts/`** : Fonts used to generate digits

## 🎥 Result Preview

To see a detailed overview of the step-by-step resolution process (contour detection, grid extraction, digit recognition and solution display), check out the **[sudoku_image.ipynb](sudoku-solver/sudoku_image.ipynb)** notebook which visually displays each step of the pipeline.

## 📦 Installation

Prerequisite: install `uv` (recommended) to manage the virtual environment and dependencies.

> macOS:

- Homebrew: `brew install uv`
- Or via pipx: `pipx install uv`

> If you don't use `uv`, you can still run with a standard Python venv and `pip`, but the commands below assume `uv`.

**Clone the repository and install dependencies:**

```bash
git clone https://github.com/amaryassa/opencv-sudoku-solver.git
cd opencv-sudoku-solver
uv sync
uv python pin
source .venv/bin/activate
```

## 🚀 Usage

### Resolution via webcam (real-time)

```bash
uv run python sudoku-solver/sudoku_video.py
```

Place a Sudoku grid in front of your webcam. The solution displays in real-time on the image. Press `q` to quit.

### Resolution from an image

```bash
uv run jupyter notebook sudoku-solver/sudoku_image.ipynb
```

Modify the image path in the notebook (default: `images/1.png`) and execute the cells to see the step-by-step resolution with visualizations.

## 🛠️ Technologies Used

- **Python 3.8+**
- **OpenCV** - Computer vision and image processing
- **TensorFlow/Keras** - Deep learning (CNN)
- **NumPy** - Numerical computing
- **Matplotlib** - Visualization
- **Pillow** - Image generation
- **Jupyter** - Interactive notebooks

## 🎓 Retrain the Model (optional)

**To recreate the recognition model from scratch:**

**Step 1 - Generate training data:**

```bash
uv run jupyter notebook digit-training/generate_digit.ipynb
```

This will create ~40,000 digit images with different fonts and variations.

**Step 2 - Train the CNN:**

```bash
uv run jupyter notebook digit-training/digit_cnn_trainning.ipynb
```

The model achieves ~95% accuracy after training.

**Step 3 - Deploy the new model:**

```bash
cp model_trained.keras ../sudoku-solver/model/
```

## 📚 Resources and Tutorials Followed

- [Formation Deep Learning Complète (2021) — YouTube playlist](https://www.youtube.com/watch?v=XUFLq6dKQok&list=PLO_fdPEVlfKoanjvTJbIbd9V5d9Pzp8Rw)
- [OpenCV Sudoku Solver Step by Step — YouTube](https://www.youtube.com/watch?v=qOXDoYUgNlU)
- [Tuto#18 - Sudoku: ne perdez pas le nord p.1 — YouTube](https://www.youtube.com/watch?v=WwPHs1SJrec)
- [Tuto#18 - Sudoku: ne perdez pas le nord p.2 — YouTube](https://www.youtube.com/watch?v=XFNg8lXe-Tk)
- [StackOverflow — How to get the cells of a Sudoku grid with OpenCV](https://stackoverflow.com/questions/59182827/how-to-get-the-cells-of-a-sudoku-grid-with-opencv)
