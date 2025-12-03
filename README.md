# Sudoku Solver - Automatic Resolution with Local Model

This project uses **OpenCV** and **deep learning** to automatically detect, recognize, and solve Sudoku grids from images or in real-time via webcam.

The digit recognition model was **created from scratch**: from training data generation to training the convolutional neural network.

## 🎥 Quick Demo

Want to see the results right away? Check out the **[sudoku_image.ipynb](sudoku-solver/sudoku_image.ipynb)** notebook which visually displays each step of the resolution pipeline (contour detection, grid extraction, digit recognition, and solution display).

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

**Clone the repository and install dependencies:**

```bash
git clone https://github.com/amaryassa/sudoku.git
cd sudoku
pip install -r requirements.txt
```

## 🚀 Usage

### Resolution via webcam (real-time)

```bash
cd sudoku-solver
python sudoku_video.py
```

Place a Sudoku grid in front of your webcam. The solution displays in real-time on the image. Press `q` to quit.

### Resolution from an image

```bash
cd sudoku-solver
jupyter notebook sudoku_image.ipynb
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
cd digit-training
jupyter notebook generate_digit.ipynb
```

This will create ~40,000 digit images with different fonts and variations.

**Step 2 - Train the CNN:**

```bash
jupyter notebook digit_cnn_trainning.ipynb
```

The model achieves ~95% accuracy after training.

**Step 3 - Deploy the new model:**

```bash
cp model_trained.keras ../sudoku-solver/model/
```

## 🔮 Future Improvements

- Support for handwritten grids
- Graphical user interface (GUI)
- Mobile application
- Improved robustness to different lighting conditions
- PDF solution export
