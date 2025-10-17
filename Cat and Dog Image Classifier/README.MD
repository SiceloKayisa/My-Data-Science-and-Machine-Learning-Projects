# 🐾 Cat vs. Dog Image Classifier

[![FreeCodeCamp Certification Project](https://img.shields.io/badge/Certification-FreeCodeCamp-blue)](https://www.freecodecamp.org/)

## 📖 Project Overview

This repository contains a **Convolutional Neural Network (CNN)** model designed to perform **binary image classification**, distinguishing between images of cats and dogs.

This project was completed as part of the **FreeCodeCamp Machine Learning Certification**.

---

## ⭐ Key Achievements (Quantifiable Metrics)

* **Classification Performance:** Achieved a final test accuracy of **74.0%** on the specialized 50-image test set, meeting the challenge passing requirements.
* **Dataset Utilization:** Model trained and validated on a total of **3,000 images** (2,000 for training, 1,000 for validation).
* **Architecture & Training:**
    * Trained the model over **5 epochs** using a `BATCH_SIZE` of 100.
    * Employed a deep architecture featuring **3 Conv2D blocks** and a **Dropout rate of 0.2** for robust regularization.
* **Input Standardization:** Successfully implemented preprocessing to resize all images to a consistent **150x150** input dimension and **normalized** pixel values to the $0-1$ range.

---

## 🎯 Project Aims

The primary goals of this project were:

1.  **Build a Robust Classifier:** Create a deep learning model capable of accurately classifying images of cats and dogs.
2.  **Master CNN Fundamentals:** Gain practical experience in designing and implementing CNN architectures, including the use of convolutional, pooling, and dense layers.
3.  **Implement Best Practices:** Utilize appropriate techniques for image preprocessing, data loading, and model compilation (e.g., using `SparseCategoricalCrossentropy` with logits).

---

## 💻 Technologies and Libraries

* **Language:** Python
* **Framework:** TensorFlow 2.x / Keras
* **Libraries:**
    * `numpy` (for numerical operations)
    * `matplotlib` (for visualization)
    * `os` (for file path management)

---

## 💾 Dataset

The model was trained on a subset of the popular Kaggle Dogs vs. Cats dataset.

* **Total Images:** 3,050 (including the 50-image test set)
* **Training Set:** 2,000 images
* **Validation Set:** 1,000 images
* **Input Shape:** $150 \times 150 \times 3$ (Height $\times$ Width $\times$ Color Channels)

---

## 🚀 Getting Started

The core code is contained in the Jupyter Notebook: `Akshar_fcc_cat_dog.ipynb`.

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository URL]
    ```
2.  **Install Dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib
    ```
3.  **Run the Notebook:** Open the `Cat and Dog Image Classifier.ipynb` file in a Jupyter environment (such as Google Colab or Jupyter Notebook) and execute the cells sequentially. The notebook handles the data download automatically.
