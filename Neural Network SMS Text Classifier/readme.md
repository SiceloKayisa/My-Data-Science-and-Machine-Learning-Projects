# SMS Text Classification (Ham/Spam)

This project develops a machine learning model to classify SMS messages as either "ham" (legitimate) or "spam" (unsolicited or malicious).

---

## 🎯 Project Aims

The primary goal of this project was to build and train a deep learning model to accurately distinguish between legitimate and spam SMS messages using the **SMS Spam Collection dataset**.

Key objectives include:

* **Model Creation:** Develop a robust machine learning model (utilizing **TensorFlow/Keras** and text preprocessing with **NLTK**) capable of natural language classification.
* **Message Prediction:** Implement a function, `predict_message`, that takes a new SMS message as input and outputs a prediction indicating the probability of the message being spam, along with the final classification ("ham" or "spam").

---

## 📈 Key Quantifiable Achievements

The developed model, built using a **Sequential Keras model** with a **TextVectorization layer** for preprocessing, achieved high performance metrics on both the validation and held-out test datasets.

| Metric | Training Result (Final Epoch) | Validation Result (Final Epoch) | Test Set Evaluation |
| :--- | :--- | :--- | :--- |
| **Accuracy** | $\approx 99.97\%$ | $\approx 98.56\%$ | $\approx 98.68\%$ |
| **Loss (Binary Cross-entropy)** | $\approx 0.0031$ | $\approx 0.0528$ | $\approx 0.0384$ |

The model successfully passed all seven final challenge test cases, demonstrating its effectiveness in classifying diverse examples of ham and spam messages.

---

## 🛠 Technology Stack

* **Python**
* **TensorFlow/Keras:** For building and training the neural network model.
* **Pandas & NumPy:** For data loading and manipulation.
* **NLTK (Natural Language Toolkit):** Used for text preprocessing steps like stop-word removal and lemmatization.
