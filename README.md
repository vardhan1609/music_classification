# 🎵 Music Classification Project

This project focuses on **classifying music/audio files** using deep learning techniques.  
It extracts audio features and predicts the music category/genre based on trained models.

---

## 📌 Features
- Audio feature extraction
- Music / genre classification
- Deep learning based prediction
- Simple and reusable code structure

---

## 🛠️ Technologies Used
- Python
- NumPy
- Pandas
- Librosa
- Scikit-learn
- Matplotlib (for visualization)
- streamlit

## 📊 Model Performance

The deep learning model was trained for 100 epochs and evaluated on a separate test dataset. The training and validation metrics indicate strong learning and good generalization.

🔹 Final Training Results

Training Accuracy: 95.97%

Training Loss: 0.1731

🔹 Validation Results

Validation Accuracy: 88.61%

Validation Loss: 0.3781

🔹 Test Results

Test Accuracy: 90.38%

Test Loss: 0.3003

## 📈 Observations

The model achieved over 96% training accuracy, showing effective feature learning.

Validation accuracy remained close to training accuracy, indicating limited overfitting.

A test accuracy of 90.38% confirms that the model generalizes well to unseen audio samples.

Minor fluctuations in validation accuracy near the final epochs suggest the model is close to optimal convergence.

##  ✅ Conclusion

The trained model demonstrates strong and reliable performance for audio classification tasks. With over 90% test accuracy, it is suitable for real-world deployment and can be further improved using techniques such as data augmentation, regularization, or hyperparameter tuning.
