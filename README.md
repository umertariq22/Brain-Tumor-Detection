
# 🧠 Brain Tumor Classification

This project focuses on the **automated classification of brain tumors** using MRI scan images. The goal is to develop a deep learning model that accurately classifies images into various tumor types or identifies the absence of a tumor. It aims to support radiologists and medical professionals in early diagnosis and treatment planning.

## 📁 Project Structure

```
brain-tumor-classification/
│
├── brain-tumor-classification.ipynb   # Jupyter notebook with complete code
├── README.md                          # Project overview and usage
```

## 📌 Features

- Image preprocessing and augmentation
- Convolutional Neural Network (CNN) architecture
- Training with validation and test split
- Performance metrics (accuracy, confusion matrix, etc.)
- Visualization of predictions

## 🧪 Technologies Used

- Python 3.x
- TensorFlow / Keras
- OpenCV
- Matplotlib / Seaborn
- NumPy / Pandas
- Scikit-learn

## 📊 Dataset

The dataset contains MRI images categorized into:
- **Glioma Tumor**
- **Meningioma Tumor**
- **Pituitary Tumor**
- **No Tumor**

> **Note:** The dataset is not included in this repository. You can download a publicly available dataset such as the one from [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri).

## 🚀 How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/brain-tumor-classification.git
   cd brain-tumor-classification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and place the dataset** inside a folder named `data/`.

4. **Run the notebook:**
   Open `brain-tumor-classification.ipynb` in Jupyter Notebook or Google Colab and execute the cells.

## ✅ Results

- **Training Accuracy:** ~99%
- **Validation Accuracy:** ~95-98%
- **Model:** CNN with data augmentation
- Confusion matrix and class-wise accuracy indicate robust performance across tumor types.

## 📈 Sample Visualization

Example MRI image predictions with labels plotted after training:
- ✅ Glioma
- ✅ Meningioma
- ✅ Pituitary
- ✅ No Tumor

## 📌 Future Improvements

- Add transfer learning using pre-trained models like VGG16, ResNet50.
- Deploy the model using Flask or Streamlit.
- Extend to 3D MRI data using volumetric CNNs.

## 🙌 Acknowledgements

- Dataset by Sartaj Bhuvaji on Kaggle.
