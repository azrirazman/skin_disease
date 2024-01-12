import joblib
import numpy as np
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from PIL import Image, ImageTk
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap

class SkinDiseasePredictor(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Skin Disease Prediction System')
        self.setGeometry(100, 100, 500, 400)

        self.image_label = QLabel()
        self.image_label.setFixedSize(300, 300)

        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.predict_skin_disease)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.predict_button)

        self.setLayout(layout)

    def predict_skin_disease(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image files (*.jpg *.jpeg *.png *.bmp)")
        file_dialog.selectNameFilter("Image files (*.jpg *.jpeg *.png *.bmp)")
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            if len(selected_files) > 0:
                img_path = selected_files[0]
                predictions = self.skin_disease_prediction(img_path)
                print("Skin Disease Predictions:", predictions)

    def skin_disease_prediction(self, img_path):
        size = (299, 299)  # Input size expected by InceptionV3
        img = image.load_img(img_path, target_size=size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Load the InceptionV3 model
        inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

        # Extract features
        img_features = inception_v3.predict(img_array)

        # Flatten the features
        img_flattened = img_features.reshape((img_features.shape[0], 8 * 8 * 2048))

        # Load the skin disease prediction model (assumed to be in .pkl format)
        skin_disease_model = joblib.load('D:/skin_disease/dataset/unmarked_resize/results/InceptionV3+kNN_Model_5C.pkl')

        # Make predictions
        predictions = skin_disease_model.predict(img_flattened)

        # Perform any post-processing on predictions as needed
        # For example, you might want to map predicted indices to class names

        return predictions


if __name__ == '__main__':
    app = QApplication([])
    predictor = SkinDiseasePredictor()
    predictor.show()
    app.exec_()