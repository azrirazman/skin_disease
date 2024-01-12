import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QTextEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image
import joblib  # Make sure to use 'joblib' instead of 'sklearn.externals.joblib'
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image as image_utils
from sklearn.preprocessing import StandardScaler



class SkinDiseaseClassifier(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Skin Disease Classifier')
        self.setGeometry(100, 100, 800, 600)

        self.current_index = 0
        self.image_paths = []

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_textbox = QTextEdit(self)
        self.result_textbox.setReadOnly(True)
        self.result_textbox.setMinimumHeight(100)

        self.prev_button = QPushButton('Previous', self)
        self.prev_button.clicked.connect(self.show_previous)

        self.next_button = QPushButton('Next', self)
        self.next_button.clicked.connect(self.show_next)

        self.load_button = QPushButton('Load Images', self)
        self.load_button.clicked.connect(self.load_images)

        self.predict_button = QPushButton('Predict', self)
        self.predict_button.clicked.connect(self.predict_skin_disease)

        # Layout
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.result_textbox)

        hbox = QHBoxLayout()
        hbox.addWidget(self.prev_button)
        hbox.addWidget(self.next_button)
        hbox.addWidget(self.load_button)
        hbox.addWidget(self.predict_button)

        vbox.addLayout(hbox)
        self.setLayout(vbox)

        # Load your skin disease classification model (replace with your model path)
        self.inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        self.kNN_model = joblib.load('D:/skin_disease/models/InceptionV3+kNN_Model_2.pkl')  # Replace with your kNN model path

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image()

    def show_next(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_image()

    def load_images(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image files (*.jpg *.jpeg *.png *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec_():
            self.image_paths = file_dialog.selectedFiles()
            self.current_index = 0
            self.update_image()

    def update_image(self):
        if not self.image_paths:
            return
        self.result_textbox.clear() 
        image_path = self.image_paths[self.current_index]
        pixmap = QPixmap(image_path)
        self.image_label.setPixmap(pixmap.scaledToWidth(400))

    def predict_skin_disease(self):
        if not self.image_paths:
            return

        image_path = self.image_paths[self.current_index]
        img = Image.open(image_path).resize((299, 299), Image.LANCZOS)
        img_array = image_utils.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        features = self.inception_v3.predict(img_array)
        flattened_features = features.reshape((features.shape[0], -1))

        # Optionally, you can perform standard scaling on the features
        scaler = StandardScaler()
        flattened_features = scaler.fit_transform(flattened_features)

        predicted_category = self.kNN_model.predict(flattened_features)

        # Display the predicted category in the result text box
        category_mapping = {
            0: 'Acne',
            1: 'Melanoma',
            2: 'Psoriasis',
            3: 'Ringworm',
            4: 'Scabies'
        }

        category_label = category_mapping.get(predicted_category[0], 'Unknown Category')
        self.result_textbox.clear()
        self.result_textbox.append(f'Predicted Category: {category_label}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SkinDiseaseClassifier()
    ex.show()
    sys.exit(app.exec_())