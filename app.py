from flask import Flask, request, render_template, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('modelo_mnist.h5')  # Asegúrate de que el modelo esté cargado correctamente

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        f = request.files['file']
        filename = f.filename
        if not os.path.exists('static'):
            os.makedirs('static')
        filepath = os.path.join('static', filename)
        f.save(filepath)
        digits, predictions, roi_path = extract_digits_and_predict(filepath, model, filename)
        results = list(zip(digits, predictions))  # Prepara los datos para el template
        return render_template('index.html', uploaded_image=filename, results=results, roi_image=roi_path)
    return render_template('index.html')

def extract_digits_and_predict(filepath, model, filename):
    image = cv2.imread(filepath)
    if image is None:
        print("Failed to load image")
        return [], [], None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted_gray = cv2.bitwise_not(gray)

    x_start, y_start, x_end, y_end = int(image.shape[1] * 0.76), int(image.shape[0] * 0.185), int(image.shape[1] * 0.955), int(image.shape[0] * 0.32)
    roi = inverted_gray[y_start:y_end, x_start:x_end]
    roi_path = f'roi_{filename}'
    cv2.imwrite(os.path.join('static', roi_path), roi)

    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    digits_paths = []
    predictions_list = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w < 10 or h < 10:
            continue
        digit = roi[y:y+h, x:x+w]
        digit_path = f'digit_{i}_{filename}'
        cv2.imwrite(os.path.join('static', digit_path), cv2.bitwise_not(digit))
        digits_paths.append(digit_path)

        digit_img_resized = cv2.resize(digit, (28, 28), interpolation=cv2.INTER_AREA)
        digit_img_normalized = digit_img_resized.astype('float32') / 255
        X = np.array([digit_img_normalized])
        predicted_digit = model.predict(X)
        predictions_list.append(np.argmax(predicted_digit))

    return digits_paths, predictions_list, roi_path

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
