from flask import Flask, request, render_template
from predict import predict
import os

app = Flask(__name__, template_folder='../templates')  # Point to the correct folder
UPLOAD_FOLDER = 'uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file:
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                print(f"File saved at: {file_path}")  # Debug log

                # Get predictions
                result = predict(file_path)

                # If an error occurred during prediction, display the error message
                if 'error' in result:
                    print(f"Prediction error: {result['error']}")  # Debug log
                    return render_template('index.html', result=None, error=result['error'])

                return render_template('index.html', result=result)

    return render_template('index.html', result=None)

if __name__ == '__main__':
    app.run(debug=True)