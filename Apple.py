from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os, uuid, json
import numpy as np
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras.models import Model

UPLOAD_FOLDER = 'cats'
EMBEDDINGS_FILE = 'embeddings.json'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)

embeddings, labels = [], []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB').resize((224, 224))
    x = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
    x = preprocess_input(x)
    return model.predict(x)[0].tolist()

def save_embeddings():
    with open(EMBEDDINGS_FILE, 'w') as f:
        json.dump({'embeddings': embeddings, 'labels': labels}, f)

def load_embeddings():
    global embeddings, labels
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, 'r') as f:
            data = json.load(f)
            embeddings = data['embeddings']
            labels = data['labels']

@app.route('/add-cat', methods=['POST'])
def add_cat():
    file = request.files['image']
    name = request.form['name']
    if file and allowed_file(file.filename):
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        path = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()) + "_" + secure_filename(file.filename))
        file.save(path)
        embeddings.append(get_embedding(path))
        labels.append(name)
        save_embeddings()
        return jsonify({'status': 'success', 'cat_name': name}), 200
    return jsonify({'status': 'error'}), 400

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file and allowed_file(file.filename):
        path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
        file.save(path)
        if not embeddings:
            return jsonify({'result': 'No cats trained yet'}), 200
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(np.array(embeddings), labels)
        pred = knn.predict([get_embedding(path)])[0]
        return jsonify({'result': pred}), 200
    return jsonify({'status': 'error'}), 400

@app.route('/cats', methods=['GET'])
def list_cats():
    return jsonify({'known_cats': list(set(labels))})

load_embeddings()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
