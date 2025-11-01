from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import pandas as pd
import numpy as np

# Load disease info only
disease_info = pd.read_csv('Model_assest/disease_info.csv')

# device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create instance of the pretrained model
model = models.resnet18(pretrained=True)

# Modify last layer
num_classes = 38
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load trained weights
model_checkpoint_path = 'Model_assest/model.pth'
model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Prediction function
def prediction(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    index = np.argmax(output.numpy())
    return index

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')




@app.route('/submit', methods=['POST'])
def submit():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'Empty file'})

    file_path = os.path.join('static/uploads', image.filename)
    image.save(file_path)

    pred = prediction(file_path)
    title = disease_info['disease_name'][pred]
    description = disease_info['description'][pred]
    possible_steps = disease_info['Possible Steps'][pred]

    data = {
        'prediction': title,
        'image': '/' + file_path,
        'description': description,
        'possible_step': possible_steps
    }

    return render_template('submit.html', data=data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
