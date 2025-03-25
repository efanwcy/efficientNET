from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import os
import requests
import torch
from torchvision import transforms
from model import efficientnet_b0 as create_model

app = Flask(__name__)
CORS(app)  # Enable CORS, allowing all domains to access

# Load YOLO models
clean_model = YOLO('clean_contamination.pt')  # Clean and contaminated wound model
depth_model = YOLO('Depth_shallowness.pt')   # Deep and shallow wound model

# Load EfficientNet models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
super_model_path = "super.pth"
clean_model_path = "clean.pth"
num_classes = 2  # Assuming binary classification for both models

def load_efficientnet_model(model_path, num_classes, device):
    """
    Load a trained EfficientNet model.
    """
    model = create_model(num_classes=num_classes).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    model.eval()
    return model

super_model = load_efficientnet_model(super_model_path, num_classes, device)
clean_model_efficientnet = load_efficientnet_model(clean_model_path, num_classes, device)

# Image preprocessing for EfficientNet
img_size = 224
data_transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.CenterCrop(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Temporary and output directories
temp_images_dir = 'temp_images'
out_images_dir = 'out_images'
os.makedirs(temp_images_dir, exist_ok=True)
os.makedirs(out_images_dir, exist_ok=True)

def clear_directory(directory):
    """Clear all files in a directory."""
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def check_and_clear_directories():
    """Check and clear directories if they contain too many files."""
    if len(os.listdir(temp_images_dir)) > 50:
        clear_directory(temp_images_dir)
        print(f"Cleared {temp_images_dir}")
    if len(os.listdir(out_images_dir)) > 50:
        clear_directory(out_images_dir)
        print(f"Cleared {out_images_dir}")

def preprocess_image(image_path, transform):
    """Preprocess an image for EfficientNet."""
    img = Image.open(image_path)
    img = transform(img)
    img = torch.unsqueeze(img, dim=0)  # Add batch dimension
    return img

def predict_efficientnet(model, image_tensor, device, model_type="super"):
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = torch.softmax(output, dim=1)  # Convert logits to probabilities
        _, predicted = torch.max(probabilities, 1)  # Get predicted class index

        # Define labels based on model type
        if model_type == "super":
            labels = ["Non-superficial surface wounds", "superficial surface wound"]  # 调整顺序
        elif model_type == "clean":
            labels = ["clean_wound", "contamination_wound"]
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        predicted_label = labels[predicted.item()]
        predicted_prob = probabilities[0][predicted.item()].item()  # Probability of the predicted class
    return predicted_label, predicted_prob

@app.route('/yolo', methods=['POST', 'OPTIONS'])
def yolo_inference():
    if request.method == 'OPTIONS':
        # Return CORS headers
        response = jsonify()
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    if request.content_type != 'application/json':
        return jsonify({"error": "Content-Type must be application/json"}), 400

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body must be valid JSON"}), 400
    except Exception as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

    if 'image_path' not in data:
        return jsonify({"error": "image_path is required"}), 400

    image_url = data['image_path']

    try:
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()

        # Generate local file path
        image_name = os.path.basename(image_url)
        local_image_path = os.path.join(temp_images_dir, image_name)

        # Save the image locally
        with open(local_image_path, 'wb') as f:
            f.write(response.content)

        # Open the image
        image = Image.open(local_image_path)

        # Run inference with the clean_contamination model
        clean_results = clean_model(image)
        if not isinstance(clean_results, list) or len(clean_results) == 0:
            return jsonify({"error": "No results found from clean_contamination model"}), 400

        # Run inference with the Depth_shallowness model
        depth_results = depth_model(image)
        if not isinstance(depth_results, list) or len(depth_results) == 0:
            return jsonify({"error": "No results found from Depth_shallowness model"}), 400

        # Save the output images with bounding boxes
        clean_output_image_path = os.path.join(out_images_dir, f"clean_{image_name}")
        depth_output_image_path = os.path.join(out_images_dir, f"depth_{image_name}")

        clean_results[0].save(filename=clean_output_image_path)  # Save clean model output
        depth_results[0].save(filename=depth_output_image_path)  # Save depth model output

        # Get detected classes from clean_contamination model
        clean_result = clean_results[0]
        clean_detected_classes = []
        for box in clean_result.boxes:
            class_id = int(box.cls)
            class_name = clean_model.names[class_id]  # Get class name
            clean_detected_classes.append(class_name)

        # Get detected classes from Depth_shallowness model
        depth_result = depth_results[0]
        depth_detected_classes = []
        for box in depth_result.boxes:
            class_id = int(box.cls)
            class_name = depth_model.names[class_id]
            depth_detected_classes.append(class_name)

        # Preprocess image for EfficientNet models
        image_tensor = preprocess_image(local_image_path, data_transform)

        # Predict using super.pth model
        super_prediction, super_prob = predict_efficientnet(super_model, image_tensor, device, model_type="super")

        # Predict using clean.pth model
        clean_prediction_efficientnet, clean_prob = predict_efficientnet(clean_model_efficientnet, image_tensor, device, model_type="clean")

        # Return the results
        response = jsonify({
            "status": "success",
            "clean_detected_classes": clean_detected_classes,
            "depth_detected_classes": depth_detected_classes,
            "super_model_prediction": super_prediction,
            "super_model_prob": super_prob,
            "clean_model_prediction": clean_prediction_efficientnet,
            "clean_model_prob": clean_prob,
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to download image: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3080)