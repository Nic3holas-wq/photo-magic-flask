import os
import cv2
import torch
import base64
import numpy as np
import tempfile
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from PIL import Image
import RRDBNet_arch as arch
from rembg import remove
import logging

app = Flask(__name__)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Helper function to convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# Load the ESRGAN model
model_path = 'RRDB_ESRGAN_x4.pth'
device = torch.device('cpu')  # Change to 'cuda' if using GPU

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
model.eval()
model = model.to(device)

@app.route('/enhance', methods=['POST'])
def enhance_image():
    print("Server running")
    if 'image' not in request.files:
        return jsonify({"error": "No image file found"}), 400

    file = request.files['image']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    filename = secure_filename(file.filename)
    
    # Save temporary input image
    temp_input_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(temp_input_path)
    print("Saved the image at:", temp_input_path)
    # Read and process image
    img = cv2.imread(temp_input_path, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    img = img / 255.0  # Normalize
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0).to(device)

    print("Now Enhancing the image...")
    # Enhance image
    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)
    print("Successfully Enhanced the image, now sending response to frontend.")
    # Convert NumPy array to PIL Image for base64 encoding
    output_pil = Image.fromarray(output)

    # Send image as base64 string for embedding in JSON response
    img_str = image_to_base64(output_pil)
    
    return jsonify({'image': img_str})

@app.route('/remove-background', methods=['POST'])
def remove_background():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        img = Image.open(file)
        
        # Remove the background using rembg
        output = remove(img)
        buffered = BytesIO()
        output.save(buffered, format="PNG")
        
        # Send image as base64 string for embedding in JSON response
        img_str = image_to_base64(output)
        return jsonify({'image': img_str})
    
    except Exception as e:
        logging.error(f"Error in remove_background: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/add-background', methods=['POST'])
def add_background():
    try:
        if 'foreground' not in request.files or 'background' not in request.files:
            return jsonify({'error': 'Foreground or background image not provided'}), 400

        foreground = request.files['foreground']
        background = request.files['background']

        foreground_img = Image.open(foreground).convert("RGBA")
        background_img = Image.open(background).convert("RGBA")

        # Resize background to match the foreground
        try:
            background_img = background_img.resize(foreground_img.size, Image.Resampling.LANCZOS)
        except AttributeError:
            background_img = background_img.resize(foreground_img.size, Image.LANCZOS)

        # Composite the images
        combined_img = Image.alpha_composite(background_img, foreground_img)

        buffered = BytesIO()
        combined_img.save(buffered, format="PNG")
        buffered.seek(0)

        return send_file(buffered, mimetype='image/png', as_attachment=True, download_name='combined_image.png')
    
    except Exception as e:
        logging.error(f"Error in add_background: {e}")
        return jsonify({'error': str(e)}), 500

# Improved logging configuration
logging.basicConfig(level=logging.INFO)

@app.route('/merge-images', methods=['POST'])
def merge_images():
    try:
        if 'background' not in request.files or 'foreground' not in request.files:
            return jsonify({'error': 'Foreground or background image not provided'}), 400

        background_file = request.files['background']
        foreground_file = request.files['foreground']

        background = Image.open(background_file).convert("RGBA")
        foreground = Image.open(foreground_file).convert("RGBA")

        # Resize the foreground to match the background size
        foreground = foreground.resize(background.size, Image.LANCZOS)

        # Merge the images
        background.paste(foreground, (0, 0), foreground)

        # Save the merged image to a BytesIO object
        buffered = BytesIO()
        background.save(buffered, format="PNG")
        buffered.seek(0)

        # Convert image to base64
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({'image': img_str})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cartoonify', methods=['POST'])
def cartoonify():
    print("Cartoonify is working...")
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file found"}), 400

        file = request.files['image']

        if file.filename == '' or not allowed_file(file.filename):
            return jsonify({"error": "Invalid file format"}), 400

        # Read image using OpenCV
        img = Image.open(file).convert("RGB")
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply median blur
        gray_blur = cv2.medianBlur(gray, 5)

        # Detect edges
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, blockSize=9, C=9)

        # Apply bilateral filter to smooth colors
        color = cv2.bilateralFilter(img, d=9, sigmaColor=300, sigmaSpace=300)

        # Combine color image with edges
        cartoon = cv2.bitwise_and(color, color, mask=edges)

        # Convert OpenCV image to PIL Image
        output_pil = Image.fromarray(cv2.cvtColor(cartoon, cv2.COLOR_BGR2RGB))

        # Convert image to base64 and return
        img_str = image_to_base64(output_pil)
        return jsonify({'image': img_str})

    except Exception as e:
        logging.error(f"Error in cartoonify: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', 1000)
