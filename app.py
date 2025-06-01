from flask import Flask, request, jsonify, render_template, send_file
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import os
from crack_detection import analyze_cracks
import base64

def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj

# Initialize Flask app with static folder configuration
app = Flask(__name__, 
            static_url_path='/static',
            static_folder='static',
            template_folder='templates')

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = '/tmp/uploads'  # Use /tmp for Vercel

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Read image file
        filestr = file.read()
        npimg = np.frombuffer(filestr, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Process image
        annotated_image, results = analyze_cracks(image)
        
        if annotated_image is None:
            return jsonify({'error': 'Failed to process image'}), 500

        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Convert results to JSON serializable format
        serializable_results = convert_to_serializable(results)
        
        # Return results
        return jsonify({
            'image': f'data:image/jpeg;base64,{img_str}',
            'results': serializable_results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add health check endpoint
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy'}), 200

app = app.wsgi_app  # For Vercel deployment

if __name__ == '__main__':
    app.run(debug=True) 