from flask import Flask, request, jsonify
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging
from flask import Flask, request, jsonify
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Function to detect device type Check if CUDA or MPS (Metal Performance Shaders for Apple Silicon) is available,  if not then use cpu
def get_device():
    """
    Detect and return the available device for computation (GPU/CPU).
    Returns a tuple of (device, number of GPUs).
    """
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps"), 0
        elif torch.cuda.is_available():
            return torch.device("cuda"), torch.cuda.device_count()
        else:
            return torch.device("cpu"), 0
    except Exception as e:
        logging.error(f"Error in get_device: {e}")
        raise

device, n_gpu = get_device()
logging.info(f"Present device: {device}")
logging.info(f"GPU count: {n_gpu}")

## Set seed for reproducibility in PyTorch, when using a GPU or not. 
class Args:
    def __init__(self, seed, n_gpu):
        self.seed = seed
        self.n_gpu = n_gpu

def set_seed(args):
    """
    Set seed for reproducibility in PyTorch, both for CPU and GPU.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# Initialize Args with the seed value and number of GPUs
args = Args(seed=42, n_gpu=n_gpu)
set_seed(args)

# Load the SentenceTransformer model
# model_path = 'sts_model'
model_path = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name_or_path=model_path)
try:
    device, n_gpu = get_device()
    args = Args(seed=42, n_gpu=n_gpu)
    set_seed(args)
    model_path = 'sts_model'
    model = SentenceTransformer(model_name_or_path=model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error during model initialization: {e}")
    raise

@app.route('/similarity', methods=['POST'])
def similarity():
    """
    Endpoint to calculate similarity between two texts.
    Expects a POST request with JSON containing 'text1' and 'text2'.
    """
    try:
        data = request.get_json()

        text1 = data.get('text1')
        text2 = data.get('text2')

        if not text1 or not text2:
            return jsonify({'error': 'text1 and text2 are required.'}), 400

        with torch.no_grad():
            embedding1 = model.encode(text1, convert_to_tensor=True).to(device)
            embedding2 = model.encode(text2, convert_to_tensor=True).to(device)
            cosine_score = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()

        return jsonify({'similarity score': cosine_score})
    except Exception as e:
        logging.error(f"Error in similarity endpoint: {e}")
        return jsonify({'error': 'An error occurred during processing.'}), 500

if __name__ == '__main__':
    app.run(debug=True)