
# Import all the necessary library
import logging
import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Disable parallel tokenization to avoid issues with multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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

try:
    # Read the csv file from INPUT directory which contains text data
    df = pd.read_csv('INPUT/Precily_Text_Similarity.csv')
    logging.info(f"DataFrame shape: {df.shape}")
    logging.info(f"DataFrame columns: {df.columns}")
    logging.info(f"Sample data:\n{df.sample(3).to_string()}")
    logging.info(f"Null values in DataFrame: {df.isna().sum()}")

    # Load SentenceTransformer model
    model_path = 'sts_model'  # load the model from local directory 
    model = SentenceTransformer(model_name_or_path=model_path)

    batch_size = 64

    def encode_sentences(sentences):
        """
        Encode sentences using the SentenceTransformer model.
        Processes sentences in batches.
        """
        model.to(device)
        return model.encode(sentences, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False).to(device)

    def encode(sentences):
        """
        Encode a list of sentences into embeddings, handling large lists by batching.
        """
        embeddings = []
        for start_index in tqdm(range(0, len(sentences), batch_size)):
            sentences_batch = sentences[start_index:start_index + batch_size]
            batch_embeddings = encode_sentences(sentences_batch)
            embeddings.append(batch_embeddings)
        return torch.cat(embeddings, dim=0)

    # Perform encoding on the text data
    embeddings1 = encode(df['text1'].tolist())
    embeddings2 = encode(df['text2'].tolist())

    # Compute cosine similarity between pairs of embeddings
    cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    df['Similarity'] = cosine_scores.diag().cpu().numpy()

    # Log and save the result
    logging.info(df.sort_values('Similarity', ascending=False))
    df.to_csv('OUTPUT/result_task_A.csv', index=False)

except Exception as e:
    logging.error(f"An error occurred: {e}")
