import faiss
from sentence_transformers import SentenceTransformer
import pickle
import os

# Initialize FAISS and embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = embedding_model.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(dimension)

# Query-response mapping
query_response_mapping = {}

# File paths
FAISS_INDEX_PATH = "faiss_data/faiss_index.bin"
QUERY_MAPPING_PATH = "faiss_data/query_mapping.pkl"

# Ensure the data folder exists
os.makedirs("faiss_data", exist_ok=True)

# Save FAISS index and mapping
def save_faiss_data():
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    with open(QUERY_MAPPING_PATH, "wb") as f:
        pickle.dump(query_response_mapping, f)

# Load FAISS index and mapping
def load_faiss_data():
    global faiss_index, query_response_mapping
    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(QUERY_MAPPING_PATH):
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(QUERY_MAPPING_PATH, "rb") as f:
            query_response_mapping = pickle.load(f)
    else:
        print("No existing FAISS data found. Starting fresh.")

# Add query and response to FAISS
def add_query_to_faiss(query, response):
    embedding = embedding_model.encode([query])
    faiss_index.add(embedding)
    query_response_mapping[len(query_response_mapping)] = {"query": query, "response": response}

# Retrieve response from FAISS
def get_response_from_faiss(query, threshold=0.7):
    embedding = embedding_model.encode([query])
    distances, indices = faiss_index.search(embedding, 1)
    if distances[0][0] <= threshold and indices[0][0] != -1:
        return query_response_mapping[indices[0][0]]["response"]
    return None
