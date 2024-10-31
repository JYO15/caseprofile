import faiss
import fitz  # PyMuPDF
import torch
import logging
import sys
import os
import numpy as np
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import time

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class OptimizedRAG:
    def __init__(self, cache_dir: str = "D:/legal_rag_project"):
        self.cache_dir = cache_dir
        self.setup_models()
        self.index = None
        self.documents = []
        self.document_chunks = []

    def setup_models(self):
        """Initialize the embedding model and LLM with specified model configurations."""
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=self.cache_dir, torch_dtype=torch.float32
            ).to("cpu")
        except Exception as e:
            logger.error("Error initializing models:", exc_info=True)
            raise

    def load_documents(self, directory_path: str) -> List[str]:
        """Load PDF documents from a directory."""
        document_texts = []
        for pdf_file in os.listdir(directory_path):
            if pdf_file.endswith(".pdf"):
                file_path = os.path.join(directory_path, pdf_file)
                try:
                    with fitz.open(file_path) as pdf:
                        text = "".join(page.get_text() for page in pdf)
                        document_texts.append(text)
                except Exception as e:
                    logger.error(f"Error loading {file_path}:", exc_info=True)
        return document_texts

    def split_documents(self, documents: List[str], chunk_size: int = 1024, chunk_overlap: int = 200) -> List[str]:
        """Split documents into chunks to optimize retrieval."""
        chunks = []
        for doc in documents:
            for i in range(0, len(doc), chunk_size - chunk_overlap):
                chunks.append(doc[i:i + chunk_size])
        return chunks

    def build_index(self, documents: List[str]):
        """Build and save the FAISS index for quick document retrieval."""
        self.documents = documents
        document_chunks = self.split_documents(documents)
        embeddings = self.embedder.encode(document_chunks)
        embeddings_array = np.array(embeddings).astype('float32')

        self.index = faiss.IndexFlatL2(embeddings_array.shape[1])
        self.index.add(embeddings_array)
        self.document_chunks = document_chunks

        faiss.write_index(self.index, "ipc_vector_db")
        logger.info("FAISS index built and saved to ipc_vector_db")
        logger.info(f"Number of document chunks indexed: {len(self.document_chunks)}")

    def load_index(self, directory_path: str):
        """Load a pre-saved FAISS index and reload document chunks to maintain alignment."""
        if os.path.exists("ipc_vector_db"):
            self.index = faiss.read_index("ipc_vector_db")
            logger.info("FAISS index loaded from ipc_vector_db")

            documents = self.load_documents(directory_path)
            self.document_chunks = self.split_documents(documents)
            logger.info(f"Number of document chunks reloaded: {len(self.document_chunks)}")
        else:
            raise ValueError("Index file not found. Please build the index first.")

    def retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve the most relevant document chunks for a query."""
        query_embedding = self.embedder.encode([query])
        logger.info(f"Query embedding shape: {query_embedding.shape}")

        distances, indices = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        logger.info(f"Distances for retrieved documents: {distances}")

        if len(indices) > 0 and len(indices[0]) > 0:
            retrieved_chunks = [self.document_chunks[i] for i in indices[0] if i < len(self.document_chunks)]
            if retrieved_chunks:
                return retrieved_chunks
            else:
                logger.warning("Indices were returned, but they are out of bounds.")
        else:
            logger.warning("No relevant documents found for the query.")

        return []

    def format_prompt(self, query: str, context: str) -> str:
        """Format the prompt for the model, combining query and context."""
        return f"<|system|>\nYou are a legal assistant. Use the context to answer.\n<|user|>\nContext: {context}\nQuestion: {query}\n<|assistant|>"

    def generate_response(self, query: str, retrieved_docs: List[str]):
        """Generate a response without streaming to debug generation issues."""
        start_time = time.time()
        context = " ".join(retrieved_docs)[:1500]  # Reduce context size for CPU usage
        prompt = self.format_prompt(query, context)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cpu")

        generation_kwargs = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'max_new_tokens': 150,
            'temperature': 0.7,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id
        }

        try:
            output = self.model.generate(**generation_kwargs)
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            time_taken = time.time() - start_time
            return generated_text, time_taken  # Return both generated_text and time_taken
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return None, None


def main():
    rag = OptimizedRAG()
    pdf_directory = "D:/legal_rag_project/"
    
    if os.path.exists("ipc_vector_db"):
        rag.load_index(pdf_directory)
    else:
        documents = rag.load_documents(pdf_directory)
        if documents:
            rag.build_index(documents)

    query = "Explain the punishment for offenses committed outside India"
    retrieved_docs = rag.retrieve_relevant_docs(query)
    
    if not retrieved_docs:
        print("No relevant documents were found for the query.")
    else:
        print("\nQuery:", query)
        response, time_taken = rag.generate_response(query, retrieved_docs)
        print("\nResponse:", response)
        print(f"Time taken: {time_taken:.2f} seconds")

if __name__ == "__main__":
    main()
