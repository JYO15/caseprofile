import streamlit as st
import os
from optimized_rag import OptimizedRAG

def run_rag_query(query: str):
    """Run the RAG query and print the response and time taken."""
    rag = OptimizedRAG()
    pdf_directory = "D:/legal_rag_project/"

    if os.path.exists("ipc_vector_db"):
        rag.load_index(pdf_directory)
    else:
        documents = rag.load_documents(pdf_directory)
        if documents:
            rag.build_index(documents)

    retrieved_docs = rag.retrieve_relevant_docs(query)
    
    if not retrieved_docs:
        st.write("No relevant documents were found for the query.")
    else:
        response, time_taken = rag.generate_response(query, retrieved_docs)
        st.write("Query:", query)
        st.write("Response:", response)
        st.write("Time taken:", time_taken)

if __name__ == "__main__":
    query = "Explain the punishment for offenses committed outside India"
    run_rag_query(query)
