
import os
from uuid import uuid4
from typing import List 
from dotenv import load_dotenv
from langsmith import traceable 
from DSPy.dspy.dspy.retrievers import embeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
# -------------------------
# Utils
# -------------------------
def extract_text(document_path: str) -> str:
    with open(document_path, "r", encoding="utf-8") as file:
        return file.read()

# -------------------------
# Vector DB Base
# -------------------------
class VectorDB:
    def __init__(self, embedding_model, collection_name=None):
        self.e_model = embedding_model
        self.collection_name = collection_name

    def add_documents(self, documents: List[str]):
        raise NotImplementedError

    def similarity_search(self, query: str, topK: int = 3) -> List[str]:
        raise NotImplementedError

# -------------------------
# Chroma Implementation
# -------------------------
class ChromaDB(VectorDB):
    def __init__(self, embedding_model=None, collection_name=None):
        super().__init__(embedding_model, collection_name)
        self.chroma_client = Chroma(collection_name=self.collection_name,
                                    embedding_function=self.e_model,
                                    persist_directory="./chroma_persistence")
        # Initialize ChromaDB client here
    def add_documents(self, documents: List[str]):
        # Code to add documents to ChromaDB
        ids = [uuid4().hex for _ in documents]
        self.chroma_client.add_texts(texts=documents, ids=ids)
    @traceable(name="vector_search")
    def similarity_search(self, query: str, topK: int = 3) -> List[str]:
        # Code to query ChromaDB and return topK results
        # Returns doc object
        results = self.chroma_client.similarity_search(query=query, k=topK)
        return [doc.page_content for doc in results]


class RAG():
    def __init__(self, vectorDB: VectorDB, knowledge_dir: str):
        self.llm = ChatOpenAI(base_url='http://localhost:8080/v1', api_key="")
        self.vector_db = vectorDB
        self.documents, self.k_dir = [], knowledge_dir
        for filename in os.listdir(self.k_dir):
            if filename != "_processed":
                self.documents.append(os.path.join(self.k_dir, filename))
        # Insert all the documents for knowledge DB.
        self._insert()
        
    def _insert(self):
        _processed_doc = []
        try:
            with open(f"{self.k_dir}/_processed", "r") as file:
                for line in file:
                    _processed_doc.append(line.strip())
        except:
            with open(f"{self.k_dir}/_processed", "a") as file:
                print(f'Creating: {self.k_dir}/_processed')
        prev_size = len(_processed_doc)
        for doc in self.documents:
            if doc not in _processed_doc:
                text = extract_text(doc)
                number_of_chunks = 20
                chunk_size = max(500, len(text) // number_of_chunks)
                chunk_overlap = max(50, chunk_size * 0.1)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = text_splitter.split_text(text)
                self.vector_db.add_documents(chunks)
                _processed_doc.append(doc)
            else: print(f'SKIP: {doc} already processed.')
        if len(_processed_doc) > prev_size:
            with open(f"{self.k_dir}/_processed", "w") as file:
                for item in _processed_doc:
                    file.write(f"{item}\n")
    
    @traceable(name="rag_based_answer")
    def answer(self, query: str, topK: int = 3) -> str:
        relevant_chunks = self.vector_db.similarity_search(query, topK)
        context = "\n\n".join(relevant_chunks)
        prompt = [
            SystemMessage(content="You are a helpful assistant that answers questions based on the provided context."),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
        ]
        response = self.llm.invoke(prompt)
        return response.content


if __name__ == "__main__":
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
    vectorDB = ChromaDB(embedding_model = embedding_model, collection_name = "chroma_collection")
    rag, text = RAG(vectorDB = vectorDB, knowledge_dir = "rag-doc"), ""
    while text != "finish":
        text = input("\nAsk a question (or type 'finish' to exit): ")
        if text.lower() != "finish":
            print(f'\nAnswer: {rag.answer(text)}')
        else: break