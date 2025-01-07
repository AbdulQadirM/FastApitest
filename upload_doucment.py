from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import Docx2txtLoader
from openai import OpenAI
from langchain_community.vectorstores import FAISS
from typing import List

class DocumentUploader:
    def vector_dict(self, vectorstore_directory: str = "Database"):
        """
        Initialize DocumentUploader with the directory for the vector store.
        The directory is set to "Database" by default.
        """
        self.vectorstore_directory = os.path.abspath(vectorstore_directory)
        os.makedirs(self.vectorstore_directory, exist_ok=True)

    def upload_documents(self, file_paths: List[str]):
        """
        Uploads documents to the vector store.

        Args:
            file_paths: A list of file paths to upload.
        """
        for file_path in file_paths:
            file_extension = os.path.splitext(file_path)[1].lower()

            # Choose the appropriate loader based on file type
            if file_extension == ".txt":
                loader = TextLoader(file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(file_path)
            elif file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            else:
                print(f"Unsupported file type: {file_extension}. Skipping {file_path}.")
                continue

            # Load the document and split into chunks
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)

            # Create the FAISS vector store and save locally
            vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
            vectorstore.save_local(folder_path=self.vectorstore_directory)
