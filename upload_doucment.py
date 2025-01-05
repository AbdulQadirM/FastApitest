from langchain_community.document_loaders import PyPDFLoader, TextLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from openai import OpenAI

class DocumentUploader:
    """
    A class for uploading and storing documents in a vector store using Chroma.
    """

    def __init__(self, vectorstore_directory='Database', openai_api_key=None):
        """
        Initialize DocumentUploader with an optional custom directory for the vector store
        and an optional OpenAI API key.
        """
        self.vectorstore = None
        self.vectorstore_directory = vectorstore_directory
        
        # Set the OpenAI API key if provided
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key

        # Ensure the directory exists
        os.makedirs(self.vectorstore_directory, exist_ok=True)

    def upload_documents(self, file_paths):
        """
        Uploads documents to a vector store.

        Args:
            file_paths: A list of file paths to upload.
        """
        for file_path in file_paths:
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension == ".txt":
                loader = TextLoader(file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(file_path)
            elif file_extension == ".pdf":
                loader = PyPDFLoader(file_path)
            else:
                print(f"Unsupported file type: {file_extension}. Skipping {file_path}.")
                continue

            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            docs = text_splitter.split_documents(documents)

            # Initialize vector store only on the first call
            if self.vectorstore is None:
                self.vectorstore = Chroma.from_documents(docs, OpenAIEmbeddings(), persist_directory=self.vectorstore_directory)
            else:
                # Add new documents to the existing vector store
                self.vectorstore.add_documents(docs)

    def get_vectorstore(self):
        """
        Returns the initialized vector store.

        Raises:
            RuntimeError: If the vector store is not initialized.
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store is not initialized. Please call upload_documents first.")
        return self.vectorstore
