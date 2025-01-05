from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

class VectorStoreLoader:
    """
    A class to initialize and load a Chroma vector store.
    """

    def __init__(self):
        self.vectorstore = None

    def initialize_vectorstore(self,folder_path):
        """
        Initializes a Chroma vector store with OpenAI embeddings.
        """
        embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(embedding_function=embeddings, persist_directory=folder_path)

    def get_vectorstore(self):
        """
        Returns the initialized vector store.

        Raises:
            RuntimeError: If the vector store is not initialized.
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store is not initialized. Please call initialize_vectorstore first.")
        return self.vectorstore
    

    


