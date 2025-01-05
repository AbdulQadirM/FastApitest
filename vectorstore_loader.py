from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS



class VectorStoreLoader:
    """
    A class to initialize and load a Chroma vector store.
    """

    def _init_(self):
        self.vectorstore = None

    def initialize_vectorstore(self,folder_path):
        """
        Initializes a Chroma vector store with OpenAI embeddings.
        """
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.load_local(folder_path=folder_path,embeddings=embeddings,allow_dangerous_deserialization=True)

    def get_vectorstore(self):
        """
        Returns the initialized vector store.

        Raises:
            RuntimeError: If the vector store is not initialized.
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store is not initialized. Please call initialize_vectorstore first.")
        return self.vectorstore
