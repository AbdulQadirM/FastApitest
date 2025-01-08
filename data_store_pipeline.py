# import os
# from openai import OpenAI
# from langchain_openai import ChatOpenAI

# class VideoTranscriber:
#     def __init__(self, openai_api_key: str):
#         # Set the OpenAI API Key dynamically
#         os.environ["OPENAI_API_KEY"] = openai_api_key
#         self.client = OpenAI()
#         self.llm = ChatOpenAI(model='gpt-4o', temperature=0.1)

#     def transcribe(self, file_path: str):
#         """
#         Method to transcribe the given audio/video file.
#         """
#         with open(file_path, "rb") as audio_file:
#             transcription = self.client.audio.transcriptions.create(
#                 model="whisper-1",
#                 file=audio_file,
#                 response_format="text"
#             )
#         return transcription


import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document  # Import the Document class

class VideoTranscriber:
    def vec_init(self, vectorstore_directory="Database"):
        self.vectorstore_directory = vectorstore_directory
        os.makedirs(self.vectorstore_directory, exist_ok=True)

    def set_openai_api_key(self, api_key: str):
        """
        Set the OpenAI API key dynamically.
        """
        os.environ["OPENAI_API_KEY"] = api_key
        self.client = OpenAI()

    def transcribe(self, file_path: str):
        """
        Method to transcribe the given audio/video file.
        """
        if not self.client:
            raise ValueError("OpenAI client is not initialized. Set the API key first.")
        with open(file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )

        # Wrap the transcription string in a Document object
        document = Document(page_content=transcription)

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents([document])

        # Create the FAISS vector store and save locally
        vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
        vectorstore.save_local(folder_path=self.vectorstore_directory)

        return transcription
