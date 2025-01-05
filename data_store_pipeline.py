import os
from openai import OpenAI
from langchain_openai import ChatOpenAI

class VideoTranscriber:
    def __init__(self, openai_api_key: str):
        # Set the OpenAI API Key dynamically
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.client = OpenAI()
        self.llm = ChatOpenAI(model='gpt-4o', temperature=0.1)

    def transcribe(self, file_path: str):
        """
        Method to transcribe the given audio/video file.
        """
        with open(file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return transcription


