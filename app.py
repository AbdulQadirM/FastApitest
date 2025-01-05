from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import List
from data_store_pipeline import VideoTranscriber
from openai import OpenAI
import os
from pydantic import BaseModel

app = FastAPI()
# Pydantic model to handle user input
class TranscriptionRequest(BaseModel):
    openai_api_key: str  # Accept OpenAI API key
    file_path: str  # Accept file path for transcription

@app.post("/transcribe-video/")
async def transcribe_video(request: TranscriptionRequest):

    # Initialize VideoTranscriber with the provided OpenAI API key
    video_transcriber = VideoTranscriber(openai_api_key=request.openai_api_key)
    
    # Perform transcription on the provided file path
    transcription = video_transcriber.transcribe(request.file_path)
    
    return {"status": "success", "transcription": transcription}
    
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use PORT from environment variables
    uvicorn.run(app, host="0.0.0.0", port=port)
