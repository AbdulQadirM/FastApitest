from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import List
from data_store_pipeline import VideoTranscriber
from upload_doucment import DocumentUploader
from chat_funtion import Chatbot 
from vectorstore_loader import VectorStoreLoader
from langchain_community.document_loaders import PyPDFLoader , TextLoader , Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from langchain_chroma import Chroma


from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
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
    """
    Endpoint to transcribe a video file using OpenAI Whisper model.
    """
    try:
        # Initialize VideoTranscriber with the provided OpenAI API key
        video_transcriber = VideoTranscriber(openai_api_key=request.openai_api_key)
        
        # Perform transcription on the provided file path
        transcription = video_transcriber.transcribe(request.file_path)
        
        return {"status": "success", "transcription": transcription}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transcription request: {str(e)}")


# Combined endpoint for both uploading documents and checking vector store status
@app.post("/manage-documents/")
async def manage_documents(
    files: List[UploadFile] = File(...), 
    vectorstore_directory: str = 'Database',
    openai_api_key: str = Form(...)  # Accept OpenAI API key as form data
):
    """
    Endpoint to upload documents to the vector store or check if the vector store is initialized.
    """
    # Set the OpenAI API key dynamically
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Initialize DocumentUploader with a custom vector store directory
    uploader = DocumentUploader(vectorstore_directory=vectorstore_directory)
    
    if files:  # If files are present, it is an upload request
        file_paths = []
        for file in files:
            file_location = f"temp_files/{file.filename}"
            try:
                # Save the file to the "temp_files" directory
                with open(file_location, "wb") as buffer:
                    buffer.write(await file.read())
                file_paths.append(file_location)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error saving file {file.filename}: {str(e)}")
        
        try:
            # Upload documents to the vector store
            uploader.upload_documents(file_paths)
            return {"status": "success", "message": f"Documents uploaded successfully to {vectorstore_directory}."}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")
    
    else:  # If no files are provided, it is a check request
        try:
            # Retrieve the vectorstore to check if it is initialized
            vectorstore = uploader.get_vectorstore()
            return {"status": "success", "message": "Vector store is initialized."}
        except RuntimeError:
            return {"status": "error", "message": "Vector store is not initialized."}


# Pydantic model to handle user input
class QueryRequest(BaseModel):
    user_query: str
    openai_api_key: str
    vectorstore_directory: str = 'Database'  # Default directory for vectorstore

@app.post("/ask-chatbot/")
async def ask_chatbot(query_request: QueryRequest):
    """
    Endpoint to process user query with chatbot, and return response from vectorstore.
    """
    try:
        # Set the OpenAI API Key
        openai_api_key = query_request.openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
     # Initialize the vector store
                
        loader = VectorStoreLoader()
        # Initialize the vector store
        loader.initialize_vectorstore(query_request.vectorstore_directory)
        # Get the initialized vector store
        vectorstores = loader.get_vectorstore()
        
        
        # Create an instance of the Chatbot class with the OpenAI API key
        chatbot = Chatbot(openai_api_key)
        
        # Get the response from the chatbot
        response, chat_history = chatbot.create_and_get_chat_response(vectorstores, query_request.user_query)

        return {"status": "success", "response": response, "chat_history": chat_history}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use PORT from environment variables
    uvicorn.run(app, host="0.0.0.0", port=port)
