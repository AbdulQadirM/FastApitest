# from fastapi import FastAPI, UploadFile, File, HTTPException, Form
# from typing import List
# from data_store_pipeline import VideoTranscriber
# from upload_doucment import DocumentUploader
# from chat_funtion import Chatbot 
# from vectorstore_loader import VectorStoreLoader
# from langchain_community.document_loaders import PyPDFLoader , TextLoader , Docx2txtLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings , ChatOpenAI

# from langchain_community.vectorstores import FAISS

# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate
# from openai import OpenAI
# import os

# from pydantic import BaseModel



# app = FastAPI()


# # Pydantic model to handle user input
# class TranscriptionRequest(BaseModel):
#     openai_api_key: str  # Accept OpenAI API key
#     file_path: str  # Accept file path for transcription

# @app.post("/transcribe-video/")
# async def transcribe_video(request: TranscriptionRequest):
#     video_transcriber = VideoTranscriber(openai_api_key=request.openai_api_key)
#     # Perform transcription on the provided file path
#     transcription = video_transcriber.transcribe(request.file_path)
    
#     return {"status": "success", "transcription": transcription}



# # Combined endpoint for both uploading documents and checking vector store status
# @app.post("/manage-documents/")
# async def manage_documents(
#     files: List[UploadFile] = File(...), 
#     vectorstore_directory: str = 'Database',
#     openai_api_key: str = Form(...)  # Accept OpenAI API key as form data
# ):
#     """
#     Endpoint to upload documents to the vector store or check if the vector store is initialized.
#     """
#     # Set the OpenAI API key dynamically
#     os.environ["OPENAI_API_KEY"] = openai_api_key

#     # Initialize DocumentUploader with a custom vector store directory
#     uploader = DocumentUploader(vectorstore_directory=vectorstore_directory)
    
#     if files:  # If files are present, it is an upload request
#         file_paths = []
#         for file in files:
#             file_location = f"temp_files/{file.filename}"
#             with open(file_location, "wb") as buffer:
#                 buffer.write(await file.read())
#                 file_paths.append(file_location)
           
#             # Upload documents to the vector store
#         uploader.upload_documents(file_paths)
#         return {"status": "success", "message": f"Documents uploaded successfully to {vectorstore_directory}."}

#     else:
#         file_paths.append(file_location)  # If no files are provided, it is a check request
  
#         vectorstore = uploader.get_vectorstore()
#         return {"status": "success", "message": "Vector store is initialized."}
       



# # Pydantic model to handle user input
# class QueryRequest(BaseModel):
#     user_query: str
#     openai_api_key: str
#     vectorstore_directory: str = 'Database'  # Default directory for vectorstore

# @app.post("/ask-chatbot/")
# async def ask_chatbot(query_request: QueryRequest):
#     openai_api_key = query_request.openai_api_key
#     os.environ["OPENAI_API_KEY"] = openai_api_key
            
#     loader = VectorStoreLoader()
#     # Initialize the vector store
#     loader.initialize_vectorstore(query_request.vectorstore_directory)
#     # Get the initialized vector store
#     vectorstores = loader.get_vectorstore()
    
    
#     # Create an instance of the Chatbot class with the OpenAI API key
#     chatbot = Chatbot(openai_api_key)
    
#     # Get the response from the chatbot
#     response, chat_history = chatbot.create_and_get_chat_response(vectorstores, query_request.user_query)

#     return {"status": "success", "response": response, "chat_history": chat_history}





from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

class VideoTranscriber:
    def _init_(self, vectorstore_directory="Database"):
        # Load OpenAI API Key from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set in the .env file")
        
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.client = OpenAI()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.vectorstore_directory = vectorstore_directory
        os.makedirs(self.vectorstore_directory, exist_ok=True)

    def transcribe(self, file_path: str):
        """
        Method to transcribe the given audio/video file.
        """
        with open(file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )
        return transcription


# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize VideoTranscriber
try:
    transcriber = VideoTranscriber()
except ValueError as e:
    raise RuntimeError(f"Initialization error: {e}")

@app.post("/upload/")
async def upload_video(file: UploadFile):
    try:
        # Save the uploaded file temporarily
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Perform transcription
        transcription = transcriber.transcribe(file_path)

        # Clean up the uploaded file
        os.remove(file_path)

        return JSONResponse(content={"transcription": transcription})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the Video Transcription API!"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use PORT from environment variables
    uvicorn.run(app, host="0.0.0.0", port=port)
