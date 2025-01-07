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




from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


class DocumentUploader:
    def _init_(self, vectorstore_directory: str = "Database"):
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

@app.post("/manage-documents/")
async def manage_documents(
    files: List[UploadFile],
    openai_api_key: str = Form(...),
):
    try:
        # Set the OpenAI API key dynamically
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # The directory is always "Database"
        vectorstore_path = os.path.abspath("Database")
        os.makedirs(vectorstore_path, exist_ok=True)

        # Save uploaded files temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        temp_file_paths = []
        for file in files:
            temp_path = os.path.join(temp_dir, file.filename)
            temp_file_paths.append(temp_path)
            with open(temp_path, "wb") as f:
                f.write(await file.read())

        # Upload documents to vector store
        uploader = DocumentUploader(vectorstore_directory=vectorstore_path)
        uploader.upload_documents(temp_file_paths)

        # Clean up temporary files
        for temp_path in temp_file_paths:
            os.remove(temp_path)

        return JSONResponse(content={"message": "Documents uploaded and vector store updated successfully."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Welcome to the Document Uploader API!"}
    
if __name__ == "__main__":
import uvicorn
port = int(os.getenv("PORT", 8000))  # Use PORT from environment variables
uvicorn.run(app, host="0.0.0.0", port=port)
