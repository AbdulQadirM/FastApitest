from fastapi import FastAPI

# Create a FastAPI app instance
app = FastAPI()

# Define a route
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}
