from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from openai import OpenAI
import os

class Chatbot:
    def __init__(self, openai_api_key: str):
        """
        Initialize the Chatbot with an OpenAI API key and set up the OpenAI client.
        """
        # Pass the API key directly to OpenAI client
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(model='gpt-4o', temperature=0.1)
        self.client = OpenAI()

    def create_and_get_chat_response(self, vectorstores, user_query):
        """
        Process the user query by fetching relevant context from the vectorstore
        and generating a helpful response using the GPT model.
        """
        try:
            # Initialize retriever for similarity search
            retriever = vectorstores.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            
            # Define the template for generating responses
            template = """You are an AI Q&A chatbot assistant. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
            {context}
            Question: {question}
            Helpful Answer:
            """

            QA_PROMPT = PromptTemplate(
                template=template, input_variables=["context", "question"]
            )
            
            # Create the conversational retrieval chain
            chat_chain = RetrievalQA.from_chain_type(
                llm=self.llm, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": QA_PROMPT}
            )
            
            # Get the response from the chain
            response = chat_chain(user_query)
            
            # Prepare chat history (if you need to track the conversation)
            chat_history = [(user_query, response['result'])]
            
            return response['result'], chat_history
        
        except Exception as e:
            print(f"Error occurred during query processing: {str(e)}")
            return "An error occurred while processing your query.", []



