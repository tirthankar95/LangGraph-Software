from langchain_openai import ChatOpenAI 
from langchain_core.messages import (
    HumanMessage, 
    SystemMessage, 
    AIMessage  
)
import mlflow

mlflow.set_tracking_uri(
    "http://localhost:5000"
)
mlflow.openai.autolog()

model = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key=""
)
prompts = [
    HumanMessage(content='What is the capital of France?')
]
response = model.invoke(prompts)
print(response.content)