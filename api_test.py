from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# Initialize the OpenAI client with the API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Use the new API format for ChatCompletion
response = client.chat.completions.create(
    model="gpt-4",  # or "gpt-3.5-turbo"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]
)

# Print the assistant's reply
print("Assistant's Reply:", response.choices[0].message.content)
