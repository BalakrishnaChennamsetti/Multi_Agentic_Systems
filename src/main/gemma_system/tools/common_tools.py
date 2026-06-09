from ollama import chat

response = chat(
    model='gemma4:12b',
    messages=[{'role': 'user', 'content': 'Hello!'}],
)
print(response.message.content)