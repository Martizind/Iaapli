import ollama

response = ollama.chat(model='llama3', messages=[
  {
    'role': 'user',
    'content': 'Diz "Sistema pronto" se me estás a ouvir.',
  },
])
print(response['message']['content'])