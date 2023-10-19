from flask import Flask, request, jsonify, render_template

import os

from transformers import GPT2TokenizerFast
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
import textract

app = Flask(__name__)

os.environ["OPENAI_API_KEY"] = "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"

# Lectura y Procesamiento del PDF
#loader = PyPDFLoader("./SK_Manual_Usuario_Credito_Empresarial_V1.4.6.pdf")


# Convertir PDF a texto
doc = textract.process("./SK_Manual_Usuario_Credito_Empresarial_V1.4.6.pdf")

# Guardar a .txt y reabrir (esto ayuda a prevenir ciertos problemas según tu código original)
with open('SK_Manual_Usuario_Credito_Empresarial_V1.4.6.txt', 'w') as f:
    f.write(doc.decode('utf-8'))

with open('SK_Manual_Usuario_Credito_Empresarial_V1.4.6.txt', 'r') as f:
    text = f.read()

# Función para contar tokens
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

# Dividir el texto en chunks basados en tokens
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 512,  # Aproximadamente el tamaño máximo para muchos modelos, pero ajusta según tus necesidades
    chunk_overlap  = 24,  # Un pequeño solapamiento para asegurar continuidad
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])

#print("voy a crear la base de datos")
# Embeddings y Creación de la base de datos vectorial
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)

# Creación de la cadena de QA
chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.1), db.as_retriever())

contexto_inicial = 'Soy un Ingeniero de Soporte especializado en el Manual de Usuario de Crédito Empresarial de SKIT-Koncilia. Estoy aquí para ayudarte a resolver cualquier duda técnica que tengas.'
chat_history = [(contexto_inicial, '')]

# Añade esta función y ruta aquí:
@app.route('/')
def index():
    return render_template('Front.html')


@app.route('/get-response', methods=['POST'])
def get_response():
    user_query = request.json['question']
    result = qa({"question": user_query, 
                 "chat_history": chat_history,
                 "context": contexto_inicial})
    
    chat_history.append((user_query, result['answer']))
    response = {"answer": result['answer']}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
