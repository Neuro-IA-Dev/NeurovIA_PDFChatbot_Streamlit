import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

st.set_page_config(page_title="Asistente Laboral - Código del Trabajo")
st.title("🧑‍⚖️ Asistente Laboral – Código del Trabajo Chileno")
st.write("Haz preguntas sobre el Código del Trabajo y recibe respuestas claras basadas en el texto oficial.")

# Agregar la API Key de OpenAI desde los secretos de Streamlit
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Carga e indexa el PDF
@st.cache_resource
def load_chain():
    loader = PyPDFLoader("codigo_trabajo.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return chain

qa_chain = load_chain()

pregunta = st.text_input("Escribe tu pregunta aquí:")

if pregunta:
    respuesta = qa_chain.run(pregunta)
    st.markdown("### ✅ Respuesta:")
    st.write(respuesta)