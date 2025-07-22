import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="Asistente Laboral - Código del Trabajo - version chorizo")
st.title("🧑‍⚖️ Asistente Laboral – Código del Trabajo Chileno version chorizo")
st.write("Haz preguntas sobre el Código del Trabajo y recibe respuestas claras basadas en el texto oficial.")

# Cargar clave de OpenAI
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_chain():
    loader = PyPDFLoader("codigo_trabajo.pdf")
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Configurar modelo
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Prompt personalizado
    custom_prompt = PromptTemplate.from_template(
        "Eres un asistente legal especializado en el Código del Trabajo chileno pero hablas chileno estilo flayte hasta con garabatos. "
        "Responde la siguiente pregunta de forma clara y profesional utilizando únicamente la información del documento. "
        "Si la respuesta no está en el documento, responde: 'No tengo información suficiente en el Código del Trabajo para responder eso.'\n\n"
        "Pregunta: {question}\n\n"
        "Contexto relevante:\n{context}\n\n"
        "Respuesta:"
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt}
    )
    return qa

qa_chain = load_chain()

pregunta = st.text_input("Escribe tu pregunta aquí:")

if pregunta:
    respuesta = qa_chain.run(pregunta)
    st.markdown("### ✅ Respuesta:")
    st.write(respuesta)
