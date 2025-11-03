import streamlit as st
import time
import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# --- Konfigurasi Halaman ---
st.set_page_config(
    page_title="Tanya Jawab TA",
    page_icon="🎓",
    layout="centered"
) #

# --- Konfigurasi Model & Database ---
NAMA_FOLDER_DB = "db_thesis" #
MODEL_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2" #
MODEL_LLM = "groq/compound" #

# --- Memuat API Key dari .env ---
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") #

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY tidak ditemukan. Pastikan file .env Anda sudah benar.")
    st.stop() #

# --- Fungsi Caching untuk Performa ---
@st.cache_resource
def muat_llm():
    """Memuat model LLM dari Groq."""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=MODEL_LLM,
        temperature=0
    ) #

@st.cache_resource
def muat_vector_store():
    """Memuat model embedding dan database vektor."""
    # Siapkan model embedding
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    ) #
    
    # Muat database vektor dari disk
    db = Chroma(
        persist_directory=NAMA_FOLDER_DB, 
        embedding_function=embeddings
    ) #
    return db

@st.cache_resource
def buat_rag_chain(_vector_store, _llm):
    """Membuat RAG chain lengkap."""
    
    # 1. Buat Retriever (si pencari)
    # --- INI ADALAH SOLUSI KEDUA KITA ---
    # Mengurangi k=10 menjadi k=5 untuk mengurangi "kebisingan" konteks.
    retriever = _vector_store.as_retriever(search_kwargs={"k": 15})

    # 2. Buat Prompt Template
    template = """
    Anda adalah asisten AI yang ahli dalam menjawab pertanyaan 
    berdasarkan sebuah Tugas Akhir (TA) penelitian. 
    
    Tugas Anda adalah menjawab pertanyaan pengguna HANYA 
    berdasarkan konteks yang diberikan di bawah ini.
    
    Jika informasi tidak ada dalam konteks, jawab dengan sopan:
    "Maaf, saya tidak dapat menemukan informasi tersebut di dalam dokumen."
    
    Jangan berspekulasi atau menambahkan informasi dari luar konteks.
    Jawab dalam Bahasa Indonesia yang baik dan jelas.

    KONTEKS:
    {context}

    PERTANYAAN:
    {input}

    JAWABAN:
    """ #
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Buat Rantai
    question_answer_chain = create_stuff_documents_chain(_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain #

# --- Tampilan Utama Aplikasi ---

st.title("🎓 Bot Tanya Jawab Tugas Akhir")
st.markdown("Ajukan pertanyaan apa pun tentang isi Tugas Akhir ini!") #

# Muat semua komponen
with st.spinner("Mempersiapkan model dan database... Ini mungkin perlu waktu sejenak."):
    try:
        llm = muat_llm()
        vector_store = muat_vector_store()
        rag_chain = buat_rag_chain(vector_store, llm)
        st.success("Bot siap!")
    except Exception as e:
        st.error(f"Gagal memuat komponen: {e}")
        st.stop() #

# Inisialisasi riwayat chat
if "messages" not in st.session_state:
    st.session_state.messages = [] #

# Tampilkan pesan-pesan sebelumnya
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) #

# Terima input dari pengguna
if prompt := st.chat_input("Apa yang ingin Anda tanyakan?"):
    # Tampilkan pesan pengguna di UI
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Tambahkan pesan pengguna ke riwayat
    st.session_state.messages.append({"role": "user", "content": prompt}) #

    # Dapatkan respons dari bot
    with st.chat_message("assistant"):
        with st.spinner("Sedang berpikir..."):
            start_time = time.time()
            
            # Panggil RAG chain
            response = rag_chain.invoke({"input": prompt}) #
            
            end_time = time.time()
            waktu_respons = end_time - start_time
            
            # Tampilkan jawaban bot
            jawaban_bot = response['answer']
            st.markdown(jawaban_bot)
            st.caption(f"Waktu respons: {waktu_respons:.2f} detik")
    
    # Tambahkan respons bot ke riwayat
    st.session_state.messages.append({"role": "assistant", "content": jawaban_bot}) #
