# 🎓 Chatbot Tugas Akhir (Streamlit + Groq + RAG)

Repositori ini berisi kode sumber untuk aplikasi web "Chatbot Tugas Akhir", sebuah portofolio yang mendemonstrasikan implementasi RAG (Retrieval-Augmented Generation) pada naskah penelitian akademis.

## 🚀 DEMO LIVE

**Coba aplikasinya langsung di sini:**
[**https://huggingface.co/spaces/wikan1602/thesis-chatbot**](https://huggingface.co/spaces/wikan1602/thesis-chatbot)

---

### [Video atau GIF demo aplikasi Anda di sini]

*(Sangat disarankan untuk merekam GIF singkat Anda menggunakan aplikasi ini dan menaruhnya di sini)*

---

## 🎯 Tentang Proyek Ini

Aplikasi ini memungkinkan pengguna untuk "bertanya" langsung ke naskah Tugas Akhir saya yang berjudul: *"Asosiasi Diabetes Melitus Tipe 2 dengan Brachial-Ankle Pulse Wave Velocity dan Komplians Pembuluh Darah"*.

Alih-alih membaca seluruh naskah, pengguna dapat mengajukan pertanyaan dalam bahasa Indonesia (misalnya: "Apa kesimpulan penelitian ini?") dan mendapatkan jawaban yang relevan, lengkap dengan sumbernya.

## 🛠️ Tumpukan Teknologi (Tech Stack)

* **Framework Aplikasi:** Streamlit
* **Orkestrasi LLM:** LangChain
* **Model LLM:** Groq (menggunakan `groq/compound`)
* **Database Vektor:** ChromaDB
* **Model Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
* **Hosting:** Hugging Face Spaces (via Docker)

## 🔧 Menjalankan Secara Lokal

1.  *Clone* repositori ini:
    ```bash
    git clone [https://github.com/username-anda/thesis-chatbot.git](https://github.com/username-anda/thesis-chatbot.git)
    cd thesis-chatbot
    ```
2.  Buat dan aktifkan *virtual environment*:
    ```bash
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
3.  Instal *dependencies*:
    ```bash
    pip install -r requirements.txt
    ```
4.  (Anda perlu membuat database `db_ta/` Anda sendiri menggunakan `make_db.py` dan `ta_teks.txt`).

5.  Buat file `.env` dan tambahkan `GROQ_API_KEY` Anda.

6.  Jalankan aplikasi:
    ```bash
    streamlit run app.py
    ```
