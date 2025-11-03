import time
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
# KITA KEMBALIKAN KE RECURSIVE, TAPI DENGAN SEPARATOR KHUSUS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- Konfigurasi ---
NAMA_FILE_TXT = "ta_teks_V1.1.txt" # Pastikan nama file ini sesuai
NAMA_FOLDER_DB = "db_thesis"  #
MODEL_EMBEDDING = "paraphrase-multilingual-MiniLM-L12-v2" #
# --------------------

def main():
    print(f"Memulai proses indexing untuk {NAMA_FILE_TXT}...")
    
    # 1. Load (Muat Dokumen)
    print(f"[1/4] Memuat dokumen '{NAMA_FILE_TXT}'...")
    loader = TextLoader(NAMA_FILE_TXT, encoding="utf-8") #
    dokumen = loader.load()
    print(f"Total {len(dokumen)} dokumen berhasil dimuat.")

    # 2. Split (Potong Teks) - INI BAGIAN YANG DIPERBARUI
    print("[2/4] Memotong dokumen menjadi potongan-potongan (chunks)...")
    
    # Daftar pemisah, dari prioritas tertinggi ke terendah
    # Ini akan memotong di "---" dulu, baru memotong sisanya
    # jika masih terlalu panjang (lebih dari 1000 karakter).
    my_separators = [
        "---",  # Pemisah logis utama kita
        "\n\n", # Baris kosong ganda (antar paragraf)
        "\n",   # Baris baru
        " ",    # Spasi
        ""      # Karakter
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, #
        chunk_overlap=200, # Kita kembalikan overlap agar konteks antar chunk terjaga
        separators=my_separators # Menggunakan daftar prioritas kita
    )
    
    potongan_teks = text_splitter.split_documents(dokumen)
    print(f"Dokumen dipotong menjadi {len(potongan_teks)} potongan teks.")

    # 3. Embed (Buat Model Embedding)
    print(f"[3/4] Menyiapkan model embedding '{MODEL_EMBEDDING}'...")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    ) #

    # 4. Store (Simpan ke Database Vektor)
    print(f"[4/4] Membuat dan menyimpan database vektor ke '{NAMA_FOLDER_DB}'...")
    start_time = time.time()
    
    db = Chroma.from_documents(
        potongan_teks, 
        embeddings, 
        persist_directory=NAMA_FOLDER_DB
    ) #
    
    end_time = time.time()
    print("\n--- Selesai! ---")
    print(f"Database vektor berhasil dibuat dan disimpan di folder '{NAMA_FOLDER_DB}'.")
    print(f"Total waktu: {end_time - start_time:.2f} detik.")

if __name__ == "__main__":
    main()
