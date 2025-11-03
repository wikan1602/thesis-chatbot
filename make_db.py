import time
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# --- Konfigurasi ---
NAMA_FILE_TXT = "ta_teks.txt"
NAMA_FOLDER_DB = "db_ta"  # Nama folder untuk menyimpan database vektor
MODEL_EMBEDDING = "all-MiniLM-L6-v2" # Model embedding gratis & populer
# --------------------

def main():
    print(f"Memulai proses indexing untuk {NAMA_FILE_TXT}...")
    
    # 1. Load (Muat Dokumen)
    # Membaca file ta_teks.txt
    print(f"[1/4] Memuat dokumen '{NAMA_FILE_TXT}'...")
    loader = TextLoader(NAMA_FILE_TXT, encoding="utf-8")
    dokumen = loader.load()
    print(f"Total {len(dokumen)} dokumen berhasil dimuat.")

    # 2. Split (Potong Teks)
    # Memotong teks menjadi potongan-potongan kecil (1000 karakter per potong)
    # overlap=200 berarti 200 karakter terakhir dari chunk 1
    # akan diulang di awal chunk 2, agar konteks tidak terputus.
    print("[2/4] Memotong dokumen menjadi potongan-potongan (chunks)...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    potongan_teks = text_splitter.split_documents(dokumen)
    print(f"Dokumen dipotong menjadi {len(potongan_teks)} potongan teks.")

    # 3. Embed (Buat Model Embedding)
    # Menyiapkan model AI yang akan mengubah teks menjadi vektor (angka)
    # 'device='cpu'' berarti proses ini akan berjalan di CPU Anda.
    # (Jika punya GPU NVIDIA, ganti ke 'cuda')
    print(f"[3/4] Menyiapkan model embedding '{MODEL_EMBEDDING}'...")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_EMBEDDING,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # 4. Store (Simpan ke Database Vektor)
    # Proses ini akan mengubah semua potongan teks menjadi vektor
    # dan menyimpannya ke folder NAMA_FOLDER_DB
    print(f"[4/4] Membuat dan menyimpan database vektor ke '{NAMA_FOLDER_DB}'...")
    start_time = time.time()
    
    # Ini adalah proses intinya:
    # Chroma.from_documents akan:
    # 1. Mengambil setiap 'potongan_teks'
    # 2. Menggunakan 'embeddings' (model) untuk mengubahnya jadi vektor
    # 3. Menyimpan hasilnya di folder 'persist_directory'
    db = Chroma.from_documents(
        potongan_teks, 
        embeddings, 
        persist_directory=NAMA_FOLDER_DB
    )
    
    end_time = time.time()
    print("\n--- Selesai! ---")
    print(f"Database vektor berhasil dibuat dan disimpan di folder '{NAMA_FOLDER_DB}'.")
    print(f"Total waktu: {end_time - start_time:.2f} detik.")

if __name__ == "__main__":
    main()