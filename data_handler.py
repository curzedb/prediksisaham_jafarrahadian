# data_handler.py
import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine
import re # Import library regular expression

# Membuat koneksi ke database SQLite. File 'stock_data.db' akan dibuat otomatis.
engine = create_engine('sqlite:///stock_data.db')

def sanitize_for_table_name(ticker_symbol):
    """Membersihkan simbol ticker agar aman untuk dijadikan nama tabel."""
    # Hanya menyisakan huruf dan angka, mengubah semuanya menjadi lowercase
    return re.sub(r'[^a-zA-Z0-9]', '', ticker_symbol).lower()

def fetch_and_store_data(ticker_symbol):
    """Mengambil data historis dari Yahoo Finance dan menyimpannya ke database."""
    print(f"Mengambil data untuk {ticker_symbol}...")
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="max", actions=False)
    
    if df.empty:
        print(f"Tidak ada data ditemukan untuk simbol {ticker_symbol}")
        return None, None

    df = df.reset_index()
    # Mengganti nama kolom agar konsisten
    df = df.rename(columns={
        'Date': 'date', 'Open': 'open', 'High': 'high', 
        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    
    # Memastikan hanya kolom-kolom standar yang ada
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            # Jika ada kolom yang hilang, tambahkan dengan nilai 0 atau NaN
            df[col] = 0 
    df = df[required_cols] # Urutkan dan pilih hanya kolom yang diperlukan

    # Konversi timezone-aware datetime ke timezone-naive date
    if pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = df['date'].dt.date
    
    # Gunakan fungsi sanitasi untuk membuat nama tabel yang aman
    table_name = sanitize_for_table_name(ticker_symbol)
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print(f"Data berhasil disimpan ke tabel '{table_name}'.")
    return table_name

def load_data_from_db(table_name):
    """Memuat data dari tabel spesifik di database."""
    # Sanitasi nama tabel sebelum memuat untuk konsistensi
    safe_table_name = sanitize_for_table_name(table_name)
    print(f"Memuat data dari tabel {safe_table_name}...")
    try:
        df = pd.read_sql(f"SELECT * FROM {safe_table_name}", engine, index_col='date', parse_dates=['date'])
        
        # Penanganan Missing Values
        if df.isnull().sum().any():
            df = df.interpolate(method='linear')
            df = df.ffill().bfill()
            print("Missing values telah ditangani.")
            
        # Hapus duplikat berdasarkan index (tanggal)
        df = df[~df.index.duplicated(keep='first')]
        print("Data duplikat telah dihapus.")
        
        return df
    except Exception as e:
        print(f"Error memuat data dari DB: {e}")
        return None