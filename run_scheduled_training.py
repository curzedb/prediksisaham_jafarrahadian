# run_scheduled_training.py
import argparse
import os
import joblib
import pandas as pd # Pastikan pandas diimpor jika belum
from data_handler import fetch_and_store_data, load_data_from_db
from model_trainer import train_and_evaluate_lstm, train_and_evaluate_cnn, train_and_evaluate_gru # Impor fungsi spesifik

def main(ticker, model_type_to_train):
    print(f"--- Memulai Pelatihan Terjadwal untuk {ticker} dengan model {model_type_to_train} ---")
    
    _ = fetch_and_store_data(ticker) # Fetch dan simpan, tapi kita akan load lagi untuk konsistensi
    df = load_data_from_db(ticker)
    
    if df is None:
        print(f"Gagal memuat data untuk {ticker}. Proses dihentikan.")
        return

    train_functions_map = {
        "LSTM": train_and_evaluate_lstm,
        "CNN": train_and_evaluate_cnn,
        "GRU": train_and_evaluate_gru
    }

    if model_type_to_train not in train_functions_map:
        print(f"Tipe model tidak valid: {model_type_to_train}. Pilih dari LSTM, CNN, GRU.")
        return
        
    train_func = train_functions_map[model_type_to_train]

    # Selalu lakukan tuning untuk pelatihan terjadwal untuk mendapatkan model terbaik
    model, scaler, metrics, _, _, best_hp = train_func(df, do_tuning=True) 
    
    sanitized_ticker = "".join(filter(str.isalnum, ticker)).lower()
    model_file_name = f'{sanitized_ticker}_{model_type_to_train.lower()}_model.keras'
    scaler_file_name = f'{sanitized_ticker}_{model_type_to_train.lower()}_scaler.joblib'
    
    # Pastikan folder 'saved_models' ada
    if not os.path.exists('saved_models'):
        os.makedirs('saved_models')
        
    model_path = os.path.join('saved_models', model_file_name)
    scaler_path = os.path.join('saved_models', scaler_file_name)
    
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"Model {model_type_to_train} untuk {ticker} berhasil dilatih ulang dan disimpan ke {model_path}")
    print(f"Metrik Uji: Test RMSE={metrics.get('Test RMSE', 0):.2f}, Test MAPE={metrics.get('Test MAPE', 0):.2f}%, Test R2={metrics.get('Test R2', 0):.4f}")
    if best_hp:
        print(f"Parameter terbaik yang ditemukan: {best_hp.values}")
    
    print(f"--- Pelatihan Terjadwal Selesai ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan pipeline pelatihan model saham.")
    parser.add_argument("--ticker", type=str, required=True, help="Simbol ticker saham, cth: ^JKSE")
    parser.add_argument("--model", type=str, default="LSTM", choices=["LSTM", "CNN", "GRU"], help="Tipe model yang akan dilatih.")
    args = parser.parse_args()
    
    main(args.ticker, args.model)