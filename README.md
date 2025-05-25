# Aplikasi Analisis dan Prediksi Harga Saham Proyek Akhir

Aplikasi web interaktif yang dibangun dengan Streamlit untuk menganalisis data historis harga saham, melatih berbagai model machine learning (LSTM, CNN, GRU) untuk prediksi harga, melakukan hyperparameter tuning otomatis, dan membandingkan performa model. Aplikasi ini juga dilengkapi dengan fitur caching model untuk mempercepat pemuatan dan otomasi pelatihan mingguan menggunakan GitHub Actions.

## Fitur Utama

* **Pengambilan Data Dinamis:** Mengambil data harga saham historis terbaru dari Yahoo Finance.
* **Penyimpanan Data Lokal:** Menggunakan SQLite untuk menyimpan data saham yang telah diambil.
* **Analisis Data Eksploratif (EDA):** Menampilkan statistik deskriptif, distribusi harga dan volume, heatmap korelasi, dan grafik time series bulanan.
* **Pelatihan Model Fleksibel:**
    * Pilihan model: LSTM, CNN, dan GRU.
    * Opsi untuk membandingkan performa ketiga model secara bersamaan.
* **Hyperparameter Tuning Otomatis:** Menggunakan KerasTuner (Hyperband) untuk mencari parameter optimal model.
* **Evaluasi Model Komprehensif:** Metrik yang ditampilkan meliputi Test RMSE, Test MAPE, dan Test R2 Score.
* **Visualisasi Kurva Loss:** Menampilkan kurva Training Loss vs Validation Loss saat model dilatih dari awal.
* **Prediksi Harga Masa Depan:** Memprediksi harga saham untuk 7 hari ke depan.
* **Caching Model & Scaler:** Menyimpan model dan scaler yang sudah dilatih untuk mempercepat pemuatan pada sesi berikutnya.
* **Opsi Paksa Latih Ulang:** Pengguna dapat memilih untuk melatih ulang model meskipun model tersimpan sudah ada.
* **Otomasi Pelatihan Mingguan:** Menggunakan GitHub Actions untuk melatih ulang model secara otomatis setiap minggu dan menyimpan versi terbaru.

## Teknologi yang Digunakan

* **Bahasa Pemrograman:** Python 3.9+
* **Framework Aplikasi Web:** Streamlit
* **Analisis & Manipulasi Data:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, TensorFlow (Keras), KerasTuner
* **Visualisasi Data:** Plotly
* **Pengambilan Data Keuangan:** yfinance
* **Database Lokal:** SQLAlchemy (untuk SQLite)
* **Penyimpanan Objek Python:** Joblib
* **Version Control:** Git & GitHub
* **Otomasi CI/CD:** GitHub Actions

## Cara Menjalankan Proyek di Perangkat Lain (Setup Lokal)

Untuk menjalankan proyek ini di komputer Anda atau perangkat lain, ikuti langkah-langkah berikut:

1.  **Prasyarat:**
    * Pastikan Anda memiliki **Git** terinstal ([cara instal Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)).
    * Pastikan Anda memiliki **Python 3.9 atau lebih tinggi** terinstal ([cara instal Python](https://www.python.org/downloads/)).
    * Pastikan `pip` (Python package installer) juga terinstal dan terbarui.

2.  **Clone Repositori:**
    Buka terminal atau command prompt Anda, lalu jalankan perintah berikut untuk mengunduh kode proyek:
    ```bash
    git clone [https://github.com/curzedb/prediksisaham_jafarrahadian.git](https://github.com/curzedb/prediksisaham_jafarrahadian.git)
    ```

3.  **Masuk ke Direktori Proyek:**
    ```bash
    cd prediksisaham_jafarrahadian 
    ```
    (Sesuaikan `prediksisaham_jafarrahadian` dengan nama folder proyek Anda jika berbeda)

4.  **Buat dan Aktifkan Virtual Environment:**
    Sangat disarankan untuk menggunakan lingkungan virtual agar dependensi proyek terisolasi.
    ```bash
    python -m venv venv 
    ```
    Kemudian aktifkan:
    * **Windows (Command Prompt/PowerShell):**
        ```powershell
        .\venv\Scripts\activate
        ```
    * **macOS/Linux (Bash/Zsh):**
        ```bash
        source venv/bin/activate
        ```
    Anda akan melihat `(venv)` di awal baris terminal jika berhasil.

5.  **Instal Dependensi:**
    Semua library yang dibutuhkan sudah tercantum dalam file `requirements.txt`. Instal dengan perintah:
    ```bash
    pip install -r requirements.txt
    ```
    Proses ini mungkin memakan waktu beberapa menit tergantung kecepatan internet Anda.

6.  **Jalankan Aplikasi Streamlit:**
    Setelah semua dependensi terinstal, jalankan aplikasi dengan perintah:
    ```bash
    streamlit run app.py
    ```
    Perintah ini akan membuka tab baru di browser web Anda yang menampilkan aplikasi prediksi harga saham.

## Struktur File Utama

* `app.py`: File utama aplikasi Streamlit, antarmuka pengguna dan logika orkestrasi.
* `data_handler.py`: Mengurus pengambilan data dari Yahoo Finance dan interaksi dengan database SQLite.
* `eda_handler.py`: Berisi fungsi-fungsi untuk menghasilkan visualisasi Analisis Data Eksploratif.
* `model_trainer.py`: Logika untuk membangun, melatih (termasuk hyperparameter tuning), dan mengevaluasi model machine learning (LSTM, CNN, GRU).
* `predictor.py`: Fungsi untuk melakukan prediksi harga saham 7 hari ke depan.
* `run_scheduled_training.py`: Skrip Python yang dirancang untuk dijalankan oleh GitHub Actions untuk pelatihan otomatis.
* `requirements.txt`: Daftar semua library Python yang dibutuhkan oleh proyek.
* `.github/workflows/weekly_training.yml`: File konfigurasi untuk GitHub Actions yang menjalankan pelatihan mingguan.
* `saved_models/`: Direktori tempat model dan scaler yang sudah dilatih disimpan (dibuat otomatis).
* `tuning_dir/`: Direktori tempat KerasTuner menyimpan hasil proses tuning (dibuat otomatis).
* `stock_data.db`: File database SQLite tempat data saham disimpan (dibuat otomatis).

## Otomatisasi dengan GitHub Actions

Proyek ini dikonfigurasi dengan GitHub Actions (`.github/workflows/weekly_training.yml`) untuk:
* Melatih ulang model (LSTM, CNN, GRU) untuk ticker saham yang ditentukan (`^JKSE`, `BBCA.JK`) secara otomatis setiap hari Minggu pukul 07:00 WIB.
* Proses pelatihan otomatis selalu menggunakan hyperparameter tuning untuk mencari konfigurasi terbaik.
* Model dan scaler yang baru dilatih akan disimpan dan di-commit kembali ke repositori.

Anda bisa memantau status eksekusi Actions di tab "Actions" pada halaman repositori GitHub Anda.

## Catatan Penting

* **Koneksi Internet:** Diperlukan untuk mengambil data dari Yahoo Finance.
* **Pelatihan Pertama Kali:** Saat menjalankan analisis untuk ticker baru atau saat memaksa latih ulang (terutama dengan hyperparameter tuning), prosesnya mungkin memakan waktu cukup lama. Model dan scaler akan disimpan setelahnya untuk mempercepat pemuatan di sesi berikutnya.
* **Folder `tuning_dir`:** Dibuat oleh KerasTuner. Anda bisa menghapusnya jika ingin memulai tuning dari awal, tetapi ini juga berarti proses tuning akan lebih lama.
* **Folder `saved_models` dan `stock_data.db`:** Akan dibuat secara otomatis saat aplikasi pertama kali dijalankan dan menyimpan data atau model.

## Lisensi

MIT License

---
Ditulis oleh: Jafar Rahadian (curzedb)