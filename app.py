# app.py (VERSI BERSIH FINAL)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import joblib
from tensorflow.keras.models import load_model

# Impor fungsi-fungsi terpisah
from data_handler import fetch_and_store_data, load_data_from_db
from eda_handler import generate_df, generate_data_info, generate_descriptive_stats, generate_monthly_timeseries, generate_price_plots, generate_correlation_heatmap
from model_trainer import train_and_evaluate_lstm, train_and_evaluate_cnn, train_and_evaluate_gru
from predictor import predict_future_lstm, predict_future_cnn, predict_future_gru

# --- Konfigurasi Halaman & Direktori ---
st.set_page_config(layout="wide", page_title="Prediksi Harga Saham", page_icon="📈")
if not os.path.exists('saved_models'):
    os.makedirs('saved_models')
if not os.path.exists('tuning_dir'):
    os.makedirs('tuning_dir')

@st.cache_data(ttl=3600) # Cache data selama 1 jam
def get_data(ticker):
    _ = fetch_and_store_data(ticker) # Selalu coba fetch, tapi load_data_from_db yang menentukan
    df = load_data_from_db(ticker)
    return df

# --- UI Sidebar ---
st.title("📈 Projek Analisis & Prediksi (Saham dan Koin Kripto)")
st.sidebar.header("⚙️ Panel Kontrol")
ticker_symbol = st.sidebar.text_input("Masukkan Simbol Ticker")
model_options = ["Bandingkan Semua Model", "LSTM", "CNN", "GRU"]
model_choice = st.sidebar.selectbox("Pilih Opsi", model_options)
do_tuning = st.sidebar.checkbox("Lakukan Hyperparameter Tuning (Proses berjalan lama)")
force_retrain = st.sidebar.checkbox("Paksa Latih Ulang Model (abaikan cache model)")
run_button = st.sidebar.button("Jalankan Analisis", type="primary")

# --- Fungsi Plotting ---
def create_single_model_plots(df, prediction_data, metrics, model_name, future_preds):
    fig1 = go.Figure()
    if prediction_data and len(prediction_data[0]) > 0:
        train_predict, test_predict, _, time_step = prediction_data
        original_prices = df['close'].values[-1000:]
        dates = df.index[-1000:]
        train_plot = np.empty_like(original_prices)
        train_plot[:] = np.nan
        train_plot[time_step:len(train_predict) + time_step] = train_predict.flatten()
        test_plot = np.empty_like(original_prices)
        test_plot[:] = np.nan
        test_start_index = len(train_predict) + (time_step * 2) + 1
        test_plot[test_start_index : test_start_index + len(test_predict)] = test_predict.flatten()
        fig1.add_trace(go.Scatter(x=dates, y=original_prices, mode='lines', name='Harga Asli', line=dict(color='deepskyblue')))
        fig1.add_trace(go.Scatter(x=dates, y=train_plot, mode='lines', name='Prediksi Latih', line=dict(color='orange')))
        fig1.add_trace(go.Scatter(x=dates, y=test_plot, mode='lines', name='Prediksi Uji', line=dict(color='limegreen')))
    fig1.update_layout(title=f'<b>Perbandingan Harga Asli vs Prediksi ({model_name})</b>', xaxis_title='Tanggal', yaxis_title='Harga (IDR)')

    fig2 = None
    if 'loss' in metrics and metrics.get('loss') and len(metrics['loss']) > 0:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(y=metrics['loss'], mode='lines', name='Training Loss'))
        fig2.add_trace(go.Scatter(y=metrics['val_loss'], mode='lines', name='Validation Loss'))
        fig2.update_layout(title=f'<b>Kurva Loss Model {model_name}</b>', xaxis_title='Epoch', yaxis_title='Loss')

    fig3 = go.Figure()
    if df is not None and not df.empty:
        last_date = df.index[-1]
        future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
        fig3.add_trace(go.Scatter(x=df.index[-30:], y=df['close'][-30:], mode='lines', name='Data 30 Hari Terakhir'))
        fig3.add_trace(go.Scatter(x=future_dates, y=future_preds, mode='lines', name='Prediksi 7 Hari ke Depan', line=dict(color='red', dash='dash')))
    fig3.update_layout(title=f'<b>Prediksi 7 Hari ke Depan ({model_name})</b>', xaxis_title='Tanggal', yaxis_title='Harga (IDR)')
    return fig1, fig2, fig3

def create_comparison_plot(df, all_predictions):
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index[-30:], y=df['close'][-30:], mode='lines', name='Data 30 Hari Terakhir', line=dict(color='lightblue', width=3)))
    colors = {'LSTM': 'blue', 'CNN': 'red', 'GRU': 'green'}
    for model_name, preds in all_predictions.items():
        fig.add_trace(go.Scatter(x=future_dates, y=preds, mode='lines', name=f'Prediksi {model_name}', line=dict(color=colors[model_name], dash='dash')))
    fig.update_layout(title='<b>Perbandingan Prediksi 8 Hari ke Depan (Semua Model)</b>', xaxis_title='Tanggal', yaxis_title='Harga (IDR)')
    return fig

# --- Logika Utama Aplikasi ---
def run_single_model(df, model_name, force_retrain_flag, do_tuning_flag):
    train_functions = {"LSTM": train_and_evaluate_lstm, "CNN": train_and_evaluate_cnn, "GRU": train_and_evaluate_gru}
    pred_functions = {"LSTM": predict_future_lstm, "CNN": predict_future_cnn, "GRU": predict_future_gru}
    train_func = train_functions[model_name]
    pred_func = pred_functions[model_name]
    
    sanitized_ticker = "".join(filter(str.isalnum, ticker_symbol)).lower()
    model_path = f'saved_models/{sanitized_ticker}_{model_name.lower()}_model.keras'
    scaler_path = f'saved_models/{sanitized_ticker}_{model_name.lower()}_scaler.joblib'

    should_train = force_retrain_flag or do_tuning_flag or not os.path.exists(model_path) or not os.path.exists(scaler_path)

    if should_train:
        with st.spinner(f"Melatih model {model_name}... (Tuning: {'Ya' if do_tuning_flag else 'Tidak'})"):
            model, scaler, metrics, plot_data, last_days_scaled, best_hp = train_func(df, do_tuning=do_tuning_flag)
            model.save(model_path)
            joblib.dump(scaler, scaler_path)
            st.success(f"Model {model_name} selesai dilatih dan disimpan.")
            if best_hp:
                st.success(f"Parameter terbaik untuk {model_name}: {best_hp.values}")
    else:
        st.info(f"Memuat model {model_name} yang sudah ada...")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        metrics = {}
        plot_data = ([], [], [], 15)
        last_days_scaled = scaler.transform(df[['close']].values[-15:])

    future_preds = pred_func(model, scaler, last_days_scaled)
    
    # Tampilkan hasil
    if metrics:
        st.header(f"Hasil Evaluasi Model {model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Test RMSE", f"{metrics.get('Test RMSE', 0):.2f}")
        col2.metric("Test MAPE", f"{metrics.get('Test MAPE', 0):.2f}%")
        col3.metric("Test R2", f"{metrics.get('Test R2', 0):.4f}")
    
    fig1, fig2, fig3 = create_single_model_plots(df, plot_data, metrics, model_name, future_preds)
    st.header(f"Visualisasi Hasil Prediksi - {model_name}")
    st.plotly_chart(fig1, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(fig3, use_container_width=True)
    with col_b:
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Kurva loss hanya ditampilkan saat model dilatih dari awal (centang 'Paksa Latih Ulang Model').")
    
    return metrics, future_preds

if run_button:
    df = get_data(ticker_symbol)
    if df is not None:
        st.success(f"Data untuk {ticker_symbol} berhasil dimuat.")
        with st.expander("📊 Lihat Analisis Data Eksploratif (EDA)"):
            st.subheader("Seluruh Data:")
            st.dataframe(generate_df(df))
            st.subheader("Statistik Deskriptif")
            st.dataframe(generate_descriptive_stats(df))    
            st.subheader("Informasi Tipe Data")
            st.text(generate_data_info(df))        
            st.plotly_chart(generate_monthly_timeseries(df), use_container_width=True)
            st.plotly_chart(generate_price_plots(df), use_container_width=True)
            st.plotly_chart(generate_correlation_heatmap(df), use_container_width=True)

        if model_choice == "Bandingkan Semua Model":
            all_metrics_summary = {}
            all_future_preds_summary = {}
            
            st.markdown("---")
            metrics_lstm, future_preds_lstm = run_single_model(df, "LSTM", force_retrain, do_tuning)
            all_metrics_summary["LSTM"] = metrics_lstm
            all_future_preds_summary["LSTM"] = future_preds_lstm
            st.markdown("---")
            
            metrics_cnn, future_preds_cnn = run_single_model(df, "CNN", force_retrain, do_tuning)
            all_metrics_summary["CNN"] = metrics_cnn
            all_future_preds_summary["CNN"] = future_preds_cnn
            st.markdown("---")
            
            metrics_gru, future_preds_gru = run_single_model(df, "GRU", force_retrain, do_tuning)
            all_metrics_summary["GRU"] = metrics_gru
            all_future_preds_summary["GRU"] = future_preds_gru
            st.markdown("---")
            
            metrics_data_summary = {name: metrics for name, metrics in all_metrics_summary.items() if metrics}
            if metrics_data_summary:
                metrics_df_summary = pd.DataFrame(metrics_data_summary).T
                st.header("Tabel Perbandingan Metrik Evaluasi Akhir")
                if not metrics_df_summary.empty and all(col in metrics_df_summary.columns for col in ['Test RMSE', 'Test MAPE', 'Test R2']):
                    st.dataframe(metrics_df_summary[['Test RMSE', 'Test MAPE', 'Test R2']].style.highlight_min(subset=['Test RMSE', 'Test MAPE'], color='lightgreen').highlight_max(subset=['Test R2'], color='lightgreen').format("{:.4f}"))
                else:
                    st.warning("Tidak semua metrik evaluasi tersedia untuk perbandingan.")
            
            comparison_fig_summary = create_comparison_plot(df, all_future_preds_summary)
            st.header("Perbandingan Prediksi 7 Hari ke Depan (Semua Model)")
            st.plotly_chart(comparison_fig_summary, use_container_width=True)
            
        else:
            run_single_model(df, model_choice, force_retrain, do_tuning)
    else:
        st.error(f"Gagal mengambil atau memproses data untuk {ticker_symbol}. Periksa konsol untuk detail.")
else:
    st.info("Pilih opsi di sidebar dan klik 'Jalankan Analisis' untuk memulai.")
