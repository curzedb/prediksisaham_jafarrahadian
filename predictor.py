# predictor.py
import numpy as np

# FUNGSI PREDIKSI MASA DEPAN TERPISAH UNTUK SETIAP MODEL
def predict_future_lstm(model, scaler, initial_input, days_to_predict=8, time_step=15):
    """Memprediksi harga saham ke depan menggunakan model LSTM."""
    print("Memprediksi masa depan dengan LSTM...")
    
    temp_input = list(initial_input.flatten())
    predictions_output = []
    i = 0
    
    while i < days_to_predict:
        x_input = np.array(temp_input[-time_step:])
        x_input = x_input.reshape((1, time_step, 1))
        yhat = model.predict(x_input, verbose=0)
        
        # Tambahkan prediksi baru ke input untuk iterasi berikutnya
        temp_input.append(yhat[0][0])
        # Simpan prediksi untuk output akhir
        predictions_output.append(yhat[0][0])
        
        i += 1
            
    future_predictions = scaler.inverse_transform(np.array(predictions_output).reshape(-1, 1))
    return future_predictions.flatten()


def predict_future_cnn(model, scaler, initial_input, days_to_predict=8, time_step=15):
    """Memprediksi harga saham ke depan menggunakan model CNN."""
    print("Memprediksi masa depan dengan CNN...")
    
    temp_input = list(initial_input.flatten())
    predictions_output = []
    i = 0
    
    while i < days_to_predict:
        x_input = np.array(temp_input[-time_step:])
        x_input = x_input.reshape((1, time_step, 1))
        yhat = model.predict(x_input, verbose=0)
        
        # Tambahkan prediksi baru ke input untuk iterasi berikutnya
        temp_input.append(yhat[0][0])
        # Simpan prediksi untuk output akhir
        predictions_output.append(yhat[0][0])
        
        i += 1
            
    future_predictions = scaler.inverse_transform(np.array(predictions_output).reshape(-1, 1))
    return future_predictions.flatten()


def predict_future_gru(model, scaler, initial_input, days_to_predict=8, time_step=15):
    """Memprediksi harga saham ke depan menggunakan model GRU."""
    print("Memprediksi masa depan dengan GRU...")
    
    temp_input = list(initial_input.flatten())
    predictions_output = []
    i = 0
    
    while i < days_to_predict:
        x_input = np.array(temp_input[-time_step:])
        x_input = x_input.reshape((1, time_step, 1))
        yhat = model.predict(x_input, verbose=0)
        
        # Tambahkan prediksi baru ke input untuk iterasi berikutnya
        temp_input.append(yhat[0][0])
        # Simpan prediksi untuk output akhir
        predictions_output.append(yhat[0][0])
        
        i += 1
            
    future_predictions = scaler.inverse_transform(np.array(predictions_output).reshape(-1, 1))
    return future_predictions.flatten()