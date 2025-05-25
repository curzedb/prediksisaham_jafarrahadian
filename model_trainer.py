# model_trainer.py (dengan solusi overfitting)
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten, GRU, Dropout # <-- Import Dropout
from tensorflow.keras.callbacks import EarlyStopping # <-- Import EarlyStopping
from tensorflow.keras import regularizers # Opsional, jika ingin L1/L2
from sklearn.metrics import mean_squared_error, r2_score

# --- Fungsi Helper---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# --- Fungsi Pabrik Model (dengan Dropout) ---
def build_model(hp, model_type='LSTM', time_step=15):
    model = Sequential()
    hp_units = hp.Int('units', min_value=32, max_value=96, step=32) # Rentang unit mungkin bisa diperkecil
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.1, max_value=0.4, step=0.1) # Dropout rate yang bisa di-tuning
    
    if model_type == 'LSTM':
        model.add(LSTM(units=hp_units, input_shape=(time_step, 1), activation="relu", return_sequences=False)) # return_sequences=False jika hanya satu layer LSTM
        model.add(Dropout(hp_dropout_rate))
    elif model_type == 'GRU':
        model.add(GRU(units=hp_units, input_shape=(time_step, 1), activation='relu', return_sequences=False))
        model.add(Dropout(hp_dropout_rate))
    elif model_type == 'CNN':
        model.add(Conv1D(filters=hp.Int('filters_cnn', min_value=32, max_value=64, step=16), kernel_size=3, activation='relu', input_shape=(time_step, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(hp_dropout_rate)) # Dropout setelah Conv/Pool
        model.add(Flatten())
        model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=48, step=16), activation='relu'))
        model.add(Dropout(hp_dropout_rate)) # Dropout setelah Dense
        
    model.add(Dense(1))
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4]) # Learning rate mungkin bisa diperkecil
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mean_squared_error')
    return model

# --- Fungsi Pelatihan (dengan Early Stopping) ---
def common_train_logic(df, time_step, model_type, X_train, y_train, X_test, y_test, do_tuning):
    best_hyperparameters = None
    model = None
    history = None

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10, # Jumlah epoch untuk menunggu jika tidak ada perbaikan
        restore_best_weights=True, # Kembalikan ke bobot terbaik saat berhenti
        verbose=1
    )

    if do_tuning:
        tuner = kt.Hyperband(
            lambda hp: build_model(hp, model_type=model_type, time_step=time_step),
            objective='val_loss', max_epochs=50, factor=3, # Mungkin kurangi max_epochs untuk tuning
            directory='tuning_dir', project_name=f'{model_type.lower()}_tuning_v2'
        )
        tuner.search(X_train, y_train, epochs=70, validation_data=(X_test, y_test), callbacks=[early_stopping_callback])
        best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.get_best_models(num_models=1)[0]
        # Latih ulang model terbaik sedikit lebih lama untuk mendapatkan history yang bersih
        history = model.fit(X_train, y_train, epochs=90, batch_size=16, validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping_callback])
    else:
        model = build_model(kt.HyperParameters(), model_type=model_type, time_step=time_step)
        history = model.fit(X_train, y_train, epochs=110, batch_size=16, # Mungkin tambah epoch jika tidak tuning, biarkan early stopping bekerja
                            validation_data=(X_test, y_test), verbose=0, callbacks=[early_stopping_callback])
    
    return model, history, best_hyperparameters

# =============================================================================
# FUNGSI TERPISAH UNTUK SETIAP MODEL (MEMANGGIL common_train_logic)
# =============================================================================

def train_and_evaluate_lstm(df, time_step=15, do_tuning=False):
    tf.keras.backend.clear_session()
    print("Memulai proses untuk model LSTM...")
    close_data = df[['close']].values[-1000:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    train_data, test_data = scaled_data[0:int(len(scaled_data)*0.8), :], scaled_data[int(len(scaled_data)*0.8):len(scaled_data), :]
    X_train_lstm, y_train_lstm = create_dataset(train_data, time_step)
    X_test_lstm, y_test_lstm = create_dataset(test_data, time_step)
    X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], 1)
    X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], 1)

    model_lstm, history_lstm, best_hp_lstm = common_train_logic(df, time_step, 'LSTM', X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, do_tuning)

    train_predict_lstm = model_lstm.predict(X_train_lstm)
    test_predict_lstm = model_lstm.predict(X_test_lstm)
    train_predict_lstm = scaler.inverse_transform(train_predict_lstm)
    test_predict_lstm = scaler.inverse_transform(test_predict_lstm)
    original_y_train_lstm = scaler.inverse_transform(y_train_lstm.reshape(-1, 1))
    original_y_test_lstm = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))
    
    metrics = {
        'Train RMSE': np.sqrt(mean_squared_error(original_y_train_lstm, train_predict_lstm)),
        'Test RMSE': np.sqrt(mean_squared_error(original_y_test_lstm, test_predict_lstm)),
        'Train MAPE': mean_absolute_percentage_error(original_y_train_lstm, train_predict_lstm),
        'Test MAPE': mean_absolute_percentage_error(original_y_test_lstm, test_predict_lstm),
        'Train R2': r2_score(original_y_train_lstm, train_predict_lstm),
        'Test R2': r2_score(original_y_test_lstm, test_predict_lstm),
        'loss': history_lstm.history['loss'],
        'val_loss': history_lstm.history['val_loss']
    }

    last_days_scaled = scaled_data[-time_step:]
    return model_lstm, scaler, metrics, (train_predict_lstm, test_predict_lstm, scaled_data, time_step), last_days_scaled, best_hp_lstm

def train_and_evaluate_cnn(df, time_step=15, do_tuning=False):
    tf.keras.backend.clear_session()
    print("Memulai proses untuk model CNN...")
    close_data = df[['close']].values[-1000:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    train_data, test_data = scaled_data[0:int(len(scaled_data)*0.8), :], scaled_data[int(len(scaled_data)*0.8):len(scaled_data), :]
    X_train_cnn, y_train_cnn = create_dataset(train_data, time_step)
    X_test_cnn, y_test_cnn = create_dataset(test_data, time_step)
    X_train_cnn = X_train_cnn.reshape(X_train_cnn.shape[0], X_train_cnn.shape[1], 1)
    X_test_cnn = X_test_cnn.reshape(X_test_cnn.shape[0], X_test_cnn.shape[1], 1)

    model_cnn, history_cnn, best_hp_cnn = common_train_logic(df, time_step, 'CNN', X_train_cnn, y_train_cnn, X_test_cnn, y_test_cnn, do_tuning)
        
    train_predict_cnn = model_cnn.predict(X_train_cnn)
    test_predict_cnn = model_cnn.predict(X_test_cnn)
    train_predict_cnn = scaler.inverse_transform(train_predict_cnn)
    test_predict_cnn = scaler.inverse_transform(test_predict_cnn)
    original_y_train_cnn = scaler.inverse_transform(y_train_cnn.reshape(-1, 1))
    original_y_test_cnn = scaler.inverse_transform(y_test_cnn.reshape(-1, 1))

    metrics = {
        'Train RMSE': np.sqrt(mean_squared_error(original_y_train_cnn, train_predict_cnn)),
        'Test RMSE': np.sqrt(mean_squared_error(original_y_test_cnn, test_predict_cnn)),
        'Train MAPE': mean_absolute_percentage_error(original_y_train_cnn, train_predict_cnn),
        'Test MAPE': mean_absolute_percentage_error(original_y_test_cnn, test_predict_cnn),
        'Train R2': r2_score(original_y_train_cnn, train_predict_cnn),
        'Test R2': r2_score(original_y_test_cnn, test_predict_cnn),
        'loss': history_cnn.history['loss'],
        'val_loss': history_cnn.history['val_loss']
    }
    
    last_days_scaled = scaled_data[-time_step:]
    return model_cnn, scaler, metrics, (train_predict_cnn, test_predict_cnn, scaled_data, time_step), last_days_scaled, best_hp_cnn

def train_and_evaluate_gru(df, time_step=15, do_tuning=False):
    tf.keras.backend.clear_session()
    print("Memulai proses untuk model GRU...")
    close_data = df[['close']].values[-1000:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    train_data, test_data = scaled_data[0:int(len(scaled_data)*0.8), :], scaled_data[int(len(scaled_data)*0.8):len(scaled_data), :]
    X_train_gru, y_train_gru = create_dataset(train_data, time_step)
    X_test_gru, y_test_gru = create_dataset(test_data, time_step)
    X_train_gru = X_train_gru.reshape(X_train_gru.shape[0], X_train_gru.shape[1], 1)
    X_test_gru = X_test_gru.reshape(X_test_gru.shape[0], X_test_gru.shape[1], 1)

    model_gru, history_gru, best_hp_gru = common_train_logic(df, time_step, 'GRU', X_train_gru, y_train_gru, X_test_gru, y_test_gru, do_tuning)

    train_predict_gru = model_gru.predict(X_train_gru)
    test_predict_gru = model_gru.predict(X_test_gru)
    train_predict_gru = scaler.inverse_transform(train_predict_gru)
    test_predict_gru = scaler.inverse_transform(test_predict_gru)
    original_y_train_gru = scaler.inverse_transform(y_train_gru.reshape(-1, 1))
    original_y_test_gru = scaler.inverse_transform(y_test_gru.reshape(-1, 1))
    
    metrics = {
        'Train RMSE': np.sqrt(mean_squared_error(original_y_train_gru, train_predict_gru)),
        'Test RMSE': np.sqrt(mean_squared_error(original_y_test_gru, test_predict_gru)),
        'Train MAPE': mean_absolute_percentage_error(original_y_train_gru, train_predict_gru),
        'Test MAPE': mean_absolute_percentage_error(original_y_test_gru, test_predict_gru),
        'Train R2': r2_score(original_y_train_gru, train_predict_gru),
        'Test R2': r2_score(original_y_test_gru, test_predict_gru),
        'loss': history_gru.history['loss'],
        'val_loss': history_gru.history['val_loss']
    }

    last_days_scaled = scaled_data[-time_step:]
    return model_gru, scaler, metrics, (train_predict_gru, test_predict_gru, scaled_data, time_step), last_days_scaled, best_hp_gru