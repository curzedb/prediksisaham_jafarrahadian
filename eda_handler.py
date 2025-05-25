# eda_handler.py
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io

def generate_df(df):
    return df

def generate_descriptive_stats(df):
    """Menghasilkan statistik deskriptif dari dataframe."""
    return df.describe()

def generate_data_info(df):
    """Menghasilkan info dataframe (tipe data, non-null)."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()

def generate_price_plots(df):
    """Membuat subplot untuk Histogram dan Box Plot."""
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=('Histogram Harga Penutupan', 'Box Plot Harga Penutupan', 
                                        'Histogram Volume', 'Box Plot Volume'))

    # Histogram Harga Penutupan
    fig.add_trace(go.Histogram(x=df['close'], name='Close Price'), row=1, col=1)
    
    # Box Plot Harga Penutupan
    fig.add_trace(go.Box(y=df['close'], name='Close Price'), row=1, col=2)

    # Histogram Volume
    fig.add_trace(go.Histogram(x=df['volume'], name='Volume'), row=2, col=1)

    # Box Plot Volume
    fig.add_trace(go.Box(y=df['volume'], name='Volume'), row=2, col=2)

    fig.update_layout(height=600, title_text="<b>Distribusi Harga dan Volume</b>", showlegend=False)
    return fig

def generate_correlation_heatmap(df):
    """Membuat heatmap korelasi."""
    corr_matrix = df[['open', 'high', 'low', 'close', 'volume']].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                    title="<b>Heatmap Korelasi Antar Fitur</b>")
    return fig

def generate_monthly_timeseries(df):
    """Membuat plot time series harga penutupan rata-rata bulanan."""
    monthly_data = df['close'].resample('M').mean()
    fig = px.line(x=monthly_data.index, y=monthly_data.values,
                  title="<b>Grafik Time Series Harga Penutupan (Rata-rata Bulanan)</b>",
                  labels={'x': 'Tahun', 'y': 'Rerata Harga Close'})
    fig.update_traces(line=dict(color='lightblue', width=2))
    return fig