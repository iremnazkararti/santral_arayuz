import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# 📌 Dashboard Ayarları
st.set_page_config(page_title="⚡ Enerji Üretim Tahmin Dashboard", layout="wide")

# **MAPE Hesaplama Fonksiyonu** - Hata önleme dahil
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = y_true.replace(0, np.nan).dropna()  # 0 olanları NaN yap, sonra temizle
    y_pred = y_pred[:len(y_true)]  # Uzunluk eşitle
    if len(y_true) == 0 or len(y_pred) == 0:  # Eğer hiç veri kalmadıysa NaN döndür
        return np.nan
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@st.cache_data
def load_data():
    file_path = "duzenlenmis_zaman_serisi.csv"  # Dosya yolunu burada güncelleyin
    try:
        # Veriyi yükle
        df = pd.read_csv(file_path)
        df.rename(columns={"Unnamed: 0": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])  # Tarih sütununu dönüştür
        df.set_index("date", inplace=True)
        df = df.resample("D").mean().interpolate()  # Eksik verileri doldur
        return df
    except Exception as e:
        st.error(f"📛 Veri yüklenirken hata oluştu: {e}")
        return pd.DataFrame()  # Hata durumunda boş dataframe döndür

# Veriyi yükle
df = load_data()

# 📌 Sidebar Filtreleri
st.sidebar.header("⚙ Filtreler")
start_date = st.sidebar.date_input("Başlangıç Tarihi", df.index.min().date())
end_date = st.sidebar.date_input("Bitiş Tarihi", df.index.max().date())
df = df.loc[start_date:end_date]

# Enerji türü seçimi
energy_type = st.sidebar.selectbox("Tahmin Yapılacak Enerji Türü Seçin:",
                                   ["Teiaş", "Net", "Gross", "BOP"])

# İlgili sütunları seç
if energy_type == "Teiaş":
    selected_columns = ["sinem_guc_teias", "deniz_guc_teias"]
elif energy_type == "Net":
    selected_columns = ["sinem_guc_net", "deniz_guc_net"]
elif energy_type == "Gross":
    selected_columns = ["sinem_guc_gross", "deniz_guc_gross"]
else:  # BOP
    selected_columns = ["sinem_guc_bop", "deniz_guc_bop"]

# 📌 Dashboard Görselleştirmeleri
st.sidebar.header("⚙ Enerji Türü ve Filtreler")

view_option = st.sidebar.radio(
    "🔍 Dashboard Seç:", 
    ["📂 Veri Seti", "📊 Zaman Serisi", "🔮 Enerji Üretim Tahminleri"]
)

# 📂 *1️⃣ Veri Seti*
if view_option == "📂 Veri Seti":
    st.subheader("📂 Veri Seti")
    st.dataframe(df.head(50))
    st.write("Seçilen tarih aralığında veri setinden ilk 50 satır.")

    # KPI Kartları
    col1, col2 = st.columns(2)
    col1.metric("🔹 Ortalama Sinem Üretimi (MW)", round(df[selected_columns[0]].mean(), 2))
    col2.metric("🔹 Ortalama Deniz Üretimi (MW)", round(df[selected_columns[1]].mean(), 2))

# 📊 *2️⃣ Zaman Serisi Analizi*
elif view_option == "📊 Zaman Serisi":
    st.subheader("📊 Enerji Üretimi Zaman Serisi")
    fig = go.Figure()

    # Gerçek veriler için Sinem ve Deniz değerleri
    fig.add_trace(go.Scatter(x=df.index, y=df[selected_columns[0]], mode='lines', name=f"Gerçek {selected_columns[0]}", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df[selected_columns[1]], mode='lines', name=f"Gerçek {selected_columns[1]}", line=dict(color='green')))
    
    st.plotly_chart(fig, use_container_width=True)

    # Haftalık, Aylık ve Yıllık Ortalamaları hesaplayalım
    df_weekly = df.resample('W').mean()
    df_monthly = df.resample('M').mean()
    df_yearly = df.resample('A').mean()

    # Haftalık, Aylık, Yıllık Görselleştirme
    st.subheader("📅 Haftalık, Aylık ve Yıllık Enerji Üretimi Grafikleri")

    # Haftalık Grafik
    fig_weekly = go.Figure()
    fig_weekly.add_trace(go.Scatter(x=df_weekly.index, y=df_weekly[selected_columns[0]], mode='lines', name=f"Haftalık Gerçek {selected_columns[0]}", line=dict(color='blue')))
    fig_weekly.add_trace(go.Scatter(x=df_weekly.index, y=df_weekly[selected_columns[1]], mode='lines', name=f"Haftalık Gerçek {selected_columns[1]}", line=dict(color='green')))
    st.plotly_chart(fig_weekly, use_container_width=True)

    # Aylık Grafik
    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly[selected_columns[0]], mode='lines', name=f"Aylık Gerçek {selected_columns[0]}", line=dict(color='blue')))
    fig_monthly.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly[selected_columns[1]], mode='lines', name=f"Aylık Gerçek {selected_columns[1]}", line=dict(color='green')))
    st.plotly_chart(fig_monthly, use_container_width=True)

    # Yıllık Grafik
    fig_yearly = go.Figure()
    fig_yearly.add_trace(go.Scatter(x=df_yearly.index, y=df_yearly[selected_columns[0]], mode='lines', name=f"Yıllık Gerçek {selected_columns[0]}", line=dict(color='blue')))
    fig_yearly.add_trace(go.Scatter(x=df_yearly.index, y=df_yearly[selected_columns[1]], mode='lines', name=f"Yıllık Gerçek {selected_columns[1]}", line=dict(color='green')))
    st.plotly_chart(fig_yearly, use_container_width=True)

# 🔮 *3️⃣ Enerji Üretim Tahminleri*
elif view_option == "🔮 Enerji Üretim Tahminleri":
    st.subheader("🔮 Enerji Üretim Tahminleri")
    model_option = st.selectbox("Tahmin Modeli Seç:", ["XGBoost"])
    days = st.slider("Tahmin İçin Gün Sayısı", 30, 365, 100)

    if st.button("📈 Tahmin Başlat"):
        with st.spinner("Tahmin yapılıyor..."):
            df_daily = df.resample("D").mean()
            forecast_values_sinem = None
            forecast_values_deniz = None

            # XGBoost Modeli
            if model_option == "XGBoost":
                df_xgb = df_daily.copy()
                for lag in range(1, 8):
                    df_xgb[f"{selected_columns[0]}_lag{lag}"] = df_xgb[selected_columns[0]].shift(lag)
                    df_xgb[f"{selected_columns[1]}_lag{lag}"] = df_xgb[selected_columns[1]].shift(lag)
                df_xgb.dropna(inplace=True)
                features = [col for col in df_xgb.columns if "lag" in col]
                X_train, X_test, y_train_sinem, y_test_sinem = train_test_split(df_xgb[features], df_xgb[selected_columns[0]], test_size=0.2, shuffle=False)
                X_train, X_test, y_train_deniz, y_test_deniz = train_test_split(df_xgb[features], df_xgb[selected_columns[1]], test_size=0.2, shuffle=False)
                xgb_model_sinem = xgb.XGBRegressor(n_estimators=100)
                xgb_model_deniz = xgb.XGBRegressor(n_estimators=100)
                xgb_model_sinem.fit(X_train, y_train_sinem)
                xgb_model_deniz.fit(X_train, y_train_deniz)
                forecast_values_sinem = xgb_model_sinem.predict(df_xgb[features].iloc[-days:])
                forecast_values_deniz = xgb_model_deniz.predict(df_xgb[features].iloc[-days:])

            forecast_dates = pd.date_range(df_daily.index[-1] + pd.DateOffset(1), periods=days, freq="D")

            # Gerçek ve tahmin edilen üretimlerin grafikle gösterimi
            fig = go.Figure()

            # Gerçek veriler için Sinem ve Deniz değerleri
            fig.add_trace(go.Scatter(x=df_daily.index, y=df[selected_columns[0]], mode='lines', name=f"Gerçek {selected_columns[0]}", line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=df_daily.index, y=df[selected_columns[1]], mode='lines', name=f"Gerçek {selected_columns[1]}", line=dict(color='green')))
            
            # Tahmin edilen veriler için Sinem ve Deniz değerleri
            fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values_sinem, mode='lines', name=f"{model_option} Tahmini {selected_columns[0]}", line=dict(color='darkred')))
            fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values_deniz, mode='lines', name=f"{model_option} Tahmini {selected_columns[1]}", line=dict(color='orange')))
            
            st.plotly_chart(fig, use_container_width=True)

            # Doğruluk oranları
            mape_sinem = mean_absolute_percentage_error(df_daily[selected_columns[0]].iloc[-days:], forecast_values_sinem)
            mape_deniz = mean_absolute_percentage_error(df_daily[selected_columns[1]].iloc[-days:], forecast_values_deniz)

            accuracy_sinem = 100 - round(mape_sinem, 2) if not np.isnan(mape_sinem) else "Veri Eksik"
            accuracy_deniz = 100 - round(mape_deniz, 2) if not np.isnan(mape_deniz) else "Veri Eksik"

            # Doğruluk oranlarını göster
            st.markdown(f"### 🎯 Model Doğruluk Oranı")
            col1, col2 = st.columns(2)
            col1.metric(f"🔵 {model_option} Sinem Tahmin Doğruluğu", f"% {accuracy_sinem}")
            col2.metric(f"🟠 {model_option} Deniz Tahmin Doğruluğu", f"% {accuracy_deniz}")   
