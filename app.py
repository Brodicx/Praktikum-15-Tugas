import streamlit as st
import pandas as pd
import numpy as np
import simpy
import random
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Simulasi Antrean Restoran",
    page_icon="🍽️",
    layout="wide"
)

# =====================
# Load Data & Parameter
# =====================
@st.cache_data
def load_data():
    """Load dataset dan hitung parameter dari data riil"""
    try:
        df = pd.read_csv('hotel_restaurant_orders.csv')
        df['OrderDate'] = pd.to_datetime(df['OrderDate'])
        df = df.sort_values('OrderDate').reset_index(drop=True)
        
        # Hitung interval kedatangan
        df['TimeDiff'] = df['OrderDate'].diff().dt.total_seconds() / 60
        df['TimeDiff'] = df['TimeDiff'].fillna(0)
        arrival_intervals = df['TimeDiff'][df['TimeDiff'] > 0].values
        
        # Estimasi waktu pelayanan
        service_time_mapping = {
            'Beverage': (2, 4),
            'Food': (3, 6)
        }
        df['EstServiceTime'] = df['MenuCategory'].map(lambda x: 
            random.uniform(*service_time_mapping.get(x, (2, 5)))
        )
        service_times = df['EstServiceTime'].values
        
        # Parameter dari data riil
        real_arrival_rate = min(np.mean(arrival_intervals), 20.0)
        real_service_min = np.percentile(service_times, 25)
        real_service_max = np.percentile(service_times, 75)
        
        return df, real_arrival_rate, real_service_min, real_service_max
    except FileNotFoundError:
        st.error("File dataset tidak ditemukan! Pastikan hotel_restaurant_orders.csv ada di direktori yang sama.")
        return None, 2.0, 2.0, 4.0

# Load data
df_data, REAL_ARRIVAL_RATE, REAL_SERVICE_MIN, REAL_SERVICE_MAX = load_data()

# =====================
# Parameter Tetap
# =====================
SIM_TIME = 480  # 8 jam operasional (menit)
RANDOM_SEED = 42

# =====================
# Fungsi Simulasi
# =====================
def run_simulation(num_cashiers, arrival_rate, service_min, service_max, sim_time, seed=42):
    """Menjalankan simulasi antrean restoran"""
    random.seed(seed)
    
    waiting_times = []
    service_times = []
    total_service_time = 0.0
    customer_count = 0
    
    def customer(env, cashier):
        nonlocal total_service_time, customer_count
        customer_count += 1
        arrival_time = env.now
        
        with cashier.request() as request:
            yield request
            wait_time = env.now - arrival_time
            waiting_times.append(wait_time)
            
            service_time = random.uniform(service_min, service_max)
            service_times.append(service_time)
            total_service_time += service_time
            
            yield env.timeout(service_time)
    
    def arrival_process(env, cashier):
        i = 0
        while True:
            yield env.timeout(random.expovariate(1.0 / arrival_rate))
            i += 1
            env.process(customer(env, cashier))
    
    env = simpy.Environment()
    cashier = simpy.Resource(env, capacity=num_cashiers)
    env.process(arrival_process(env, cashier))
    env.run(until=sim_time)
    
    if len(waiting_times) > 0:
        avg_wait = statistics.mean(waiting_times)
        max_wait = max(waiting_times)
        median_wait = statistics.median(waiting_times)
    else:
        avg_wait = max_wait = median_wait = 0
    
    utilization = total_service_time / (num_cashiers * sim_time) if num_cashiers > 0 else 0
    
    return {
        'num_cashiers': num_cashiers,
        'total_customers': customer_count,
        'avg_wait_time': avg_wait,
        'max_wait_time': max_wait,
        'median_wait_time': median_wait,
        'utilization': utilization,
        'waiting_times': waiting_times,
        'service_times': service_times
    }

# =====================
# STREAMLIT UI
# =====================
st.title("🍽️ Simulasi Antrean Restoran")
st.markdown("""
Aplikasi interaktif untuk simulasi antrean restoran menggunakan **Discrete Event Simulation (SimPy)**.
Ubah parameter di bawah untuk melakukan **What-If Analysis** dan melihat dampaknya terhadap performa sistem.
""")

# Sidebar untuk informasi dataset
with st.sidebar:
    st.header("📊 Informasi Dataset")
    if df_data is not None:
        st.success(f"✅ Dataset dimuat: {len(df_data)} order")
        st.caption(f"Parameter dari data riil:")
        st.caption(f"• Interval kedatangan: {REAL_ARRIVAL_RATE:.2f} menit")
        st.caption(f"• Waktu pelayanan: {REAL_SERVICE_MIN:.2f} - {REAL_SERVICE_MAX:.2f} menit")
    else:
        st.warning("⚠️ Menggunakan parameter default")
    
    st.markdown("---")
    st.header("⚙️ Parameter Simulasi")
    
    # Slider untuk parameter
    arrival_rate = st.slider(
    "Rata-rata Kedatangan (menit/pelanggan)",
    min_value=1.0,
    max_value=20.0,
    value=float(REAL_ARRIVAL_RATE),
    step=1.0,
    help="Semakin kecil nilai, semakin sering pelanggan datang"
)

    
    service_min = st.slider(
        "Waktu Pelayanan Minimum (menit)",
        min_value=1.0,
        max_value=5.0,
        value=float(REAL_SERVICE_MIN) if REAL_SERVICE_MIN else 2.0,
        step=0.5
    )
    
    service_max = st.slider(
        "Waktu Pelayanan Maksimum (menit)",
        min_value=2.0,
        max_value=10.0,
        value=float(REAL_SERVICE_MAX) if REAL_SERVICE_MAX else 4.0,
        step=0.5
    )
    
    num_cashiers = st.slider(
        "Jumlah Kasir",
        min_value=1,
        max_value=3,
        value=2,
        help="Jumlah kasir yang melayani pelanggan (maksimal 3 kasir)"
    )
    
    st.markdown("---")
    st.caption("💡 **Tip**: Gunakan slider untuk melakukan What-If Analysis")

# =====================
# Jalankan Simulasi
# =====================
with st.spinner("Menjalankan simulasi..."):
    result = run_simulation(
        num_cashiers=num_cashiers,
        arrival_rate=arrival_rate,
        service_min=service_min,
        service_max=service_max,
        sim_time=SIM_TIME,
        seed=RANDOM_SEED
    )

# =====================
# OUTPUT METRIK
# =====================
st.markdown("---")
st.subheader("📈 Hasil Simulasi")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rata-rata Waktu Tunggu", f"{result['avg_wait_time']:.2f} menit", 
            delta=f"{result['median_wait_time']:.2f} (median)" if result['median_wait_time'] > 0 else None)
col2.metric("Waktu Tunggu Maksimum", f"{result['max_wait_time']:.2f} menit")
col3.metric("Utilisasi Sistem", f"{result['utilization']:.2%}")
col4.metric("Pelanggan Terlayani", f"{result['total_customers']}")

# Interpretasi
st.markdown("---")
if result['utilization'] >= 0.9:
    st.warning(f"⚠️ **Sistem Overload**: Utilisasi {result['utilization']:.1%} mendekati kapasitas maksimal. Waktu tunggu tinggi ({result['avg_wait_time']:.2f} menit).")
elif result['utilization'] >= 0.6:
    st.success(f"✅ **Sistem Optimal**: Utilisasi {result['utilization']:.1%} menunjukkan sistem efisien dan stabil. Waktu tunggu {result['avg_wait_time']:.2f} menit.")
else:
    st.info(f"ℹ️ **Kapasitas Berlebih**: Utilisasi {result['utilization']:.1%} menunjukkan kapasitas idle. Waktu tunggu rendah ({result['avg_wait_time']:.2f} menit).")

# =====================
# VISUALISASI
# =====================
st.markdown("---")
st.subheader("📊 Visualisasi Hasil")

tab1, tab2, tab3 = st.tabs(["Grafik Utama", "Distribusi Waktu Tunggu", "What-If Analysis"])

with tab1:
    # Grafik utama
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Utilisasi sistem
    axes[0, 0].barh(['Utilisasi'], [result['utilization']], color='green' if result['utilization'] < 0.9 else 'red', alpha=0.7)
    axes[0, 0].axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Kapasitas Maksimal')
    axes[0, 0].set_xlim(0, 1.1)
    axes[0, 0].set_xlabel('Utilisasi')
    axes[0, 0].set_title('Utilisasi Sistem')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # 2. Waktu tunggu
    wait_metrics = [result['avg_wait_time'], result['median_wait_time'], result['max_wait_time']]
    wait_labels = ['Rata-rata', 'Median', 'Maksimum']
    axes[0, 1].bar(wait_labels, wait_metrics, color=['blue', 'orange', 'red'], alpha=0.7)
    axes[0, 1].set_ylabel('Waktu Tunggu (menit)')
    axes[0, 1].set_title('Metrik Waktu Tunggu')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Distribusi waktu tunggu (histogram)
    if len(result['waiting_times']) > 0:
        axes[1, 0].hist(result['waiting_times'], bins=30, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 0].axvline(result['avg_wait_time'], color='r', linestyle='--', 
                          label=f'Mean: {result["avg_wait_time"]:.2f}')
        axes[1, 0].set_xlabel('Waktu Tunggu (menit)')
        axes[1, 0].set_ylabel('Frekuensi')
        axes[1, 0].set_title('Distribusi Waktu Tunggu Pelanggan')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
    else:
        axes[1, 0].text(0.5, 0.5, 'Tidak ada data', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Distribusi Waktu Tunggu (Tidak Ada Data)')
    
    # 4. Ringkasan performa
    performance_categories = ['Waktu Tunggu', 'Utilisasi', 'Throughput']
    performance_scores = [
        min(100, max(0, 100 - result['avg_wait_time'] * 10)),  # Skor waktu tunggu (semakin kecil semakin baik)
        result['utilization'] * 100,
        min(100, (result['total_customers'] / SIM_TIME) * 60 * 10)  # Skor throughput
    ]
    axes[1, 1].barh(performance_categories, performance_scores, color=['blue', 'green', 'orange'], alpha=0.7)
    axes[1, 1].set_xlim(0, 100)
    axes[1, 1].set_xlabel('Skor (%)')
    axes[1, 1].set_title('Ringkasan Performa Sistem')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    # Distribusi detail
    if len(result['waiting_times']) > 0:
        fig2, ax = plt.subplots(figsize=(10, 6))
        ax.hist(result['waiting_times'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(result['avg_wait_time'], color='r', linestyle='--', linewidth=2, 
                  label=f'Mean: {result["avg_wait_time"]:.2f} menit')
        ax.axvline(result['median_wait_time'], color='g', linestyle='--', linewidth=2,
                  label=f'Median: {result["median_wait_time"]:.2f} menit')
        ax.set_xlabel('Waktu Tunggu (menit)', fontsize=12)
        ax.set_ylabel('Frekuensi Pelanggan', fontsize=12)
        ax.set_title('Distribusi Detail Waktu Tunggu Pelanggan', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig2)
        
        # Statistik distribusi
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        with col_stat1:
            st.metric("Percentile 25%", f"{np.percentile(result['waiting_times'], 25):.2f} menit")
        with col_stat2:
            st.metric("Percentile 75%", f"{np.percentile(result['waiting_times'], 75):.2f} menit")
        with col_stat3:
            st.metric("Std Dev", f"{np.std(result['waiting_times']):.2f} menit")
    else:
        st.info("Tidak ada data waktu tunggu untuk divisualisasikan.")

with tab3:
    # What-If Analysis: Bandingkan berbagai skenario jumlah kasir
    st.markdown("### 🔍 What-If Analysis: Pengaruh Jumlah Kasir")
    st.caption("Simulasi untuk berbagai jumlah kasir dengan parameter yang sama")
    
    whatif_cashiers = [1, 2, 3]
    whatif_results = []
    
    with st.spinner("Menjalankan analisis What-If..."):
        for num_cash in whatif_cashiers:
            whatif_result = run_simulation(
                num_cashiers=num_cash,
                arrival_rate=arrival_rate,
                service_min=service_min,
                service_max=service_max,
                sim_time=SIM_TIME,
                seed=RANDOM_SEED
            )
            whatif_results.append(whatif_result)
    
    # Visualisasi perbandingan
    fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Grafik waktu tunggu
    wait_times = [r['avg_wait_time'] for r in whatif_results]
    axes[0].plot(whatif_cashiers, wait_times, marker='o', linewidth=2, markersize=10, color='blue')
    axes[0].set_xlabel('Jumlah Kasir')
    axes[0].set_ylabel('Rata-rata Waktu Tunggu (menit)')
    axes[0].set_title('Pengaruh Jumlah Kasir terhadap Waktu Tunggu')
    axes[0].set_xticks(whatif_cashiers)
    axes[0].grid(True, alpha=0.3)
    
    # Grafik utilisasi
    utilizations = [r['utilization'] for r in whatif_results]
    axes[1].plot(whatif_cashiers, utilizations, marker='s', linewidth=2, markersize=10, color='green')
    axes[1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Kapasitas Maksimal')
    axes[1].set_xlabel('Jumlah Kasir')
    axes[1].set_ylabel('Utilisasi Sistem')
    axes[1].set_title('Pengaruh Jumlah Kasir terhadap Utilisasi')
    axes[1].set_ylim(0, 1.1)
    axes[1].set_xticks(whatif_cashiers)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig3)
    
    # Tabel perbandingan
    st.markdown("### 📋 Tabel Perbandingan Skenario")
    comparison_data = {
        'Jumlah Kasir': whatif_cashiers,
        'Rata-rata Waktu Tunggu (menit)': [f"{r['avg_wait_time']:.2f}" for r in whatif_results],
        'Waktu Tunggu Maks (menit)': [f"{r['max_wait_time']:.2f}" for r in whatif_results],
        'Utilisasi (%)': [f"{r['utilization']:.1%}" for r in whatif_results],
        'Pelanggan Terlayani': [r['total_customers'] for r in whatif_results]
    }
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Rekomendasi
    optimal_idx = 1  # Default: 2 kasir
    for i, r in enumerate(whatif_results):
        if 0.6 <= r['utilization'] < 0.9:
            if r['avg_wait_time'] < whatif_results[optimal_idx]['avg_wait_time']:
                optimal_idx = i
    
    st.success(
        f"🎯 **Rekomendasi Optimal (berdasarkan skenario saat ini)**: "
        f"Gunakan **{whatif_results[optimal_idx]['num_cashiers']} kasir** "
        f"dengan utilisasi {whatif_results[optimal_idx]['utilization']:.1%} dan "
        f"waktu tunggu {whatif_results[optimal_idx]['avg_wait_time']:.2f} menit"
    )

    st.markdown("---")
    st.markdown("### ⏰ Skenario What-If **Jam Sibuk** vs Kondisi Riil")
    st.caption(
        "Dataset digunakan sebagai **kondisi normal (data riil)**. "
        "Skenario **jam sibuk** dimodelkan dengan interval kedatangan yang jauh lebih pendek."
    )

    # Pilih jumlah kasir untuk perbandingan jam sibuk
    busy_cashiers = st.slider(
        "Jumlah kasir untuk skenario jam sibuk (perbandingan dengan data riil)",
        min_value=1,
        max_value=3,
        value=2,
        key="busy_hour_cashiers"
    )

    # Interval kedatangan jam sibuk (lebih kecil dari data riil)
    busy_interval = st.slider(
        "Interval kedatangan JAM SIBUK (menit/pelanggan)",
        min_value=5.0,
        max_value=30.0,
        value=12.0,
        step=1.0,
        help="Semakin kecil interval, semakin padat (jam sibuk). Contoh: 10–15 menit."
    )

    with st.spinner("Mensimulasikan kondisi normal vs jam sibuk..."):
        # Kondisi normal berbasis data riil
        baseline_busy = run_simulation(
            num_cashiers=busy_cashiers,
            arrival_rate=float(REAL_ARRIVAL_RATE) if REAL_ARRIVAL_RATE else arrival_rate,
            service_min=float(REAL_SERVICE_MIN) if REAL_SERVICE_MIN else service_min,
            service_max=float(REAL_SERVICE_MAX) if REAL_SERVICE_MAX else service_max,
            sim_time=SIM_TIME,
            seed=RANDOM_SEED,
        )

        # Skenario jam sibuk (interval kedatangan diturunkan)
        peak_busy = run_simulation(
            num_cashiers=busy_cashiers,
            arrival_rate=busy_interval,
            service_min=float(REAL_SERVICE_MIN) if REAL_SERVICE_MIN else service_min,
            service_max=float(REAL_SERVICE_MAX) if REAL_SERVICE_MAX else service_max,
            sim_time=SIM_TIME,
            seed=RANDOM_SEED + 1,
        )

    col_norm, col_peak = st.columns(2)
    with col_norm:
        st.markdown("#### Kondisi **Normal (Data Riil)**")
        st.metric("Rata-rata Waktu Tunggu", f"{baseline_busy['avg_wait_time']:.2f} menit")
        st.metric("Utilisasi Sistem", f"{baseline_busy['utilization']:.2%}")
        st.metric("Pelanggan Terlayani", f"{baseline_busy['total_customers']}")

    with col_peak:
        st.markdown("#### Skenario **Jam Sibuk (Simulasi)**")
        st.metric("Rata-rata Waktu Tunggu", f"{peak_busy['avg_wait_time']:.2f} menit")
        st.metric("Utilisasi Sistem", f"{peak_busy['utilization']:.2%}")
        st.metric("Pelanggan Terlayani", f"{peak_busy['total_customers']}")

    # Grafik perbandingan normal vs jam sibuk
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 5))

    axes4[0].bar(
        ["Normal (Riil)", "Jam Sibuk (Simulasi)"],
        [baseline_busy["avg_wait_time"], peak_busy["avg_wait_time"]],
        color=["steelblue", "darkred"],
        alpha=0.8,
    )
    axes4[0].set_ylabel("Rata-rata Waktu Tunggu (menit)")
    axes4[0].set_title("Perbandingan Waktu Tunggu: Normal vs Jam Sibuk")
    axes4[0].grid(True, axis="y", alpha=0.3)

    axes4[1].bar(
        ["Normal (Riil)", "Jam Sibuk (Simulasi)"],
        [baseline_busy["utilization"], peak_busy["utilization"]],
        color=["green", "orange"],
        alpha=0.8,
    )
    axes4[1].set_ylabel("Utilisasi Sistem")
    axes4[1].set_ylim(0, 1.1)
    axes4[1].set_title("Perbandingan Utilisasi: Normal vs Jam Sibuk")
    axes4[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig4)

    st.info(
        "Interpretasi singkat: **data riil merepresentasikan kondisi normal**, sedangkan "
        "**skenario jam sibuk dibangun murni dari simulasi** dengan interval kedatangan yang diturunkan. "
        "Ini menjadi dasar untuk rekomendasi kebijakan operasional (misalnya penambahan kasir saat jam sibuk)."
    )

# =====================
# Footer
# =====================
st.markdown("---")
st.caption("💻 **Aplikasi Simulasi Antrean Restoran** | Dibuat dengan Streamlit dan SimPy | "
           "Dataset: hotel_restaurant_orders.csv")
