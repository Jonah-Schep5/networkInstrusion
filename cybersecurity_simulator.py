import streamlit as st
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from PostgresAgent import PostgresAgent
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(page_title="Cybersecurity Simulator", layout="wide", page_icon="🛡️")

# Load environment and database connection
@st.cache_resource
def load_db_connection():
    load_dotenv(f'{os.getcwd()}/.env')
    agent = PostgresAgent(
        os.getenv('POSTGRES_USER'),
        os.getenv('POSTGRES_PASSWD'),
        os.getenv('POSTGRES_HOST'),
        os.getenv('POSTGRES_PORT'),
        os.getenv('POSTGRES_DATABASE')
    )
    username = os.environ.get('USER')
    return agent, username

# Load models
@st.cache_resource
def load_models():
    models = {}
    model_files = {
        'phishing': 'phishing_xgboost_model.pkl',
        'zeroday': 'zeroday_xgboost_model.pkl',
        'itd': 'itd_xgboost_model.pkl',
        'apt': 'apt_xgboost_model.pkl'
    }
    for name, path in model_files.items():
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    return models

# Load sample data from database
@st.cache_data
def load_sample_data(_agent, username, limit=10000):
    sql = f"""
    SELECT * FROM {username}.NETWORK_TRAFFIC_HISTORY
    ORDER BY RANDOM()
    LIMIT {limit}
    """
    df = _agent.execute_dml(sql)
    return df

# Prepare features for each model
def prepare_phishing_features(row):
    """Prepare features for phishing model"""
    encoder = LabelEncoder()
    features = {
        'protocol': row['protocol'],
        'duration': row['duration'],
        'packets': row['packets'],
        'bytes': row['bytes'],
        'tcp_flags': row['tcp_flags'],
        'bytes_per_packet': row['bytes_per_packet'],
        'packets_per_second': row['packets_per_second'],
        'service': row['service'],
        'is_weekend': 1 if row['is_weekend'] == 'True' else 0,
        'hour_of_day': row['hour_of_day'],
        'day_of_week': row['day_of_week'],
        'bytes_ratio': row['bytes_ratio'],
        'packet_size_variance': row['packet_size_variance'],
        'connection_frequency': row['connection_frequency'],
        'unique_ports_per_source': row['unique_ports_per_source']
    }

    # Encode categorical features
    protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
    tcp_flags_map = {'SYN': 0, 'ACK': 1, 'FIN': 2, 'RST': 3, 'PSH': 4, 'URG': 5}
    service_map = {'HTTP': 0, 'HTTPS': 1, 'SSH': 2, 'FTP': 3, 'DNS': 4, 'SMTP': 5,
                   'RDP': 6, 'MSSQL': 7, 'PostgreSQL': 8, 'MySQL': 9}

    features['protocol'] = protocol_map.get(features['protocol'], 0)
    features['tcp_flags'] = tcp_flags_map.get(features['tcp_flags'], 0)
    features['service'] = service_map.get(features['service'], 0)

    return pd.DataFrame([features])

def prepare_zeroday_features(row):
    """Same as phishing for this model"""
    return prepare_phishing_features(row)

def prepare_itd_features(row):
    """Prepare features for ITD model"""
    # Check if source/dest are internal (10.x.x.x or 172.x.x.x or 192.168.x.x)
    def is_internal(ip):
        if isinstance(ip, str):
            return ip.startswith('10.') or ip.startswith('172.') or ip.startswith('192.168.')
        return False

    features = {
        'dest_port': row['dest_port'],
        'protocol': row['protocol'],
        'duration': row['duration'],
        'packets': row['packets'],
        'bytes': row['bytes'],
        'bytes_per_packet': row['bytes_per_packet'],
        'tcp_flags': row['tcp_flags'],
        'packet_size_variance': row['packet_size_variance'],
        'connection_frequency': row['connection_frequency'],
        'off_hours': 1 if row['hour_of_day'] < 6 or row['hour_of_day'] > 20 else 0,
        'is_weekend': 1 if row['is_weekend'] == 'True' else 0,
        'hour_of_day': row['hour_of_day'],
        'is_internal_source': 1 if is_internal(row['source_ip']) else 0,
        'is_internal_dest': 1 if is_internal(row['dest_ip']) else 0,
        'time_period': row['hour_of_day'] // 6,  # 0-3 for 4 time periods
        'bytes_percentile': row['bytes'] / 100000,  # Simplified percentile
        'packets_percentile': row['packets'] / 1000,
        'duration_percentile': row['duration'] / 100,
        'bpp_percentile': row['bytes_per_packet'] / 1000,
        'service': row['service']
    }

    # Encode categorical
    protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
    tcp_flags_map = {'SYN': 0, 'ACK': 1, 'FIN': 2, 'RST': 3, 'PSH': 4, 'URG': 5}
    service_map = {'HTTP': 0, 'HTTPS': 1, 'SSH': 2, 'FTP': 3, 'DNS': 4, 'SMTP': 5,
                   'RDP': 6, 'MSSQL': 7, 'PostgreSQL': 8, 'MySQL': 9}

    features['protocol'] = protocol_map.get(features['protocol'], 0)
    features['tcp_flags'] = tcp_flags_map.get(features['tcp_flags'], 0)
    features['service'] = service_map.get(features['service'], 0)

    return pd.DataFrame([features])

def prepare_apt_features(row):
    """Prepare features for APT model - includes timestamp and IPs"""
    features = {
        'timestamp': time.mktime(row['timestamp'].timetuple()) if isinstance(row['timestamp'], datetime) else 0,
        'source_ip': hash(row['source_ip']) % 10000,  # Simple hash encoding
        'dest_ip': hash(row['dest_ip']) % 10000,
        'source_port': row['source_port'],
        'dest_port': row['dest_port'],
        'protocol': row['protocol'],
        'duration': row['duration'],
        'packets': row['packets'],
        'bytes': row['bytes'],
        'tcp_flags': row['tcp_flags'],
        'bytes_per_packet': row['bytes_per_packet'],
        'packets_per_second': row['packets_per_second'],
        'service': row['service'],
        'severity_score': row['severity_score'],
        'is_weekend': 1 if row['is_weekend'] == 'True' else 0,
        'hour_of_day': row['hour_of_day'],
        'day_of_week': row['day_of_week'],
        'bytes_ratio': row['bytes_ratio'],
        'packet_size_variance': row['packet_size_variance'],
        'connection_frequency': row['connection_frequency'],
        'unique_ports_per_source': row['unique_ports_per_source']
    }

    # Encode categorical
    protocol_map = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
    tcp_flags_map = {'SYN': 0, 'ACK': 1, 'FIN': 2, 'RST': 3, 'PSH': 4, 'URG': 5}
    service_map = {'HTTP': 0, 'HTTPS': 1, 'SSH': 2, 'FTP': 3, 'DNS': 4, 'SMTP': 5,
                   'RDP': 6, 'MSSQL': 7, 'PostgreSQL': 8, 'MySQL': 9}

    features['protocol'] = protocol_map.get(features['protocol'], 0)
    features['tcp_flags'] = tcp_flags_map.get(features['tcp_flags'], 0)
    features['service'] = service_map.get(features['service'], 0)

    return pd.DataFrame([features])

# Initialize session state
if 'total_saved' not in st.session_state:
    st.session_state.total_saved = 0
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'current_day' not in st.session_state:
    st.session_state.current_day = 0
if 'total_processed' not in st.session_state:
    st.session_state.total_processed = 0
if 'attack_counts' not in st.session_state:
    st.session_state.attack_counts = {'phishing': 0, 'zeroday': 0, 'itd': 0, 'apt': 0}
if 'running' not in st.session_state:
    st.session_state.running = False

# Title and description
st.title("🛡️ Cybersecurity Network Intrusion Detection Simulator")
st.markdown("Real-time simulation of network traffic with ML-based attack detection")

# Sidebar controls
st.sidebar.header("⚙️ Simulation Controls")

# Attack cost configuration
st.sidebar.subheader("💰 Attack Costs")
phishing_cost = st.sidebar.number_input("Phishing Cost ($)", value=50000, step=10000)
zeroday_cost = st.sidebar.number_input("Zero-Day Cost ($)", value=500000, step=50000)
itd_cost = st.sidebar.number_input("Insider Threat Cost ($)", value=200000, step=25000)
apt_cost = st.sidebar.number_input("APT Cost ($)", value=1000000, step=100000)

# Operational cost configuration
st.sidebar.subheader("📊 Operational Costs")
ops_cost_per_day = st.sidebar.slider("Cost per Day ($)", min_value=100, max_value=10000, value=1000, step=100)

# Simulation speed
st.sidebar.subheader("⚡ Speed Settings")
days_per_second = st.sidebar.slider("Days per Second", min_value=1, max_value=30, value=5, step=1)
st.sidebar.info(f"Simulating {days_per_second} days of traffic per second")

# Attack injection buttons
st.sidebar.subheader("🚨 Inject Attacks")
col1, col2 = st.sidebar.columns(2)
inject_phishing = col1.button("📧 Phishing", use_container_width=True)
inject_zeroday = col2.button("💣 Zero-Day", use_container_width=True)
inject_itd = col1.button("👤 Insider", use_container_width=True)
inject_apt = col2.button("🎯 APT", use_container_width=True)

# Reset button
if st.sidebar.button("🔄 Reset Simulation", use_container_width=True):
    st.session_state.total_saved = 0
    st.session_state.total_cost = 0
    st.session_state.detections = []
    st.session_state.current_day = 0
    st.session_state.total_processed = 0
    st.session_state.attack_counts = {'phishing': 0, 'zeroday': 0, 'itd': 0, 'apt': 0}
    st.rerun()

# Load data and models
agent, username = load_db_connection()
models = load_models()
data = load_sample_data(agent, username, limit=50000)

# Filter normal traffic for faster simulation
normal_data = data[data['attack_state'] == 'Normal'].copy()
attack_data = {
    'phishing': data[data['attack_state'] == 'Phishing'].copy(),
    'zeroday': data[data['attack_state'] == 'Zero_Day'].copy(),
    'itd': data[data['attack_state'] == 'Insider_Threat'].copy(),
    'apt': data[data['attack_state'] == 'APT'].copy()
}

# Main metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("💰 Total Saved", f"${st.session_state.total_saved:,.0f}")
with col2:
    st.metric("💸 Operational Cost", f"${st.session_state.total_cost:,.0f}")
with col3:
    net_benefit = st.session_state.total_saved - st.session_state.total_cost
    st.metric("📈 Net Benefit", f"${net_benefit:,.0f}", delta=f"${net_benefit:,.0f}")
with col4:
    st.metric("📅 Days Simulated", f"{st.session_state.current_day:.1f}")

# Detection metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📧 Phishing Detected", st.session_state.attack_counts['phishing'])
with col2:
    st.metric("💣 Zero-Day Detected", st.session_state.attack_counts['zeroday'])
with col3:
    st.metric("👤 Insider Detected", st.session_state.attack_counts['itd'])
with col4:
    st.metric("🎯 APT Detected", st.session_state.attack_counts['apt'])

# Recent detections
st.subheader("🚨 Recent Attack Detections")
detection_container = st.container()

with detection_container:
    if st.session_state.detections:
        recent_detections = st.session_state.detections[-10:][::-1]  # Last 10, reversed
        for detection in recent_detections:
            alert_type = detection['type'].upper()
            timestamp = detection['timestamp']
            cost_saved = detection['cost']

            # Color coding
            color_map = {
                'PHISHING': '🟡',
                'ZERODAY': '🔴',
                'ITD': '🟠',
                'APT': '🔴'
            }

            st.warning(f"{color_map.get(alert_type, '⚠️')} **{alert_type} DETECTED** | Day {timestamp:.2f} | Cost Prevented: ${cost_saved:,.0f}")
    else:
        st.info("No attacks detected yet. Waiting for malicious traffic...")

# Streaming simulation
st.subheader("📊 Live Traffic Stream")
stream_container = st.empty()

# Chart for financial tracking
chart_container = st.empty()

# Process attack injection
injected_attack = None
injected_cost = 0

if inject_phishing and len(attack_data['phishing']) > 0:
    injected_attack = ('phishing', attack_data['phishing'].sample(1).iloc[0], phishing_cost)
elif inject_zeroday and len(attack_data['zeroday']) > 0:
    injected_attack = ('zeroday', attack_data['zeroday'].sample(1).iloc[0], zeroday_cost)
elif inject_itd and len(attack_data['itd']) > 0:
    injected_attack = ('itd', attack_data['itd'].sample(1).iloc[0], itd_cost)
elif inject_apt and len(attack_data['apt']) > 0:
    injected_attack = ('apt', attack_data['apt'].sample(1).iloc[0], apt_cost)

# Simulation loop
if injected_attack:
    attack_type, attack_row, attack_cost = injected_attack

    # Prepare features and run prediction
    try:
        if attack_type == 'phishing':
            features = prepare_phishing_features(attack_row)
            prediction = models['phishing'].predict(features)[0]
        elif attack_type == 'zeroday':
            features = prepare_zeroday_features(attack_row)
            prediction = models['zeroday'].predict(features)[0]
        elif attack_type == 'itd':
            features = prepare_itd_features(attack_row)
            prediction = models['itd'].predict(features)[0]
        elif attack_type == 'apt':
            features = prepare_apt_features(attack_row)
            prediction = models['apt'].predict(features)[0]

        # If detected (prediction = 1 or True)
        if prediction:
            st.session_state.attack_counts[attack_type] += 1
            st.session_state.total_saved += attack_cost
            st.session_state.detections.append({
                'type': attack_type,
                'timestamp': st.session_state.current_day,
                'cost': attack_cost
            })
            st.success(f"✅ {attack_type.upper()} attack detected and blocked! ${attack_cost:,.0f} saved!")
        else:
            st.error(f"⚠️ {attack_type.upper()} attack MISSED by the model!")
    except Exception as e:
        st.error(f"Error processing attack: {e}")

    st.rerun()

# Simulate normal traffic streaming
with stream_container:
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Simulate a batch of traffic
    batch_size = 100
    for i in range(batch_size):
        # Update operational costs
        day_increment = days_per_second / batch_size
        st.session_state.current_day += day_increment
        st.session_state.total_cost = st.session_state.current_day * ops_cost_per_day
        st.session_state.total_processed += 1

        # Sample normal traffic (no need to run through model)
        sample = normal_data.sample(1).iloc[0]

        status_text.text(f"Processing traffic: {sample['source_ip']} → {sample['dest_ip']} | "
                        f"{sample['protocol']} | {sample['service']} | Status: ✅ Normal")

        progress_bar.progress((i + 1) / batch_size)
        time.sleep(1.0 / batch_size / days_per_second)

    status_text.text(f"Processed {st.session_state.total_processed:,} records | Day {st.session_state.current_day:.1f}")

# Update chart
if len(st.session_state.detections) > 0:
    detection_df = pd.DataFrame(st.session_state.detections)
    detection_df['cumulative_saved'] = detection_df['cost'].cumsum()

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Financial Impact Over Time", "Attack Detection Timeline"),
        vertical_spacing=0.15
    )

    # Financial chart
    days = [0] + detection_df['timestamp'].tolist() + [st.session_state.current_day]
    saved = [0] + detection_df['cumulative_saved'].tolist() + [detection_df['cumulative_saved'].iloc[-1]]
    cost = [0] + [st.session_state.current_day * ops_cost_per_day / len(days)] * (len(days) - 2) + [st.session_state.total_cost]

    fig.add_trace(
        go.Scatter(x=days, y=saved, name="Money Saved", line=dict(color='green', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=days, y=cost, name="Operational Cost", line=dict(color='red', width=3, dash='dash')),
        row=1, col=1
    )

    # Attack timeline
    colors = {'phishing': 'gold', 'zeroday': 'red', 'itd': 'orange', 'apt': 'darkred'}
    for attack_type in ['phishing', 'zeroday', 'itd', 'apt']:
        type_detections = detection_df[detection_df['type'] == attack_type]
        if len(type_detections) > 0:
            fig.add_trace(
                go.Scatter(
                    x=type_detections['timestamp'],
                    y=type_detections['cost'],
                    mode='markers',
                    name=attack_type.upper(),
                    marker=dict(size=12, color=colors[attack_type])
                ),
                row=2, col=1
            )

    fig.update_xaxes(title_text="Days", row=1, col=1)
    fig.update_xaxes(title_text="Days", row=2, col=1)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
    fig.update_yaxes(title_text="Cost Prevented ($)", row=2, col=1)

    fig.update_layout(height=700, showlegend=True)

    with chart_container:
        st.plotly_chart(fig, use_container_width=True)

# Auto-refresh
time.sleep(0.1)
st.rerun()
