# Cybersecurity Network Intrusion Detection Simulator

## Overview
An interactive real-time cybersecurity simulator that demonstrates ML-based network intrusion detection. The simulator streams network traffic data and allows you to inject attacks to see how the trained models detect and prevent security breaches.

## Features

### Real-Time Streaming
- Simulates days of network traffic in seconds
- Displays live traffic flow (source → destination)
- Auto-refreshing dashboard with current metrics

### Attack Detection Models
Four specialized XGBoost models detect different attack types:
- **Phishing** (15 features) - Detects phishing attempts
- **Zero-Day** (15 features) - Identifies zero-day exploits
- **Insider Threat Detection (ITD)** (20 features) - Catches insider threats
- **Advanced Persistent Threat (APT)** (21 features) - Detects APT campaigns

### Interactive Controls

#### Simulation Speed
- **Days per Second Slider**: Control how fast time passes (1-30 days/sec)
- Fast simulation for quick demonstrations
- Normal traffic doesn't run through models for performance

#### Attack Cost Configuration
Adjust the financial impact of each attack type:
- Phishing: Default $50,000
- Zero-Day: Default $500,000
- Insider Threat: Default $200,000
- APT: Default $1,000,000

#### Operational Costs
- Configurable cost per day ($100 - $10,000)
- Increases linearly over simulated time
- Shows trade-off between detection and operational expenses

#### Attack Injection Buttons
Manually trigger attacks to test detection:
- 📧 **Phishing** - Inject phishing attack
- 💣 **Zero-Day** - Inject zero-day exploit
- 👤 **Insider** - Inject insider threat
- 🎯 **APT** - Inject APT attack

### Visualizations

#### Key Metrics Dashboard
- **Total Saved**: Money prevented from detected attacks
- **Operational Cost**: Cumulative cost of running detection system
- **Net Benefit**: Savings minus operational costs
- **Days Simulated**: Current simulation time

#### Detection Counters
Real-time count of each attack type detected

#### Recent Alerts
Live feed of the last 10 attack detections with:
- Attack type with color coding
- Timestamp (simulation day)
- Cost prevented

#### Financial Impact Chart
Two-panel chart showing:
1. **Money Saved vs Operational Cost** - Cumulative comparison over time
2. **Attack Detection Timeline** - Scatter plot of when attacks were detected and their costs

## How to Run

### Prerequisites
```bash
pip install streamlit plotly xgboost pandas numpy python-dotenv psycopg2
```

### Launch Simulator
```bash
streamlit run cybersecurity_simulator.py
```

The app will open in your browser at `http://localhost:8501`

### Using the Simulator

1. **Adjust Settings** (Left Sidebar)
   - Set attack costs to realistic values for your scenario
   - Configure operational costs
   - Adjust simulation speed

2. **Watch Normal Traffic**
   - The simulator automatically streams normal network traffic
   - Normal traffic is sampled from the database (not run through models)
   - Progress bar shows current batch processing

3. **Inject Attacks**
   - Click any attack button in the sidebar
   - Watch for detection alerts
   - See financial impact updated in real-time

4. **Monitor Metrics**
   - Top dashboard shows cumulative savings and costs
   - Detection counters show effectiveness by attack type
   - Charts visualize ROI over time

5. **Reset Simulation**
   - Click "Reset Simulation" to start fresh
   - Clears all metrics and detections

## Architecture

### Data Flow
1. **Database**: PostgreSQL with 525,000 network traffic records
2. **Sampling**: Random sampling of normal and attack traffic
3. **Feature Engineering**: Attack-specific feature preparation
4. **Model Inference**: Real-time prediction on injected attacks
5. **Visualization**: Streamlit real-time updates

### Performance Optimization
- Normal traffic bypasses model inference (only displayed)
- Attack traffic triggers model prediction
- Database queries cached with `@st.cache_data`
- Models loaded once with `@st.cache_resource`

### Model Features

**Phishing & Zero-Day Models** (15 features):
- protocol, duration, packets, bytes, tcp_flags
- bytes_per_packet, packets_per_second, service
- is_weekend, hour_of_day, day_of_week
- bytes_ratio, packet_size_variance
- connection_frequency, unique_ports_per_source

**ITD Model** (20 features):
- All above plus: dest_port, off_hours
- is_internal_source, is_internal_dest, time_period
- bytes_percentile, packets_percentile, duration_percentile, bpp_percentile

**APT Model** (21 features):
- All phishing features plus: timestamp, source_ip, dest_ip
- source_port, dest_port, severity_score

## Configuration

### Environment Variables
Create `.env` file with:
```
POSTGRES_USER=your_username
POSTGRES_PASSWD=your_password
POSTGRES_HOST=your_host
POSTGRES_PORT=5432
POSTGRES_DATABASE=your_database
USER=your_username
```

### Model Files Required
- `phishing_xgboost_model.pkl`
- `zeroday_xgboost_model.pkl`
- `itd_xgboost_model.pkl`
- `apt_xgboost_model.pkl`

### Database Schema
Requires `NETWORK_TRAFFIC_HISTORY` table with 22 columns including:
- timestamp, source_ip, dest_ip, ports, protocol
- duration, packets, bytes, tcp_flags, service
- attack_state, severity_score, temporal features
- derived features (bytes_ratio, packet_size_variance, etc.)

## Business Value Demonstration

The simulator demonstrates:
1. **ROI Calculation**: Direct comparison of savings vs operational costs
2. **Detection Effectiveness**: Attack-specific detection rates
3. **Cost-Benefit Analysis**: Configurable costs for different scenarios
4. **Real-Time Monitoring**: Live dashboard for security operations
5. **What-If Analysis**: Test different cost assumptions

Perfect for:
- Executive presentations
- Security team training
- Budget justification
- Product demonstrations
- Research demonstrations

## Technical Details

- **Framework**: Streamlit for interactive web interface
- **ML Models**: XGBoost binary classifiers
- **Database**: PostgreSQL with 525K records
- **Visualization**: Plotly for interactive charts
- **Encoding**: LabelEncoder for categorical features
- **Auto-refresh**: Sub-second updates for real-time feel

## Limitations

- Normal traffic is not actually processed by models (performance optimization)
- Attack detection based on pre-trained models (not adaptive)
- Simplified feature engineering (some features approximated)
- Single-threaded streaming (suitable for demo purposes)
- Model predictions are deterministic (not probabilistic display)

## Future Enhancements

Possible improvements:
- Add false positive simulation
- Include detection confidence scores
- Multi-user concurrent simulations
- Historical playback from specific date ranges
- Export reports and metrics
- Add more attack types
- Implement adaptive threat response
- Integration with real SIEM systems
