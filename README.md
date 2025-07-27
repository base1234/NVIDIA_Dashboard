
# 🧠 RL Agent vs Random Policy Dashboard

This project presents an interactive dashboard built using **Streamlit** and **Plotly** to visualize and compare the performance of a Reinforcement Learning (RL) agent against a random policy. The dataset includes system-level telemetry collected from an NVIDIA Jetson Nano device during CPU, GPU, memory, and mixed workload executions.

## 📌 Project Overview

- **Goal**: Evaluate power and thermal efficiency of an RL-based agent vs. a random policy on embedded workloads.
- **Device**: NVIDIA Jetson Nano
- **Metrics Tracked**:
  - CPU, GPU, RAM, and SWAP utilization
  - Power consumption
  - Temperature readings
  - Reward scores
  - Action distributions
  - State transitions

## 📊 Dashboard Features

- Toggle between **RL Agent** and **Random Policy** visualizations
- Track average and percentage-based savings in power and temperature
- Filter insights by **task type** (e.g., CPU-bound, GPU-heavy)
- Visualize system metric transitions before and after actions
- Interactive graphs for:
  - Action distributions
  - Reward trends
  - Utilization and temperature deltas

## 🏗️ Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly
- **Backend**: Python (pandas, numpy)
- **Data Source**: Synthetic telemetry CSV
- **Environment**: venv / conda

## 📁 Project Structure

```
jetson dashboard
├── README.md
├── streamlit_dashboard.py
└── synthetic_rl_combined_dataset_noisy.csv
```

## ⚙️ Getting Started

1. **Clone the repo**

```bash
git clone https://github.com/your-username/rl-dashboard.git
cd rl-dashboard
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Launch the dashboard**

```bash
streamlit run rl_dashboard.py
```

## 📷 Screenshots

> (Place your visuals in a `/screenshots` folder and reference them here)

### Screenshot 1
![Screenshot 1](./Screenshot%202025-07-24%20205458.png)

### Screenshot 2
![Screenshot 2](./Screenshot%202025-07-24%20205755.png)

### Screenshot 3
![Screenshot 3](./Screenshot%202025-07-27%20154229.png)

### Screenshot 4
![Screenshot 4](./Screenshot%202025-07-27%20154321.png)

### Screenshot 5
![Screenshot 5](./Screenshot%202025-07-27%20154438.png)

## 🚀 Future Enhancements

- Real-time metric ingestion from Jetson via MQTT or sockets
- Online RL agent monitoring and retraining interface
- Export insights to CSV or PDF
- Cloud-hosted version with user authentication

## 🤝 Contributing

Pull requests and issues are welcome. Please open a discussion for feature requests or bug reports.

## 👤 Author

**Srikanth**  
[LinkedIn](https://www.linkedin.com/in/sriknar13) | [GitHub](https://github.com/base1234)
