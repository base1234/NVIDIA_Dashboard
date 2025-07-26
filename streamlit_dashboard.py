import streamlit as st
import pandas as pd
import plotly.express as px

# NVIDIA green color
nvidia_green = "#76B900"
nvidia_colors = {
    "rl_agent": nvidia_green,
    "random": "#9dbf56"
}

def style_fig(fig, policy_names):
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color=nvidia_green),
        legend=dict(font=dict(color=nvidia_green)),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    for trace in fig.data:
        if hasattr(trace, 'name') and trace.name in policy_names:
            color = nvidia_colors.get(trace.name, None)
            if color:
                if hasattr(trace, 'line'):
                    trace.line.color = color
                if hasattr(trace, 'marker'):
                    trace.marker.color = color
    return fig

@st.cache_data
def load_data():
    df = pd.read_csv(
        "synthetic_rl_combined_dataset_noisy.csv",
        parse_dates=["timestamp_before", "timestamp_after"]
    )
    return df

def build_natural_sequence(df, metric_base):
    before = df[['timestamp_before', metric_base + '_before']].rename(
        columns={'timestamp_before': 'timestamp', metric_base + '_before': metric_base}
    )
    after = df[['timestamp_after', metric_base + '_after']].rename(
        columns={'timestamp_after': 'timestamp', metric_base + '_after': metric_base}
    )
    combined = pd.concat([before, after], ignore_index=True)
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    return combined

df = load_data()

st.set_page_config(layout="wide")

# CSS to reduce top margin/padding for Streamlit 1.46.1 and NVIDIA theme
st.markdown(f"""
    <style>
        /* Stronger selector for main container padding/margin */
        main.css-1d391kg {{
            padding-top: 0 !important;
            margin-top: 0 !important;
        }}

        /* Reduce margin above all headings */
        h1, h2, h3, h4 {{
            margin-top: 0.1rem !important;
            margin-bottom: 0.1rem !important;
            color: {nvidia_green} !important;
        }}

        /* Insert a hidden div to pull content upward */
        .reduce-top-margin {{
            margin-top: -40px;
        }}

        /* Global app background and text color */
        .stApp {{
            background-color: #000000 !important;
            color: {nvidia_green} !important;
        }}

        /* Metric values and labels color */
        .stMetric-value, .stMetric-label {{
            color: {nvidia_green} !important;
        }}
    </style>

    <!-- Negative margin spacer before content -->
    <div class="reduce-top-margin"></div>
""", unsafe_allow_html=True)


st.title("RL vs Random Power-Performance Tradeoff Optimization Policy Dashboard")
st.markdown("Use the sidebar filters. Metrics shown in 4 green columns on black background.")

policy_filter = st.sidebar.selectbox("Select Policy", ["All", "rl_agent", "random"])
task_filter = st.sidebar.selectbox("Filter by Task Type", ["All"] + sorted(df["task_type"].unique()))

filtered_df = df.copy()
if task_filter != "All":
    filtered_df = filtered_df[filtered_df["task_type"] == task_filter]

def avg_metrics_full(df, policy):
    d = df[df["policy_type"] == policy]
    return (
        d["reward_score"].mean(),
        d["power_after"].mean(),
        d["temp_after"].mean(),
        d["cpu_util_after"].mean(),
        d["gpu_util_after"].mean(),
        d["ram_after"].mean() * 100,
        d["swap_after"].mean() * 100
    )

def percent_diff(new, old):
    if old == 0 or pd.isna(old) or pd.isna(new):
        return None
    return (new - old) / abs(old) * 100

if policy_filter == "All":
    avg_rl = avg_metrics_full(filtered_df, "rl_agent")
    avg_rand = avg_metrics_full(filtered_df, "random")

    reward_impr = percent_diff(avg_rl[0], avg_rand[0])
    power_saving = percent_diff(avg_rand[1], avg_rl[1])
    temp_saving = percent_diff(avg_rand[2], avg_rl[2])
    cpu_util_diff = percent_diff(avg_rl[3], avg_rand[3])
    gpu_util_diff = percent_diff(avg_rl[4], avg_rand[4])
    ram_diff = percent_diff(avg_rl[5], avg_rand[5])
    swap_diff = percent_diff(avg_rand[6], avg_rl[6])  # Swap inverted for savings

    st.subheader("Overall Summary Metrics")
    cols = st.columns(4)

    metrics_rl = [
        ("Avg Reward (RL Agent)", f"{avg_rl[0]:.2f}", reward_impr, "% higher vs Random"),
        ("Avg Power (RL Agent)", f"{avg_rl[1]:.2f} mW", power_saving, "% lower power vs Random"),
        ("Avg Temp (RL Agent)", f"{avg_rl[2]:.2f} °C", temp_saving, "% lower temp vs Random"),
    ]
    metrics_rand = [
        ("Avg Reward (Random)", f"{avg_rand[0]:.2f}", None, ""),
        ("Avg Power (Random)", f"{avg_rand[1]:.2f} mW", None, ""),
        ("Avg Temp (Random)", f"{avg_rand[2]:.2f} °C", None, ""),
    ]
    metrics_util_ram_swap_rl = [
        ("Avg CPU Utilization (RL Agent)", f"{avg_rl[3]:.2f} %", cpu_util_diff, "% difference vs Random"),
        ("Avg GPU Utilization (RL Agent)", f"{avg_rl[4]:.2f} %", gpu_util_diff, "% difference vs Random"),
        ("Avg RAM Usage (RL Agent)", f"{avg_rl[5]:.2f} %", ram_diff, "% difference vs Random"),
        ("Avg SWAP Usage (RL Agent)", f"{avg_rl[6]:.2f} %", swap_diff, "% lower swap vs Random"),
    ]
    metrics_util_ram_swap_rand = [
        ("Avg CPU Utilization (Random)", f"{avg_rand[3]:.2f} %", None, ""),
        ("Avg GPU Utilization (Random)", f"{avg_rand[4]:.2f} %", None, ""),
        ("Avg RAM Usage (Random)", f"{avg_rand[5]:.2f} %", None, ""),
        ("Avg SWAP Usage (Random)", f"{avg_rand[6]:.2f} %", None, ""),
    ]

    for label, value, delta, desc in metrics_rl:
        safe_delta = f"{delta:.1f}{desc}" if delta is not None else ""
        cols[0].metric(label, value, delta=safe_delta)

    for label, value, delta, desc in metrics_rand:
        cols[1].metric(label, value, delta="")

    for label, value, delta, desc in metrics_util_ram_swap_rl:
        safe_delta = f"{delta:.1f}{desc}" if delta is not None else ""
        cols[2].metric(label, value, delta=safe_delta)

    for label, value, delta, desc in metrics_util_ram_swap_rand:
        cols[3].metric(label, value, delta="")

else:
    avg_sel = avg_metrics_full(filtered_df, policy_filter)
    st.subheader(f"Summary Metrics for {policy_filter.upper()}")
    cols = st.columns(4)
    metrics_single = [
        ("Average Reward", f"{avg_sel[0]:.2f}"),
        ("Average Power After (mW)", f"{avg_sel[1]:.2f}"),
        ("Average Temperature After (°C)", f"{avg_sel[2]:.2f}"),
        ("Average CPU Utilization After (%)", f"{avg_sel[3]:.2f}"),
        ("Average GPU Utilization After (%)", f"{avg_sel[4]:.2f}"),
        ("Average RAM Usage After (%)", f"{avg_sel[5]:.2f}"),
        ("Average SWAP Usage After (%)", f"{avg_sel[6]:.2f}"),
    ]
    chunk_size = (len(metrics_single) + 3) // 4
    for i, col in enumerate(cols):
        chunk = metrics_single[i*chunk_size:(i+1)*chunk_size]
        for label, value in chunk:
            col.metric(label, value)

st.markdown("<hr style='margin:10px 0; border-color:#76B900;'>", unsafe_allow_html=True)

metrics = {
    "CPU Utilization (%)": "cpu_util",
    "GPU Utilization (%)": "gpu_util",
    "Power (mW)": "power",
    "Temperature (°C)": "temp",
    "RAM Usage": "ram",
    "SWAP Usage": "swap"
}

tab_labels = list(metrics.keys()) + ["Reward Distribution", "Action Distribution"]
tabs = st.tabs(tab_labels)

for tab, label in zip(tabs, tab_labels):
    with tab:
        if label in metrics:
            metric_base = metrics[label]
            if policy_filter == "All":
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("RL Agent")
                    df_rl = filtered_df[filtered_df["policy_type"] == "rl_agent"]
                    seq_rl = build_natural_sequence(df_rl, metric_base)
                    fig_rl = px.line(
                        seq_rl,
                        x="timestamp",
                        y=metric_base,
                        title=f"RL Agent - {label}",
                        labels={"timestamp": "Time", metric_base: label},
                        height=220,
                        color_discrete_sequence=[nvidia_green]
                    )
                    fig_rl = style_fig(fig_rl, ["rl_agent"])
                    st.plotly_chart(fig_rl, use_container_width=True)

                with col2:
                    st.subheader("Random Policy")
                    df_rand = filtered_df[filtered_df["policy_type"] == "random"]
                    seq_rand = build_natural_sequence(df_rand, metric_base)
                    fig_rand = px.line(
                        seq_rand,
                        x="timestamp",
                        y=metric_base,
                        title=f"Random Policy - {label}",
                        labels={"timestamp": "Time", metric_base: label},
                        height=220,
                        color_discrete_sequence=["#9dbf56"]
                    )
                    fig_rand = style_fig(fig_rand, ["random"])
                    st.plotly_chart(fig_rand, use_container_width=True)

            else:
                df_sel = filtered_df[filtered_df["policy_type"] == policy_filter]
                seq_sel = build_natural_sequence(df_sel, metric_base)
                fig_sel = px.line(
                    seq_sel,
                    x="timestamp",
                    y=metric_base,
                    title=f"{policy_filter.upper()} - {label}",
                    labels={"timestamp": "Time", metric_base: label},
                    height=220,
                    color_discrete_sequence=[nvidia_green if policy_filter == "rl_agent" else "#9dbf56"]
                )
                fig_sel = style_fig(fig_sel, [policy_filter])
                st.plotly_chart(fig_sel, use_container_width=True)

        elif label == "Reward Distribution":
            if policy_filter == "All":
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("RL Agent")
                    df_rl = filtered_df[filtered_df["policy_type"] == "rl_agent"]
                    fig_rl = px.histogram(df_rl, x="reward_score", nbins=40, title="RL Agent - Reward Distribution", height=280,
                                          color_discrete_sequence=[nvidia_green])
                    fig_rl = style_fig(fig_rl, ["rl_agent"])
                    st.plotly_chart(fig_rl, use_container_width=True)

                with col2:
                    st.subheader("Random Policy")
                    df_rand = filtered_df[filtered_df["policy_type"] == "random"]
                    fig_rand = px.histogram(df_rand, x="reward_score", nbins=40, title="Random Policy - Reward Distribution", height=280,
                                            color_discrete_sequence=["#9dbf56"])
                    fig_rand = style_fig(fig_rand, ["random"])
                    st.plotly_chart(fig_rand, use_container_width=True)

            else:
                df_sel = filtered_df[filtered_df["policy_type"] == policy_filter]
                fig_reward = px.histogram(df_sel, x="reward_score", nbins=40,
                                          title=f"Reward Distribution for {policy_filter.upper()} Policy", height=280,
                                          color_discrete_sequence=[nvidia_green if policy_filter == "rl_agent" else "#9dbf56"])
                fig_reward = style_fig(fig_reward, [policy_filter])
                st.plotly_chart(fig_reward, use_container_width=True)

        elif label == "Action Distribution":
            if policy_filter == "All":
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("RL Agent")
                    df_rl = filtered_df[filtered_df["policy_type"] == "rl_agent"]
                    action_counts_rl = df_rl["action"].value_counts(normalize=True).reset_index()
                    action_counts_rl.columns = ["action", "percentage"]
                    action_counts_rl["percentage"] *= 100  # convert to %
                    fig_rl = px.bar(
                        action_counts_rl,
                        x="action",
                        y="percentage",
                        title="RL Agent - Action Distribution (%)",
                        height=220,
                        color_discrete_sequence=[nvidia_green]
                    )
                    fig_rl = style_fig(fig_rl, ["rl_agent"])
                    st.plotly_chart(fig_rl, use_container_width=True)

                with col2:
                    st.subheader("Random Policy")
                    df_rand = filtered_df[filtered_df["policy_type"] == "random"]
                    action_counts_rand = df_rand["action"].value_counts(normalize=True).reset_index()
                    action_counts_rand.columns = ["action", "percentage"]
                    action_counts_rand["percentage"] *= 100  # convert to %
                    fig_rand = px.bar(
                        action_counts_rand,
                        x="action",
                        y="percentage",
                        title="Random Policy - Action Distribution (%)",
                        height=220,
                        color_discrete_sequence=["#9dbf56"]
                    )
                    fig_rand = style_fig(fig_rand, ["random"])
                    st.plotly_chart(fig_rand, use_container_width=True)

            else:
                df_sel = filtered_df[filtered_df["policy_type"] == policy_filter]
                action_counts_sel = df_sel["action"].value_counts(normalize=True).reset_index()
                action_counts_sel.columns = ["action", "percentage"]
                action_counts_sel["percentage"] *= 100  # convert to %
                fig_action = px.bar(
                    action_counts_sel,
                    x="action",
                    y="percentage",
                    title=f"Action Distribution for {policy_filter.upper()} Policy (%)",
                    height=220,
                    color_discrete_sequence=[nvidia_green if policy_filter == "rl_agent" else "#9dbf56"]
                )
                fig_action = style_fig(fig_action, [policy_filter])
                st.plotly_chart(fig_action, use_container_width=True)

