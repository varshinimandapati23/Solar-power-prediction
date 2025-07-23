import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import time



# --- Load Model ---
model = joblib.load("ac_power_model_cleaned.pkl")

# --- Expected Feature Order ---
expected_features = ['HOUR_cos', 'HOUR_sin', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']

# --- SQLite Setup ---
conn = sqlite3.connect("user_data.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT
)
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS logs (
    username TEXT,
    timestamp TEXT,
    hour INTEGER,
    ambient_temp REAL,
    module_temp REAL,
    irradiation REAL,
    predicted_power REAL
)
''')
conn.commit()

# --- Password Hashing ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- User Auth Functions ---
def register_user(username, password):
    try:
        hashed_pw = hash_password(password)
        cursor.execute("INSERT INTO users VALUES (?, ?)", (username, hashed_pw))
        conn.commit()
        return True
    except Exception:
        return False

def login_user(username, password):
    hashed_pw = hash_password(password)
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, hashed_pw))
    return cursor.fetchone() is not None

# --- Sidebar Login/Register ---
def login_section():
    st.title("🔋 Solar AC Power Prediction")
    st.subheader("Login/Register to estimate AC power output based on environmental conditions")

    st.sidebar.title("🔐 Login")
    auth_mode = st.sidebar.radio("Select", ["Login", "Register"])

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if auth_mode == "Register":
        if st.sidebar.button("Register"):
            if register_user(username, password):
                st.sidebar.success("User registered!")
            else:
                st.sidebar.error("Username already exists.")

    if st.sidebar.button("Login"):
        if login_user(username, password):
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["login_time"] = time.time()
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")

# --- Prediction Function ---
def predict_ac_power(irradiation, module_temp, ambient_temp, hour):
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    input_df = pd.DataFrame([[
        irradiation,
        module_temp,
        ambient_temp,
        hour_sin,
        hour_cos
    ]], columns = ['IRRADIATION', 'MODULE_TEMPERATURE', 'AMBIENT_TEMPERATURE', 'HOUR_sin', 'HOUR_cos'])

    return model.predict(input_df)[0], input_df

# --- Logging Function ---
def log_prediction(username, hour, ambient_temp, module_temp, irradiation, predicted_power):
    cursor.execute("""
        INSERT INTO logs VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (username, datetime.datetime.now().isoformat(), hour, ambient_temp, module_temp, irradiation, predicted_power))
    conn.commit()

# --- Main App ---
def main_app():
    session_timeout_minutes = 10
    if time.time() - st.session_state.get("login_time", 0) > session_timeout_minutes * 60:
        st.warning("Session expired. Please login again.")
        st.session_state.clear()
        st.rerun()

    if st.sidebar.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()

    tab1, tab2 = st.tabs(["📈 Predict", "📊 Logs"])

    if "username" not in st.session_state:
        st.error("Please login first.")
        return

    username = st.session_state["username"]

    with tab1:
        st.sidebar.title("🧾 Provide Inputs")

        hour = st.sidebar.slider("Hour of the Day", 0, 23, 12)
        ambient_temp = st.sidebar.number_input("Ambient Temperature (°C)", 0.0, 100.0, 25.0)
        module_temp = st.sidebar.number_input("Module Temperature (°C)", 0.0, 100.0, 30.0)
        irradiation = st.sidebar.number_input("Irradiation (W/m²)", 0.0, 1500.0, 800.0)

        if st.sidebar.button("Predict AC Power"):
            predicted_power, input_df = predict_ac_power(hour, ambient_temp, module_temp, irradiation)

            st.title("🔋 Solar AC Power Prediction")
            st.subheader("Input Summary")
            st.metric("Hour", hour)
            st.metric("Ambient Temp", f"{ambient_temp} °C")
            st.metric("Module Temp", f"{module_temp} °C")
            st.metric("Irradiation", f"{irradiation} W/m²")
            st.success(f"Predicted AC Power: {predicted_power:.2f} kW")

            log_prediction(username, hour, ambient_temp, module_temp, irradiation, predicted_power)

    with tab2:
        st.title("📜 Prediction Logs")
        cursor.execute("SELECT * FROM logs WHERE username=?", (username,))
        logs = cursor.fetchall()
        log_df = pd.DataFrame(logs, columns=["User", "Timestamp", "Hour", "Ambient Temp", "Module Temp", "Irradiation", "Predicted Power"])

        log_df["Date"] = pd.to_datetime(log_df["Timestamp"]).dt.date
        selected_date = st.date_input("Filter by Date", value=datetime.date.today())
        filtered_logs = log_df[log_df["Date"] == selected_date]
        st.dataframe(filtered_logs.drop(columns=["User", "Date"]))

        if not filtered_logs.empty:
            st.subheader("📊 Daily Power Trend")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=filtered_logs, x="Hour", y="Predicted Power", marker="o", ax=ax)
            st.pyplot(fig)

        csv = filtered_logs.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download Logs as CSV", data=csv, file_name="ac_power_logs.csv", mime="text/csv")

# --- Run ---
st.set_page_config(page_title="Solar Power Predictor", layout="wide")

if "logged_in" not in st.session_state:
    login_section()
else:
    main_app()
