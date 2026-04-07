import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import random

st.title("🚗 Smart Traffic Prediction System")

# -----------------------------
# Generate Dummy Dataset
# -----------------------------
n = 300
data = pd.DataFrame({
    "Speed": np.random.randint(5, 80, n),
    "Distance": np.random.randint(1, 20, n),
    "Vibration": np.random.uniform(0, 1, n),
    "Vehicles": np.random.randint(10, 150, n)
})

def classify(row):
    if row["Speed"] < 20 and row["Distance"] < 5:
        return "Heavy"
    elif row["Speed"] < 40:
        return "High"
    else:
        return "Low"

data["Traffic"] = data.apply(classify, axis=1)

# -----------------------------
# Train Model
# -----------------------------
X = data[["Speed", "Distance", "Vibration", "Vehicles"]]
y = data["Traffic"]

model = RandomForestClassifier()
model.fit(X, y)

# -----------------------------
# User Input Section
# -----------------------------
st.header("📊 Enter Sensor Values")

speed = st.slider("Speed (km/h)", 0, 100, 30)
distance = st.slider("Distance (m)", 1, 20, 10)
vibration = st.slider("Road Condition (0 smooth - 1 rough)", 0.0, 1.0, 0.3)
vehicles = st.slider("Number of Vehicles", 0, 200, 50)

# Prediction
if st.button("Predict Traffic"):
    result = model.predict([[speed, distance, vibration, vehicles]])
    st.success(f"🚦 Traffic Level: {result[0]}")

# -----------------------------
# Real-Time Simulation
# -----------------------------
st.header("🔄 Live Traffic Simulation")

if st.button("Generate Live Data"):
    live_data = [
        random.randint(5, 80),
        random.randint(1, 20),
        random.uniform(0, 1),
        random.randint(10, 150)
    ]
    
    pred = model.predict([live_data])
    
    st.write("Live Sensor Data:", live_data)
    st.success(f"Predicted Traffic: {pred[0]}")

# -----------------------------
# Graph Visualization
# -----------------------------
st.header("📈 Traffic Data Visualization")

fig, ax = plt.subplots()
ax.hist(data["Speed"], bins=20)
ax.set_title("Speed Distribution")

st.pyplot(fig)