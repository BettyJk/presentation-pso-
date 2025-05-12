import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt

# === Task definitions ===
tasks = [
    {"name": "Tâche 1", "duration": 2, "dependencies": []},
    {"name": "Tâche 2", "duration": 3, "dependencies": [0]},  # Depends on Task 1
    {"name": "Tâche 3", "duration": 2, "dependencies": [1]}   # Depends on Task 2
]

n_tasks = len(tasks)

# === Objective Function ===
def evaluate(position):
    start_times = list(position)
    penalty = 0
    finish_times = [start_times[i] + tasks[i]['duration'] for i in range(n_tasks)]

    for i, task in enumerate(tasks):
        for dep in task["dependencies"]:
            if start_times[i] < finish_times[dep]:
                penalty += (finish_times[dep] - start_times[i]) * 100  # heavy penalty

    makespan = max(finish_times)
    return makespan + penalty

# === PSO Implementation ===
class Particle:
    def __init__(self):
        self.position = np.random.uniform(0, 10, n_tasks)
        self.velocity = np.random.uniform(-1, 1, n_tasks)
        self.best_position = self.position.copy()
        self.best_score = evaluate(self.position)

    def update_velocity(self, global_best, w=0.5, c1=1.5, c2=1.5):
        r1, r2 = np.random.rand(n_tasks), np.random.rand(n_tasks)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best - self.position)
        self.velocity = w * self.velocity + cognitive + social

    def move(self):
        self.position += self.velocity
        self.position = np.clip(self.position, 0, 20)
        score = evaluate(self.position)
        if score < self.best_score:
            self.best_score = score
            self.best_position = self.position.copy()
        return score

# === PSO Loop ===
def run_pso(n_particles=30, n_iterations=20):
    swarm = [Particle() for _ in range(n_particles)]
    global_best = min(swarm, key=lambda p: p.best_score).best_position.copy()
    best_score = evaluate(global_best)

    history = []
    for _ in range(n_iterations):
        for particle in swarm:
            particle.update_velocity(global_best)
            score = particle.move()
            if score < evaluate(global_best):
                global_best = particle.position.copy()
                best_score = score
        history.append(best_score)
    return global_best, history

# === Run PSO ===
best_start_times, convergence = run_pso()

# === Prepare data ===
def get_schedule_df(start_times):
    data = []
    for i, task in enumerate(tasks):
        start = start_times[i]
        duration = task["duration"]
        data.append({
            "Tâche": task["name"],
            "Début": round(start, 2),
            "Durée": duration,
            "Fin": round(start + duration, 2)
        })
    return pd.DataFrame(data)

df_before = pd.DataFrame({
    'Tâche': [t["name"] for t in tasks],
    'Début': [0, 3, 7],
    'Durée': [t["duration"] for t in tasks],
    'Fin': [3, 6, 9]
})

df_after = get_schedule_df(best_start_times)

# === Streamlit Output ===
st.title("PSO Scheduling Optimizer Demo")

st.markdown("## 🎯 Objectif")
st.markdown("Minimiser la durée globale du projet tout en respectant les dépendances entre tâches.")

# Before Optimization
st.markdown("### 🟠 Diagramme de Gantt - Avant optimisation")
fig1 = go.Figure()
for _, row in df_before.iterrows():
    fig1.add_trace(go.Bar(
        x=[row['Durée']],
        y=[row['Tâche']],
        base=[row['Début']],
        orientation='h',
        name=row['Tâche'],
        hovertext=f"{row['Tâche']}: {row['Début']} → {row['Fin']}"
    ))
fig1.update_layout(barmode='stack', title='Avant optimisation', xaxis_title='Temps',
                   yaxis_title='Tâche', yaxis=dict(autorange='reversed'))
st.plotly_chart(fig1)

# After Optimization
st.markdown("### 🟢 Diagramme de Gantt - Après optimisation (via PSO)")
fig2 = go.Figure()
for _, row in df_after.iterrows():
    fig2.add_trace(go.Bar(
        x=[row['Durée']],
        y=[row['Tâche']],
        base=[row['Début']],
        orientation='h',
        name=row['Tâche'],
        hovertext=f"{row['Tâche']}: {row['Début']} → {row['Fin']}"
    ))
fig2.update_layout(barmode='stack', title='Après optimisation par PSO', xaxis_title='Temps',
                   yaxis_title='Tâche', yaxis=dict(autorange='reversed'))
st.plotly_chart(fig2)

# Convergence
st.markdown("### 📉 Graphe de convergence (Fitness vs. Itération)")
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(convergence)+1), convergence, marker='o', color='green')
plt.title("Convergence réelle de l'algorithme PSO")
plt.xlabel("Itération")
plt.ylabel("Fitness")
plt.grid(True)
st.pyplot(plt)
