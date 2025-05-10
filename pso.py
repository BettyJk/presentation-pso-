import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

# === Titre principal ===
st.title("Démo : Optimisation de l'Ordonnancement par PSO")

# === Objectif ===
st.markdown("## 🎯 Objectif")
st.markdown("Minimiser la durée globale du projet tout en respectant les dépendances entre tâches.")

# === Données exemple - Avant optimisation ===
df_pre = pd.DataFrame({
    'Tâche': ['Tâche 1', 'Tâche 2', 'Tâche 3'],
    'Début': [0, 3, 7],
    'Durée': [3, 4, 2],
    'Fin': [3, 7, 9]
})

# === Diagramme de Gantt - Avant optimisation ===
st.markdown("### 🟠 Diagramme de Gantt - Avant optimisation")

fig1 = go.Figure()

for i, row in df_pre.iterrows():
    fig1.add_trace(go.Bar(
        x=[row['Durée']],
        y=[row['Tâche']],
        base=[row['Début']],
        orientation='h',
        name=row['Tâche'],
        hovertext=f"{row['Tâche']}: {row['Début']} → {row['Fin']}"
    ))

fig1.update_layout(
    barmode='stack',
    title='Planification avant optimisation',
    xaxis_title='Temps',
    yaxis_title='Tâche',
    yaxis=dict(autorange='reversed')
)

st.plotly_chart(fig1)

# === Données exemple - Après optimisation par PSO ===
df_post = pd.DataFrame({
    'Tâche': ['Tâche 1', 'Tâche 2', 'Tâche 3'],
    'Début': [0, 2, 5],
    'Durée': [2, 3, 2],
    'Fin': [2, 5, 7]
})

# === Diagramme de Gantt - Après optimisation ===
st.markdown("### 🟢 Diagramme de Gantt - Après optimisation (via PSO)")

fig2 = go.Figure()

for i, row in df_post.iterrows():
    fig2.add_trace(go.Bar(
        x=[row['Durée']],
        y=[row['Tâche']],
        base=[row['Début']],
        orientation='h',
        name=row['Tâche'],
        hovertext=f"{row['Tâche']}: {row['Début']} → {row['Fin']}"
    ))

fig2.update_layout(
    barmode='stack',
    title='Planification après optimisation par PSO',
    xaxis_title='Temps',
    yaxis_title='Tâche',
    yaxis=dict(autorange='reversed')
)

st.plotly_chart(fig2)

# === Graphe de convergence ===
st.markdown("### 📉 Graphe de convergence (Fitness vs. Itération)")

iterations = np.arange(1, 21)
fitness = np.exp(-0.2 * iterations) * 100 + np.random.normal(0, 2, 20)  # Simulation réaliste

plt.figure(figsize=(8, 4))
plt.plot(iterations, fitness, marker='o', color='green')
plt.title("Convergence de l'algorithme PSO")
plt.xlabel("Itération")
plt.ylabel("Fitness")
plt.grid(True)
st.pyplot(plt)
