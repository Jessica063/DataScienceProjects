# 📊 Streamlit Dashboard: Employee Ticket Optimization
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary
import random

# Page configuration
st.set_page_config(page_title="Ticket Assignment Optimizer", layout="wide")
st.markdown("""
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem;}
        .css-1d391kg {margin-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("🎯 Employee Ticket Assignment Optimizer")
st.markdown("Optimize ticket distribution among employees to minimize response time.")

# File upload section
with st.sidebar:
    st.header("📄 Upload Data")
    emp_file = st.file_uploader("Upload Employees CSV", type="csv")
    tick_file = st.file_uploader("Upload Tickets CSV", type="csv")

if emp_file and tick_file:
    # Load data
    employees_df = pd.read_csv(emp_file)
    tickets_df = pd.read_csv(tick_file)
    employees_df['Skills'] = employees_df['Skills'].apply(lambda x: x.split(','))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👥 Employees")
        st.dataframe(employees_df, height=300)
    with col2:
        st.subheader("🎫 Tickets")
        st.dataframe(tickets_df, height=300)

    # Prepare valid assignment mappings
    assignments = [
        (t_idx, e_idx)
        for t_idx, ticket in tickets_df.iterrows()
        for e_idx, emp in employees_df.iterrows()
        if ticket['Issue Type'] in emp['Skills']
    ]

    # Linear Programming model
    model = LpProblem("Ticket_Assignment_Optimization", LpMinimize)
    x = LpVariable.dicts("assign", assignments, cat=LpBinary)
    priority_weights = {'high': 3, 'medium': 2, 'low': 1}

    model += lpSum([
        x[(t, e)] * tickets_df.loc[t, 'Response Time'] * priority_weights[tickets_df.loc[t, 'Priority']]
        for (t, e) in assignments
    ])

    for t in tickets_df.index:
        model += lpSum([x[(t, e)] for (tt, e) in assignments if tt == t]) == 1

    avg_load = len(tickets_df) / len(employees_df)
    max_load = int(avg_load + 1)
    for e in employees_df.index:
        model += lpSum([x[(t, e)] for (t, ee) in assignments if ee == e]) <= max_load
        model += lpSum([x[(t, e)] for (t, ee) in assignments if ee == e]) >= 1

    model.solve()

    # Extract assignment results
    results = [
        {
            'Ticket ID': tickets_df.loc[t, 'Ticket ID'],
            'Issue Type': tickets_df.loc[t, 'Issue Type'],
            'Assigned To': employees_df.loc[e, 'Employee'],
            'Response Time': tickets_df.loc[t, 'Response Time'],
            'Priority': tickets_df.loc[t, 'Priority']
        }
        for (t, e) in assignments if x[(t, e)].varValue == 1
    ]
    assignment_df = pd.DataFrame(results)

    # Display assignment results
    st.subheader("📋 Optimized Assignments")
    st.dataframe(assignment_df, height=400)

    # Workload distribution chart
    st.subheader("📊 Workload Distribution")
    workload = assignment_df.groupby('Assigned To').size().reset_index(name='Tickets')
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = sns.color_palette("viridis", len(workload))
    bars = sns.barplot(data=workload, y='Assigned To', x='Tickets', ax=ax, palette=palette)

    # Add average line
    avg = workload['Tickets'].mean()
    ax.axvline(avg, ls='--', color='red', label=f'Avg: {avg:.1f}')
    ax.set_title("Tickets Assigned per Employee", fontsize=12)
    ax.set_xlabel("# Tickets")
    ax.set_ylabel("Employee")
    ax.legend()
    st.pyplot(fig)


