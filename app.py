import streamlit as st

from src.dashboard.final_project_app import render_final_project_app


st.set_page_config(page_title="Computer Science Journal Finder", layout="wide")
render_final_project_app()
