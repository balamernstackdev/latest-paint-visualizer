"""
Styles module - CSS styling setup for the AI Paint Visualizer.

This module manages all CSS styling including theme colors, responsive layouts,
sidebar styling, mobile optimizations, and canvas display formatting.
"""

import streamlit as st
import os


def setup_styles():
    """
    Apply comprehensive CSS styling to the Streamlit application.
    
    Loads external CSS from assets/style.css and applies additional inline styles for:
    - Light theme enforcement
    - Sidebar layout and responsiveness
    - Mobile-specific UI adjustments
    - Canvas display formatting
    - Custom button and control styles
    - Responsive zoom control visibility
    
    The function also injects mobile-specific sidebar state management via CSS
    transitions based on session state.
    
    Note:
        This function should be called once during app initialization.
        The extensive inline CSS is temporarily kept here pending further
        modularization into separate CSS files.
    """
    # Load external CSS file if available
    css_path = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "style.css")
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # NOTE: The extensive inline CSS (530+ lines) from the original ui_components.py
    # is temporarily preserved for backward compatibility. Future refactoring should
    # move this to separate CSS files in assets/css/ for better maintainability.
    from ..ui_components import setup_styles as _original_setup_styles
    _original_setup_styles()
