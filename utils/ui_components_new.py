"""
UI Components - Backward compatibility wrapper.

This module maintains backward compatibility while delegating to the new modular
UI package structure. All imports from utils.ui_components will continue to work.

New modular structure:
  - utils.ui.canvas - Canvas wrapper with caching
  - utils.ui.styles - CSS styling setup
  - utils.ui.fragments - Fragment components (color picker, zoom, editor)
  - utils.ui.sidebar - Sidebar controls and management

For new code, prefer importing directly from utils.ui submodules.
"""

# Import all components from new modular structure
from .ui import (
    st_canvas,
    setup_styles,
    sidebar_paint_fragment,
    render_zoom_controls,
    render_editor_fragment,
    sidebar_toggle_fragment,
    render_sidebar
)

# Re-export for backward compatibility
__all__ = [
    'st_canvas',
    'setup_styles',
    'sidebar_paint_fragment',
    'render_zoom_controls',
    'render_editor_fragment',
    'sidebar_toggle_fragment',
    'render_sidebar'
]
