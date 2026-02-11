import sys
import types
from io import BytesIO
import base64
import streamlit as st
import os
import torch
import warnings
import logging

# 🎯 CRITICAL: Must be the VERY FIRST Streamlit command
st.set_page_config(
    page_title="Color Visualizer Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --- WARNING SHIELD: Titanium Silence v4 ---
st.components.v1.html("""
    <script>
        (function() {
            const silence = (w) => {
                try {
                    if (!w || !w.console || w.console.__isMuted) return;
                    ['warn', 'error', 'log'].forEach(m => {
                        const original = w.console[m];
                        if (!original) return;
                        w.console[m] = function(...args) {
                            try {
                                const msg = String(args[0] || "");
                                if (/Invalid color|theme\\.sidebar|widgetBackground|skeletonBackground|Unrecognized feature|ambient-light|battery|wake-lock|sandbox|document-domain|oversized-images|vr|fragment rerun/i.test(msg)) return;
                            } catch(e) {}
                            original.apply(this, args);
                        };
                    });
                    w.console.__isMuted = true;
                } catch(e) {}
            };
            const run = () => {
                silence(window);
                try { if (window.parent && window.parent !== window) silence(window.parent); } catch(e) {}
            };
            run();
            setInterval(run, 500);
        })();
    </script>
""", height=0)

# --- UTILITIES IMPORT ---
from utils.encoding import image_to_url_patch
from utils.sam_loader import get_sam_engine, ensure_model_exists, CHECKPOINT_PATH, MODEL_TYPE
from utils.state_manager import initialize_session_state
from utils.ui_components import setup_styles, render_sidebar, render_visualizer_engine_v11
from config.constants import PerformanceConfig

# --- MONKEY PATCHING ---
# Create spoof modules for older/newer Streamlit library structures
for mod_name in ["streamlit.elements.lib", "streamlit.elements.lib.image_utils"]:
    if mod_name not in sys.modules:
        m = types.ModuleType(mod_name)
        m.image_to_url = image_to_url_patch
        sys.modules[mod_name] = m
try:
    import streamlit.elements.lib as sel
    sel.image_utils = sys.modules["streamlit.elements.lib.image_utils"]
except Exception as e:
    # Monkey patching failed - may affect image display
    import logging
    logging.warning(f"Failed to patch streamlit.elements.lib: {e}")

# Patch the standard image module immediately
import streamlit.elements.image as st_image
st_image.image_to_url = image_to_url_patch

# --- FRAGMENT BACKWARD COMPATIBILITY ---
if not hasattr(st, 'fragment'):
    if hasattr(st, 'experimental_fragment'): st.fragment = st.experimental_fragment
    else:
        def fragment_noop(func=None, **kwargs):
            if func is None: return lambda f: f
            return func
        st.fragment = fragment_noop

# --- WARNING SHIELD ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("timm").setLevel(logging.ERROR)
logging.getLogger("mobile_sam").setLevel(logging.ERROR)

# --- RESOURCES ---
os.environ["OMP_NUM_THREADS"] = str(PerformanceConfig.OMP_NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(PerformanceConfig.MKL_NUM_THREADS)
torch.set_num_threads(PerformanceConfig.TORCH_NUM_THREADS)
torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = False

def main():
    setup_styles()
    initialize_session_state()
    # Reset loop guard for the new app cycle
    st.session_state["loop_guarded"] = False
    ensure_model_exists()
    
    # Identify device for engine optimization
    device_str = "cpu"
    if torch.cuda.is_available(): device_str = "cuda"
    elif torch.backends.mps.is_available(): device_str = "mps"

    sam = get_sam_engine(CHECKPOINT_PATH, MODEL_TYPE)
    q_params = st.query_params
    print(f"DEBUG: APP RERUN - Tool: {st.session_state.get('selection_tool')}, Params: {list(q_params.keys())}")
    
    render_sidebar(sam, device_str)

    if st.session_state.get("image") is not None:
        render_visualizer_engine_v11(800)
    else:
        # Landing Page
        empty_top = st.empty()
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.markdown("""
            <div class="landing-container">
                <div class="landing-header">
                    <h1>Welcome to Color Visualizer</h1>
                </div>
                <div class="landing-sub">
                    <p>Upload a photo of your room to start experimenting with colors.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.info("👈 Use the sidebar to upload an image.")

    # --- 🤖 HIDDEN TECHNICAL BRIDGE (Bottom of script) ---
    st.markdown('<div id="global-sync-anchor"></div>', unsafe_allow_html=True)
    st.button("GLOBAL SYNC", key="global_sync_btn", help="Hidden sync for JS", type="secondary")
    st.markdown('<div class="global-sync-marker" style="display:none;" data-sync-id="global_sync"></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
