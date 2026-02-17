import streamlit as st
import subprocess
import tempfile
import os
import sys
from pathlib import Path
import PIL.Image
from setuptools import setup
st.set_page_config(page_title="readgssi GPR Viewer", layout="wide")
st.title("🛰️ readgssi GPR Data Viewer")
st.markdown("Upload a GSSI GPR (`.dzt`) file to view a basic radargram and metadata.")

# --- Check if readgssi module is available ---
def check_readgssi():
    try:
        # Use python -m readgssi to ensure we use the current environment's interpreter
        result = subprocess.run(
            [sys.executable, "-m", "readgssi", "-V"],
            capture_output=True, text=True, check=True
        )
        return True, result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, ImportError):
        return False, None

is_available, version_info = check_readgssi()
if not is_available:
    st.error(
        "`readgssi` module not found. Please ensure it is installed in the current Python environment:\n"
        "```bash\npip install readgssi\n```"
    )
    st.stop()
else:
    st.sidebar.success(f"✅ readgssi {version_info}")

# --- Sidebar for processing options ---
st.sidebar.header("Processing Options")
gain = st.sidebar.selectbox(
    "Gain type",
    options=["none", "agc", "ew"],
    index=1,
    help="AGC (automatic gain control) or EW (exponential weighted). 'none' for raw data."
)
bandpass = st.sidebar.text_input(
    "Bandpass filter (MHz)",
    value="",
    placeholder="e.g., 100 300",
    help="Two frequencies: lowcut highcut. Leave empty to disable."
)
color_scheme = st.sidebar.selectbox(
    "Color scheme",
    options=["seismic", "gray", "viridis"],
    index=0
)

# --- File uploader ---
uploaded_file = st.file_uploader("Choose a GPR file", type=['dzt'], help="Supports GSSI .dzt files")

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dzt") as tmp_input:
        tmp_input.write(uploaded_file.getvalue())
        input_path = tmp_input.name

    # Prepare output files in a temporary directory
    with tempfile.TemporaryDirectory() as output_dir:
        output_image = Path(output_dir) / "radargram.png"
        output_meta = Path(output_dir) / "metadata.txt"

        # Build the readgssi command using python -m readgssi
        cmd = [
            sys.executable, "-m", "readgssi",
            "-i", input_path,
            "-o", str(output_image),
            "-gz", gain,
            "--cs", color_scheme,
            "--write-metadata", str(output_meta),
            "--dpi", "150",
            "--width", "12"
        ]

        # Add bandpass filter if provided
        if bandpass:
            try:
                low, high = bandpass.split()
                cmd.extend(["--bp", low, high])
            except ValueError:
                st.sidebar.warning("Bandpass must be two numbers separated by space. Disabling filter.")

        # Add progress bar and run command
        with st.spinner("Processing file with readgssi..."):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)

                # --- Display results ---
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.subheader("📊 Radargram")
                    if output_image.exists():
                        image = PIL.Image.open(output_image)
                        st.image(image, use_container_width=True)
                    else:
                        st.warning("No image was generated.")

                with col2:
                    st.subheader("📄 Metadata")
                    if output_meta.exists():
                        with open(output_meta, "r") as f:
                            meta_content = f.read()
                        st.text_area("File metadata", meta_content, height=400)
                    else:
                        st.info("Metadata file not found.")

                    with st.expander("📋 Processing Log"):
                        st.text(result.stdout)
                        if result.stderr:
                            st.error(result.stderr)

            except subprocess.TimeoutExpired:
                st.error("Processing timed out. The file might be too large or corrupted.")
            except subprocess.CalledProcessError as e:
                st.error(f"readgssi command failed with error:\n{e.stderr}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

    # Clean up the temporary input file
    os.unlink(input_path)

else:
    st.info("👆 Please upload a .dzt file to begin.")

# --- Usage instructions ---
with st.sidebar.expander("ℹ️ How to use"):
    st.markdown("""
    1.  Ensure `readgssi` is installed (`pip install readgssi`).
    2.  Upload a GSSI `.dzt` file.
    3.  Adjust processing options on the left.
    4.  The radargram image and metadata will appear automatically.

    **Note:** This is a basic interface. For full `readgssi` capabilities, use the command line.
    """)




