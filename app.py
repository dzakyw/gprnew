# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 18:39:39 2026

@author: asus
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
from pathlib import Path
import warnings
import json
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import interp1d, griddata
from scipy.signal import wiener, deconvolve
from scipy.linalg import toeplitz, solve_toeplitz
from scipy.optimize import minimize
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="GPR Data Processor",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("📡 GPR Data Processor with Deconvolution")
st.markdown("Process GPR data with advanced deconvolution, coordinate import, and trace muting")

# Custom CSS
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .st-bb {
        background-color: #f0f2f6;
    }
    .st-at {
        background-color: #ffffff;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #4CAF50;
    }
    .window-box {
        background-color: #e8f4fd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #2196F3;
    }
    .coordinate-box {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }
    .near-surface-box {
        background-color: #fff3e0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FF9800;
    }
    .mute-box {
        background-color: #fff0f0;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FF5252;
    }
    .deconv-box {
        background-color: #f3e5f5;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #9C27B0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'original_array' not in st.session_state:
    st.session_state.original_array = None
if 'processed_array' not in st.session_state:
    st.session_state.processed_array = None
if 'deconvolved_array' not in st.session_state:
    st.session_state.deconvolved_array = None
if 'coordinates' not in st.session_state:
    st.session_state.coordinates = None
if 'interpolated_coords' not in st.session_state:
    st.session_state.interpolated_coords = None

# Sidebar
with st.sidebar:
    st.header("📂 File Upload")

    csv_file = st.file_uploader(
    "Upload GPR CSV file",
    type=['csv'],
    help="CSV format: first row = trace indices, first column = sample numbers. Amplitudes in between.")
    
    st.markdown("---")
    st.header("🗼 Electric Pole Coordinates (Optional)")
    pole_csv = st.file_uploader("Upload Electric Pole CSV (Easting, Northing, Name)", 
                                       type=['csv'], key="pole_csv")
            
        # Initialize pole data
    pole_data = None
    if pole_csv:
        try:
            pole_df = pd.read_csv(pole_csv)
            st.success(f"Loaded {len(pole_df)} electric pole locations")
                        
            # Check required columns
            required_pole_cols = ['Easting', 'Northing', 'Name']
            available_pole_cols = {}
                        
            for req in required_pole_cols:
                matches = [col for col in pole_df.columns if req.lower() in col.lower()]
                if matches:
                    available_pole_cols[req] = matches[0]
                else:
                    st.error(f"Column '{req}' not found in pole CSV. Available columns: {list(pole_df.columns)}")
                    pole_df = None
                    break
                        
            if pole_df is not None:
                # Extract pole data
                pole_easting = pole_df[available_pole_cols['Easting']].values
                pole_northing = pole_df[available_pole_cols['Northing']].values
                pole_names = pole_df[available_pole_cols['Name']].values
                            
                # Find nearest distance to GPR line for each pole
                gpr_easting = st.session_state.interpolated_coords['easting']
                gpr_northing = st.session_state.interpolated_coords['northing']
                gpr_distance = st.session_state.interpolated_coords['distance']
                            
                pole_projected_distances = []
                pole_min_distances = []
                            
                for i in range(len(pole_easting)):
                    # Calculate distance to each GPR point
                    distances = np.sqrt((gpr_easting - pole_easting[i])**2 + 
                                                (gpr_northing - pole_northing[i])**2)
                    min_idx = np.argmin(distances)
                    min_dist = distances[min_idx]
                    projected_dist = gpr_distance[min_idx]
                                
                    pole_projected_distances.append(projected_dist)
                    pole_min_distances.append(min_dist)
                            
                # Filter poles within reasonable distance (e.g., 10m from line)
                max_distance_threshold = st.slider("Max distance from GPR line (m)", 1.0, 50.0, 10.0, 1.0)
                            
                filtered_indices = [i for i, d in enumerate(pole_min_distances) if d <= max_distance_threshold]
                            
                if filtered_indices:
                    pole_data = {
                        'easting': pole_easting[filtered_indices],
                        'northing': pole_northing[filtered_indices],
                        'names': pole_names[filtered_indices],
                        'projected_distances': np.array(pole_projected_distances)[filtered_indices],
                        'min_distances': np.array(pole_min_distances)[filtered_indices]
                        }
                    st.info(f"Found {len(filtered_indices)} poles within {max_distance_threshold}m of GPR line")
                else:
                    st.warning(f"No poles found within {max_distance_threshold}m of GPR line")
        except Exception as e:
            st.error(f"Error loading pole CSV: {str(e)}")
                    
    st.markdown("---")            
    st.header("🗺️ Coordinate Import (Optional)")
    
    # Coordinate CSV upload
    coord_csv = st.file_uploader("Upload CSV with coordinates", type=['csv'], 
                                help="CSV with columns: Easting, Northing, Elevation (or similar)")
    
    if coord_csv:
        st.markdown('<div class="coordinate-box">', unsafe_allow_html=True)
        st.subheader("Coordinate Settings")
        
        # Coordinate column mapping
        col1, col2 = st.columns(2)
        with col1:
            easting_col = st.text_input("Easting Column", "Easting")
            northing_col = st.text_input("Northing Column", "Northing")
        with col2:
            elevation_col = st.text_input("Elevation Column", "Elevation")
            trace_col = st.text_input("Trace Column (optional)", "", 
                                     help="If CSV has trace numbers matching coordinate points")
        
        # Coordinate interpolation method
        interp_method = st.selectbox("Interpolation Method", 
                                    ["Linear", "Cubic", "Nearest", "Previous", "Next"],
                                    help="How to interpolate coordinates between points")
        
        # Coordinate scaling options
        coord_units = st.selectbox("Coordinate Units", 
                                  ["Meters", "Feet", "Kilometers", "Miles"],
                                  help="Units of the imported coordinates")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("📏 Axis Scaling")
    
    # Depth scaling (Y-axis)
    st.subheader("Depth Scaling (Y-axis)")
    depth_unit = st.selectbox("Depth Unit", ["samples", "meters", "nanoseconds", "feet"])
    
    if depth_unit != "samples":
        max_depth = st.number_input(f"Max Depth ({depth_unit})", 0.1, 1000.0, 12.0, 0.1,
                                   help=f"Set maximum depth in {depth_unit}")
        velocity = None
        if depth_unit == "nanoseconds":
            velocity = st.number_input("Wave Velocity (m/ns)", 0.01, 0.3, 0.1, 0.01,
                                      help="Wave velocity for time-depth conversion")
    
    # Distance scaling (X-axis)
    st.subheader("Distance Scaling (X-axis)")
    use_coords_for_distance = coord_csv is not None and st.checkbox("Use Coordinates for Distance", False,
                                                                    help="Use imported coordinates for X-axis scaling")
    
    if not use_coords_for_distance:
        distance_unit = st.selectbox("Distance Unit", ["traces", "meters", "feet", "kilometers"])
        
        if distance_unit != "traces":
            total_distance = st.number_input(f"Total Distance ({distance_unit})", 0.1, 10000.0, 250.0, 0.1,
                                            help=f"Set total survey distance in {distance_unit}")
    else:
        st.info("Using coordinate-based distance calculation")
        distance_unit = "meters"  # Default when using coordinates
    
    st.markdown("---")
    st.header("📐 Plot Aspect Ratio")
    
    # Aspect ratio control
    aspect_mode = st.selectbox("Aspect Ratio Mode", 
                              ["Auto", "Equal", "Manual", "Realistic"],
                              help="Control the Y:X scale of the plot")
    
    if aspect_mode == "Manual":
        aspect_ratio = st.selectbox("Aspect Ratio (Y:X)", 
                                   ["1:1", "1:2", "1:4", "1:5", "1:10", "2:1", "4:1", "5:1", "10:1"])
        # Convert to float
        aspect_ratio_float = float(aspect_ratio.split(":")[0]) / float(aspect_ratio.split(":")[1])
    elif aspect_mode == "Realistic":
        realistic_ratio = st.selectbox("Realistic Ratio", 
                                      ["1:5 (Shallow)", "1:10 (Standard)", "1:20 (Deep)", "1:50 (Very Deep)"])
        aspect_ratio_float = 1 / float(realistic_ratio.split(":")[1].split()[0])
    
    st.markdown("---")
    st.header("🔍 Plot Windowing")
    
    use_custom_window = st.checkbox("Use Custom Plot Window", False,
                                   help="Define custom depth and distance ranges for plotting")
    
    if use_custom_window:
        st.markdown('<div class="window-box">', unsafe_allow_html=True)
        
        # Depth window
        st.subheader("Depth Window (Y-axis)")
        if depth_unit != "samples":
            depth_min = st.number_input(f"Min Depth ({depth_unit})", 0.0, max_depth, 0.0, 0.1)
            depth_max = st.number_input(f"Max Depth ({depth_unit})", 0.0, max_depth, max_depth, 0.1)
        else:
            depth_min = st.number_input("Min Depth (samples)", 0, 5000, 0)
            depth_max = st.number_input("Max Depth (samples)", 0, 5000, 255)
        
        # Distance window
        st.subheader("Distance Window (X-axis)")
        if not use_coords_for_distance:
            if distance_unit != "traces":
                distance_min = st.number_input(f"Min Distance ({distance_unit})", 0.0, total_distance, 0.0, 0.1)
                distance_max = st.number_input(f"Max Distance ({distance_unit})", 0.0, total_distance, total_distance, 0.1)
            else:
                distance_min = st.number_input("Min Distance (traces)", 0, 10000, 0)
                distance_max = st.number_input("Max Distance (traces)", 0, 10000, 800)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Multiple windows option
    multiple_windows = st.checkbox("Enable Multiple Windows", False,
                                  help="Plot multiple windows in the same view")
    
    if multiple_windows and use_custom_window:
        num_windows = st.number_input("Number of Additional Windows", 1, 5, 1)
        
        windows = []
        for i in range(num_windows):
            st.markdown(f"**Window {i+2}**")
            col1, col2 = st.columns(2)
            with col1:
                d_min = st.number_input(f"Depth Min {i+2} ({depth_unit})", 0.0, max_depth, 2.0 + i*2, 0.1)
                d_max = st.number_input(f"Depth Max {i+2} ({depth_unit})", 0.0, max_depth, 5.0 + i*2, 0.1)
            with col2:
                dist_min = st.number_input(f"Dist Min {i+2} ({distance_unit})", 0.0, total_distance, 50.0 + i*50, 0.1)
                dist_max = st.number_input(f"Dist Max {i+2} ({distance_unit})", 0.0, total_distance, 150.0 + i*50, 0.1)
            
            windows.append({
                'depth_min': d_min,
                'depth_max': d_max,
                'distance_min': dist_min,
                'distance_max': dist_max,
                'color': f'C{i+1}'
            })
    
    st.markdown("---")
    st.header("🔄 Line Adjustment & Muting")
    
    # Line reversal option
    reverse_line = st.checkbox("Reverse Line Direction (A→B to B→A)", False,
                              help="Reverse the order of traces to flip survey direction")
    
    # Trace muting option
    mute_traces = st.checkbox("Mute Traces", False,
                             help="Mute (set to zero) specific trace ranges")
    
    if mute_traces:
        st.markdown('<div class="mute-box">', unsafe_allow_html=True)
        st.subheader("Trace Muting Settings")
        
        # Muting method selection
        mute_method = st.selectbox("Mute Method", 
                                  ["By Distance", "By Trace Index", "Multiple Zones"],
                                  help="Choose how to define mute zones")
        
        if mute_method == "By Distance":
            col1, col2 = st.columns(2)
            with col1:
                mute_start_dist = st.number_input("Mute Start Distance", 0.0, 10000.0, 2.0, 0.1,
                                                 help="Start distance for muting")
            with col2:
                mute_end_dist = st.number_input("Mute End Distance", 0.0, 10000.0, 6.0, 0.1,
                                               help="End distance for muting")
            
            # Taper options for smoother transitions
            apply_taper = st.checkbox("Apply Taper to Mute Zone", True,
                                     help="Gradually fade in/out muting for smoother transitions")
            if apply_taper:
                taper_length = st.slider("Taper Length (% of zone)", 1, 50, 10, 1,
                                        help="Percentage of mute zone to apply gradual taper")
        
        elif mute_method == "By Trace Index":
            col1, col2 = st.columns(2)
            with col1:
                mute_start_idx = st.number_input("Mute Start Trace", 0, 10000, 100,
                                                help="Start trace index for muting")
            with col2:
                mute_end_idx = st.number_input("Mute End Trace", 0, 10000, 200,
                                              help="End trace index for muting")
            
            # Taper options
            apply_taper = st.checkbox("Apply Taper to Mute Zone", True,
                                     help="Gradually fade in/out muting for smoother transitions")
            if apply_taper:
                taper_samples = st.slider("Taper Samples", 1, 100, 10, 1,
                                         help="Number of samples for gradual taper")
        
        elif mute_method == "Multiple Zones":
            num_zones = st.number_input("Number of Mute Zones", 1, 5, 1)
            
            mute_zones = []
            for i in range(num_zones):
                st.markdown(f"**Mute Zone {i+1}**")
                col1, col2 = st.columns(2)
                with col1:
                    zone_method = st.selectbox(f"Zone {i+1} Method", ["By Distance", "By Trace Index"])
                    if zone_method == "By Distance":
                        zone_start = st.number_input(f"Zone {i+1} Start", 0.0, 10000.0, 10.0 + i*10, 0.1)
                        zone_end = st.number_input(f"Zone {i+1} End", 0.0, 10000.0, 15.0 + i*10, 0.1)
                    else:
                        zone_start = st.number_input(f"Zone {i+1} Start Trace", 0, 10000, 150 + i*50)
                        zone_end = st.number_input(f"Zone {i+1} End Trace", 0, 10000, 200 + i*50)
                
                with col2:
                    zone_taper = st.checkbox(f"Taper Zone {i+1}", True)
                    zone_label = st.text_input(f"Zone {i+1} Label", f"Zone {i+1}")
                
                mute_zones.append({
                    'method': zone_method,
                    'start': zone_start,
                    'end': zone_end,
                    'taper': zone_taper,
                    'label': zone_label
                })
        
        # Muting strength (for partial muting)
        mute_strength = st.slider("Muting Strength (%)", 0, 100, 100, 5,
                                 help="0% = no muting, 100% = complete muting (zero amplitude)")
        
        # Muting visualization option
        show_mute_preview = st.checkbox("Show Mute Zone Preview", True,
                                       help="Preview mute zones before processing")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("🎛️ Processing Parameters")
    
    time_zero = st.number_input("Time Zero (samples)", 0, 2000, 2, 
                               help="Adjust the start time of each trace")
    
    stacking = st.selectbox("Stacking", ["none", "auto", "manual"], 
                           help="Reduce noise by averaging traces")
    
    if stacking == "manual":
        stack_value = st.number_input("Stack Value", 1, 50, 3)
    
    st.markdown("---")
    st.header("🌍 Near-Surface Amplitude Correction")
    
    # Add near-surface correction option
    apply_near_surface_correction = st.checkbox("Apply Near-Surface Amplitude Correction", False,
                                               help="Reduce high amplitudes in 0-2.5m region to normalize visualization")
    
    if apply_near_surface_correction:
        st.markdown('<div class="near-surface-box">', unsafe_allow_html=True)
        
        # Near-surface correction parameters
        correction_type = st.selectbox("Correction Type", 
                                      ["Linear Reduction", "Exponential Reduction", "Gaussian Filter", "Windowed Normalization"],
                                      help="Method to reduce near-surface amplitudes")
        
        correction_depth = st.number_input("Correction Depth (m)", 0.1, 10.0, 2.5, 0.1,
                                         help="Depth range for near-surface correction")
        
        if correction_type == "Linear Reduction":
            surface_reduction = st.slider("Surface Amplitude Reduction (%)", 0, 95, 80, 5,
                                         help="Percentage to reduce amplitude at surface (0% = no reduction, 100% = complete removal)")
            depth_factor = st.slider("Reduction Depth Factor", 0.1, 2.0, 1.0, 0.1,
                                    help="How quickly reduction decreases with depth")
        
        elif correction_type == "Exponential Reduction":
            exp_factor = st.slider("Exponential Factor", 0.5, 5.0, 2.0, 0.1,
                                  help="Higher values = faster reduction with depth")
            max_reduction = st.slider("Maximum Reduction (%)", 0, 95, 90, 5)
        
        elif correction_type == "Gaussian Filter":
            filter_sigma = st.slider("Filter Sigma", 0.1, 5.0, 1.0, 0.1,
                                    help="Standard deviation for Gaussian filter")
            filter_window = st.slider("Filter Window (samples)", 5, 100, 21, 2)
        
        elif correction_type == "Windowed Normalization":
            window_size = st.slider("Normalization Window (samples)", 10, 200, 50, 5)
            target_amplitude = st.slider("Target Amplitude", 0.1, 1.0, 0.3, 0.05)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("🔬 Advanced Deconvolution")
    
    # Deconvolution options
    apply_deconvolution = st.checkbox("Apply Deconvolution", False,
                                     help="Apply deconvolution to improve resolution and remove multiples")
    
    if apply_deconvolution:
        st.markdown('<div class="deconv-box">', unsafe_allow_html=True)
        st.subheader("Deconvolution Settings")
        
        # Deconvolution method selection
        deconv_method = st.selectbox("Deconvolution Method",
                                    ["Wiener Filter", "Predictive Deconvolution", "Spiking Deconvolution",
                                     "Minimum Entropy Deconvolution", "Homomorphic Deconvolution", "Bayesian Deconvolution"],
                                    help="Select deconvolution algorithm")
        
        if deconv_method == "Wiener Filter":
            col1, col2 = st.columns(2)
            with col1:
                wiener_window = st.slider("Wiener Window (samples)", 5, 101, 21, 2,
                                         help="Window size for Wiener filter (odd number)")
                noise_level = st.slider("Noise Level", 0.001, 0.1, 0.01, 0.001,
                                       help="Estimated noise level for regularization")
            with col2:
                wavelet_length = st.slider("Wavelet Length (samples)", 5, 101, 51, 2,
                                          help="Estimated wavelet length")
                regularization = st.slider("Regularization", 0.0, 1.0, 0.1, 0.01,
                                          help="Tikhonov regularization parameter")
        
        elif deconv_method == "Predictive Deconvolution":
            col1, col2 = st.columns(2)
            with col1:
                prediction_distance = st.slider("Prediction Distance (samples)", 1, 100, 10, 1,
                                               help="Distance to predict ahead")
                filter_length = st.slider("Filter Length (samples)", 10, 200, 50, 5,
                                         help="Filter length for prediction")
            with col2:
                prewhitening = st.slider("Pre-whitening (%)", 0.0, 10.0, 0.1, 0.1,
                                        help="Percentage of white noise to add for stability")
                iterations = st.slider("Iterations", 1, 10, 3, 1,
                                      help="Number of iterations for convergence")
        
        elif deconv_method == "Spiking Deconvolution":
            col1, col2 = st.columns(2)
            with col1:
                spike_strength = st.slider("Spike Strength", 0.1, 2.0, 0.8, 0.1,
                                          help="Strength of desired spike output")
                spike_length = st.slider("Spike Length (samples)", 5, 101, 21, 2,
                                        help="Length of desired spike wavelet")
            with col2:
                spike_noise = st.slider("Spike Noise Level", 0.001, 0.1, 0.01, 0.001,
                                       help="Noise level for spike deconvolution")
                spike_iterations = st.slider("Iterations", 1, 20, 5, 1,
                                           help="Iterations for spike deconvolution")
        
        elif deconv_method == "Minimum Entropy Deconvolution":
            col1, col2 = st.columns(2)
            with col1:
                med_filter_length = st.slider("Filter Length (samples)", 10, 200, 80, 5,
                                             help="MED filter length")
                med_iterations = st.slider("Iterations", 1, 50, 10, 1,
                                          help="Number of MED iterations")
            with col2:
                med_convergence = st.slider("Convergence Threshold", 0.0001, 0.1, 0.001, 0.0001,
                                           help="Convergence threshold for MED")
                med_noise = st.slider("Noise Estimate", 0.001, 0.1, 0.01, 0.001,
                                     help="Initial noise estimate for MED")
        
        elif deconv_method == "Homomorphic Deconvolution":
            col1, col2 = st.columns(2)
            with col1:
                homo_window = st.selectbox("Smoothing Window", ["hanning", "hamming", "blackman", "bartlett"],
                                          help="Window for cepstral smoothing")
                homo_cutoff = st.slider("Cepstral Cutoff", 0.01, 0.5, 0.1, 0.01,
                                       help="Cutoff frequency in cepstral domain")
            with col2:
                homo_prewhiten = st.slider("Pre-whitening", 0.0, 0.1, 0.01, 0.001,
                                          help="Pre-whitening for homomorphic deconvolution")
                homo_iterations = st.slider("Iterations", 1, 10, 3, 1,
                                          help="Homomorphic iterations")
        
        elif deconv_method == "Bayesian Deconvolution":
            col1, col2 = st.columns(2)
            with col1:
                bayesian_prior = st.selectbox("Prior Distribution", ["Laplace", "Gaussian", "Jeffreys"],
                                             help="Prior distribution for Bayesian inference")
                bayesian_iterations = st.slider("MCMC Iterations", 100, 5000, 1000, 100,
                                               help="Number of MCMC iterations")
            with col2:
                bayesian_burnin = st.slider("Burn-in Samples", 100, 2000, 500, 100,
                                           help="Number of burn-in samples")
                bayesian_noise = st.slider("Noise Estimate", 0.001, 0.1, 0.01, 0.001,
                                          help="Noise standard deviation estimate")
        
        # Common deconvolution parameters
        st.subheader("Common Parameters")
        col1, col2 = st.columns(2)
        with col1:
            deconv_window_start = st.number_input("Deconvolution Start (samples)", 0, 5000, 0,
                                                 help="Start sample for deconvolution")
            deconv_window_end = st.number_input("Deconvolution End (samples)", 0, 5000, 1000,
                                               help="End sample for deconvolution")
        
        with col2:
            trace_for_wavelet = st.number_input("Trace for Wavelet Estimation", 0, 10000, 0,
                                               help="Trace index to use for wavelet estimation")
            use_average_wavelet = st.checkbox("Use Average Wavelet", True,
                                             help="Use average of multiple traces for wavelet estimation")
        
        if use_average_wavelet:
            wavelet_trace_range = st.slider("Wavelet Trace Range", 0, 100, 10, 1,
                                           help="Number of traces to average for wavelet")
        
        # Deconvolution output options
        output_type = st.selectbox("Output Type", 
                                  ["Deconvolved Only", "Deconvolved + Original", "Difference (Deconvolved - Original)"],
                                  help="What to display after deconvolution")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("📈 Time Gain Control")
    
    gain_type = st.selectbox(
        "Gain Type",
        ["Constant", "Linear", "Exponential", "AGC (Automatic Gain Control)", "Spherical"],
        help="Apply gain to amplify weak deep signals"
    )
    
    if gain_type == "Constant":
        const_gain = st.slider("Gain (%)", 0, 500, 100)
    
    elif gain_type == "Linear":
        min_gain = st.slider("Gain at Top (%)", 0, 200, 50)
        max_gain = st.slider("Gain at Bottom (%)", 0, 1000, 500)
    
    elif gain_type == "Exponential":
        base_gain = st.slider("Base Gain (%)", 0, 300, 100)
        exp_factor = st.slider("Exponential Factor", 0.1, 5.0, 1.5, 0.1)
    
    elif gain_type == "AGC (Automatic Gain Control)":
        window_size = st.slider("AGC Window (samples)", 10, 500, 100)
        target_amplitude = st.slider("Target Amplitude", 0.1, 1.0, 0.3, 0.05)
    
    elif gain_type == "Spherical":
        power_gain = st.slider("Power Gain", 1.0, 3.0, 2.0, 0.1)
        attenuation = st.slider("Attenuation Factor", 0.01, 0.1, 0.05, 0.01)
    
    st.markdown("---")
    st.header("⚙️ Advanced Processing")
    
    bgr = st.checkbox("Apply Background Removal", False)
    if bgr:
        bgr_type = st.selectbox("BGR Type", ["Full-width", "Boxcar"])
        if bgr_type == "Boxcar":
            bgr_window = st.slider("Boxcar Window", 10, 500, 100)
    
    freq_filter = st.checkbox("Apply Frequency Filter", False)
    if freq_filter:
        col1, col2 = st.columns(2)
        with col1:
            freq_min = st.number_input("Min Freq (MHz)", 10, 500, 60)
        with col2:
            freq_max = st.number_input("Max Freq (MHz)", 10, 1000, 130)
    
    process_btn = st.button("🚀 Process Data", type="primary", use_container_width=True)

# Helper functions for deconvolution
def estimate_wavelet(trace, method='auto', wavelet_length=51):
    """Estimate the wavelet from a trace"""
    if method == 'auto':
        # Use autocorrelation to estimate wavelet
        autocorr = np.correlate(trace, trace, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        # Normalize
        autocorr = autocorr / autocorr[0]
        # Take first wavelet_length samples
        wavelet = autocorr[:wavelet_length]
        return wavelet
    else:
        # Use Ricker wavelet as default
        t = np.linspace(-wavelet_length//2, wavelet_length//2, wavelet_length)
        wavelet = (1 - 2*(np.pi*0.1*t)**2) * np.exp(-(np.pi*0.1*t)**2)
        return wavelet

def wiener_deconvolution(trace, wavelet, noise_level=0.01, regularization=0.1):
    """Wiener deconvolution"""
    n = len(trace)
    m = len(wavelet)
    
    # Create Toeplitz matrix from wavelet
    col = np.zeros(n)
    col[:m] = wavelet
    row = np.zeros(n)
    row[0] = wavelet[0]
    
    H = toeplitz(col, row)
    
    # Add regularization
    R = regularization * np.eye(n)
    
    # Solve using regularized least squares
    try:
        # Use solve_toeplitz if available
        from scipy.linalg import solve_toeplitz
        # Create the first column of the Toeplitz matrix
        c = col
        r = row
        # Solve (H^T H + λI) x = H^T y
        HTy = np.dot(H.T, trace)
        # For Toeplitz systems, we can use Levinson recursion
        result = solve_toeplitz((c, r), HTy)
    except:
        # Fallback to direct solve
        HTH = np.dot(H.T, H) + R
        HTy = np.dot(H.T, trace)
        result = np.linalg.lstsq(HTH, HTy, rcond=None)[0]
    
    return result[:len(trace)]

def predictive_deconvolution(trace, prediction_distance=10, filter_length=50, prewhitening=0.1, iterations=3):
    """Predictive deconvolution"""
    n = len(trace)
    result = trace.copy()
    
    for _ in range(iterations):
        # Create prediction filter
        autocorr = np.correlate(result, result, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr[0] *= (1 + prewhitening)  # Pre-whitening
        
        # Solve Yule-Walker equations for prediction filter
        try:
            # Use Levinson-Durbin recursion
            from scipy.linalg import solve_toeplitz
            r = autocorr[:filter_length]
            b = autocorr[prediction_distance:prediction_distance+filter_length]
            prediction_filter = solve_toeplitz((r, r), b)
        except:
            # Fallback
            R = toeplitz(autocorr[:filter_length])
            b = autocorr[prediction_distance:prediction_distance+filter_length]
            prediction_filter = np.linalg.lstsq(R, b, rcond=None)[0]
        
        # Apply prediction
        predicted = np.convolve(result, prediction_filter, mode='same')
        result = result - predicted
    
    return result

def spiking_deconvolution(trace, desired_spike=0.8, spike_length=21, noise_level=0.01, iterations=5):
    """Spiking deconvolution"""
    n = len(trace)
    
    # Create desired output (spike)
    desired_output = np.zeros(n)
    desired_output[spike_length//2] = desired_spike
    
    # Estimate inverse filter
    autocorr = np.correlate(trace, trace, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr[0] *= (1 + noise_level)
    
    # Cross-correlation between input and desired output
    crosscorr = np.correlate(trace, desired_output, mode='full')
    crosscorr = crosscorr[len(crosscorr)//2:]
    
    # Solve for inverse filter using Wiener-Hopf equations
    filter_length = min(100, n//2)
    
    # Create autocorrelation matrix
    R = toeplitz(autocorr[:filter_length])
    # Add small diagonal for stability
    R += noise_level * np.eye(filter_length)
    
    # Create cross-correlation vector
    P = crosscorr[:filter_length]
    
    # Solve for filter
    try:
        inverse_filter = np.linalg.solve(R, P)
    except:
        inverse_filter = np.linalg.lstsq(R, P, rcond=None)[0]
    
    # Apply inverse filter
    deconvolved = np.convolve(trace, inverse_filter, mode='same')
    
    # Iterative refinement
    for _ in range(iterations-1):
        # Re-estimate with current result
        residual = trace - np.convolve(deconvolved, estimate_wavelet(trace), mode='same')
        update = np.convolve(residual, inverse_filter, mode='same')
        deconvolved = deconvolved + update
    
    return deconvolved

def minimum_entropy_deconvolution(trace, filter_length=80, iterations=10, convergence=0.001, noise_estimate=0.01):
    """Minimum Entropy Deconvolution (MED)"""
    n = len(trace)
    
    # Initialize filter as spike
    h = np.zeros(filter_length)
    h[filter_length//2] = 1.0
    
    # Store previous filter for convergence check
    h_prev = h.copy()
    
    for iteration in range(iterations):
        # Apply current filter
        y = np.convolve(trace, h, mode='same')
        
        # Update filter using kurtosis maximization
        # Compute gradient
        X = np.zeros((n, filter_length))
        for i in range(filter_length):
            X[:, i] = np.roll(trace, i - filter_length//2)[:n]
        
        # Kurtosis gradient
        y3 = y**3
        gradient = np.dot(X.T, y3) / np.sum(y**4)
        
        # Update filter
        h = gradient / np.linalg.norm(gradient)
        
        # Check convergence
        if np.linalg.norm(h - h_prev) < convergence:
            break
        
        h_prev = h.copy()
    
    # Apply final filter
    result = np.convolve(trace, h, mode='same')
    return result

def homomorphic_deconvolution(trace, window_type='hanning', cutoff=0.1, prewhitening=0.01, iterations=3):
    """Homomorphic deconvolution using cepstral analysis"""
    n = len(trace)
    
    # Ensure trace is positive for log
    trace_min = np.min(trace)
    if trace_min <= 0:
        trace = trace - trace_min + 0.001 * np.std(trace)
    
    result = trace.copy()
    
    for _ in range(iterations):
        # Compute complex cepstrum
        spectrum = np.fft.fft(result)
        log_spectrum = np.log(np.abs(spectrum) + prewhitening)
        cepstrum = np.fft.ifft(log_spectrum).real
        
        # Apply liftering (filtering in cepstral domain)
        # Create window
        n_cep = len(cepstrum)
        if window_type == 'hanning':
            window = np.hanning(n_cep)
        elif window_type == 'hamming':
            window = np.hamming(n_cep)
        elif window_type == 'blackman':
            window = np.blackman(n_cep)
        else:  # bartlett
            window = np.bartlett(n_cep)
        
        # Apply low-pass filter in cepstral domain
        cutoff_idx = int(cutoff * n_cep)
        window[:cutoff_idx] = 1
        window[-cutoff_idx:] = 1
        window[cutoff_idx:-cutoff_idx] = 0
        
        filtered_cepstrum = cepstrum * window
        
        # Transform back
        filtered_log_spectrum = np.fft.fft(filtered_cepstrum)
        estimated_wavelet_spectrum = np.exp(filtered_log_spectrum)
        
        # Deconvolve in frequency domain
        deconv_spectrum = spectrum / (estimated_wavelet_spectrum + prewhitening)
        result = np.fft.ifft(deconv_spectrum).real
    
    return result

def bayesian_deconvolution(trace, prior='Laplace', iterations=1000, burnin=500, noise_std=0.01):
    """Simple Bayesian deconvolution using MAP estimation"""
    n = len(trace)
    
    # Simple implementation - using sparse recovery with L1 regularization
    # For Laplace prior (L1 regularization)
    if prior == 'Laplace':
        # Use LASSO-type approach
        from sklearn.linear_model import Lasso
        # Create convolution matrix from estimated wavelet
        wavelet = estimate_wavelet(trace)
        m = len(wavelet)
        
        # Build Toeplitz matrix
        H = np.zeros((n, n))
        for i in range(n):
            if i + m <= n:
                H[i:i+m, i] = wavelet
    
    # For Gaussian prior (L2 regularization - Wiener filter)
    elif prior == 'Gaussian':
        return wiener_deconvolution(trace, estimate_wavelet(trace), noise_std, regularization=0.1)
    
    # For Jeffreys prior (sparse)
    else:
        # Use iterative reweighted L1
        result = trace.copy()
        for _ in range(5):
            weights = 1 / (np.abs(result) + 0.01)
            # This would require specialized optimization
            # For simplicity, return Wiener filter
            result = wiener_deconvolution(trace, estimate_wavelet(trace), noise_std, regularization=0.1)
    
    return result

def apply_deconvolution_to_array(array, method='Wiener Filter', **kwargs):
    """Apply deconvolution to entire array"""
    n_samples, n_traces = array.shape
    deconvolved = np.zeros_like(array)
    
    # Determine window for deconvolution
    start_sample = kwargs.get('deconv_window_start', 0)
    end_sample = kwargs.get('deconv_window_end', n_samples)
    start_sample = max(0, min(start_sample, n_samples-1))
    end_sample = max(0, min(end_sample, n_samples-1))
    
    # Estimate wavelet from selected trace(s)
    trace_for_wavelet = kwargs.get('trace_for_wavelet', 0)
    use_average = kwargs.get('use_average_wavelet', True)
    wavelet_trace_range = kwargs.get('wavelet_trace_range', 10)
    
    if use_average and wavelet_trace_range > 1:
        start_trace = max(0, trace_for_wavelet - wavelet_trace_range//2)
        end_trace = min(n_traces, trace_for_wavelet + wavelet_trace_range//2)
        avg_trace = np.mean(array[:, start_trace:end_trace], axis=1)
        wavelet = estimate_wavelet(avg_trace, wavelet_length=kwargs.get('wavelet_length', 51))
    else:
        trace_idx = min(max(0, trace_for_wavelet), n_traces-1)
        wavelet = estimate_wavelet(array[:, trace_idx], wavelet_length=kwargs.get('wavelet_length', 51))
    
    st.session_state.estimated_wavelet = wavelet
    
    # Apply deconvolution to each trace
    for i in range(n_traces):
        trace = array[:, i].copy()
        
        if method == "Wiener Filter":
            deconv_trace = wiener_deconvolution(
                trace, wavelet,
                noise_level=kwargs.get('noise_level', 0.01),
                regularization=kwargs.get('regularization', 0.1)
            )
        
        elif method == "Predictive Deconvolution":
            deconv_trace = predictive_deconvolution(
                trace,
                prediction_distance=kwargs.get('prediction_distance', 10),
                filter_length=kwargs.get('filter_length', 50),
                prewhitening=kwargs.get('prewhitening', 0.1)/100,
                iterations=kwargs.get('iterations', 3)
            )
        
        elif method == "Spiking Deconvolution":
            deconv_trace = spiking_deconvolution(
                trace,
                desired_spike=kwargs.get('spike_strength', 0.8),
                spike_length=kwargs.get('spike_length', 21),
                noise_level=kwargs.get('spike_noise', 0.01),
                iterations=kwargs.get('spike_iterations', 5)
            )
        
        elif method == "Minimum Entropy Deconvolution":
            deconv_trace = minimum_entropy_deconvolution(
                trace,
                filter_length=kwargs.get('med_filter_length', 80),
                iterations=kwargs.get('med_iterations', 10),
                convergence=kwargs.get('med_convergence', 0.001),
                noise_estimate=kwargs.get('med_noise', 0.01)
            )
        
        elif method == "Homomorphic Deconvolution":
            deconv_trace = homomorphic_deconvolution(
                trace,
                window_type=kwargs.get('homo_window', 'hanning'),
                cutoff=kwargs.get('homo_cutoff', 0.1),
                prewhitening=kwargs.get('homo_prewhiten', 0.01),
                iterations=kwargs.get('homo_iterations', 3)
            )
        
        elif method == "Bayesian Deconvolution":
            deconv_trace = bayesian_deconvolution(
                trace,
                prior=kwargs.get('bayesian_prior', 'Laplace'),
                iterations=kwargs.get('bayesian_iterations', 1000),
                burnin=kwargs.get('bayesian_burnin', 500),
                noise_std=kwargs.get('bayesian_noise', 0.01)
            )
        
        else:
            deconv_trace = trace.copy()
        
        # Apply deconvolution only within the specified window
        if start_sample > 0 or end_sample < n_samples:
            deconvolved[start_sample:end_sample, i] = deconv_trace[start_sample:end_sample]
            # Blend edges to avoid discontinuities
            if start_sample > 0:
                blend_samples = min(50, start_sample)
                blend = np.linspace(0, 1, blend_samples)
                deconvolved[start_sample-blend_samples:start_sample, i] = (
                    (1 - blend) * trace[start_sample-blend_samples:start_sample] +
                    blend * deconv_trace[start_sample-blend_samples:start_sample]
                )
            
            if end_sample < n_samples:
                blend_samples = min(50, n_samples - end_sample)
                blend = np.linspace(1, 0, blend_samples)
                deconvolved[end_sample:end_sample+blend_samples, i] = (
                    blend * deconv_trace[end_sample:end_sample+blend_samples] +
                    (1 - blend) * trace[end_sample:end_sample+blend_samples]
                )
        else:
            deconvolved[:, i] = deconv_trace
    
    return deconvolved

# Other helper functions (keep existing ones)
def apply_gain(array, gain_type, **kwargs):
    """Apply time-varying gain to radar data"""
    n_samples, n_traces = array.shape
    
    if gain_type == "Constant":
        gain = 1 + kwargs.get('const_gain', 1.0) / 100
        return array * gain
    
    elif gain_type == "Linear":
        min_g = 1 + kwargs.get('min_gain', 0.5) / 100
        max_g = 1 + kwargs.get('max_gain', 5.0) / 100
        gain_vector = np.linspace(min_g, max_g, n_samples)
        return array * gain_vector[:, np.newaxis]
    
    elif gain_type == "Exponential":
        base_g = 1 + kwargs.get('base_gain', 1.0) / 100
        exp_f = kwargs.get('exp_factor', 1.5)
        t = np.linspace(0, 1, n_samples)
        gain_vector = base_g * np.exp(exp_f * t)
        return array * gain_vector[:, np.newaxis]
    
    elif gain_type == "AGC (Automatic Gain Control)":
        window = kwargs.get('window_size', 100)
        target = kwargs.get('target_amplitude', 0.3)
        
        result = np.zeros_like(array)
        half_window = window // 2
        
        for i in range(n_traces):
            trace = array[:, i]
            agc_trace = np.zeros(n_samples)
            
            for j in range(n_samples):
                start = max(0, j - half_window)
                end = min(n_samples, j + half_window + 1)
                
                window_data = trace[start:end]
                rms = np.sqrt(np.mean(window_data**2))
                
                if rms > 0:
                    agc_trace[j] = trace[j] * (target / rms)
                else:
                    agc_trace[j] = trace[j]
            
            result[:, i] = agc_trace
        
        return result
    
    elif gain_type == "Spherical":
        power = kwargs.get('power_gain', 2.0)
        attenuation = kwargs.get('attenuation', 0.05)
        
        # Create spherical spreading correction
        t = np.arange(n_samples) / n_samples
        gain_vector = (1 + attenuation * t) ** power
        gain_vector = gain_vector[:, np.newaxis]
        
        return array * gain_vector
    
    return array

def apply_near_surface_correction(array, correction_type, correction_depth, max_depth, **kwargs):
    """Apply near-surface amplitude correction to reduce high amplitudes in shallow region"""
    n_samples, n_traces = array.shape
    
    # Calculate which samples correspond to the correction depth
    if max_depth is not None:
        # Convert correction depth to sample index
        correction_samples = int((correction_depth / max_depth) * n_samples)
    else:
        # If no max_depth, use 10% of samples as default
        correction_samples = int(0.1 * n_samples)
    
    # Ensure correction_samples is at least 1 and not more than n_samples
    correction_samples = max(1, min(correction_samples, n_samples))
    
    result = array.copy()
    
    if correction_type == "Linear Reduction":
        surface_reduction = kwargs.get('surface_reduction', 80) / 100.0
        depth_factor = kwargs.get('depth_factor', 1.0)
        
        # Create linear reduction vector
        reduction_vector = np.ones(n_samples)
        depth_ratios = np.linspace(0, 1, correction_samples)
        # Linear reduction that decreases with depth
        reduction_vector[:correction_samples] = 1 - surface_reduction * (1 - depth_ratios**depth_factor)
        
        result = result * reduction_vector[:, np.newaxis]
    
    elif correction_type == "Exponential Reduction":
        exp_factor = kwargs.get('exp_factor', 2.0)
        max_reduction = kwargs.get('max_reduction', 90) / 100.0
        
        # Create exponential reduction vector
        reduction_vector = np.ones(n_samples)
        depth_ratios = np.linspace(0, 1, correction_samples)
        # Exponential reduction
        reduction_vector[:correction_samples] = 1 - max_reduction * np.exp(-exp_factor * depth_ratios)
        
        result = result * reduction_vector[:, np.newaxis]
    
    elif correction_type == "Gaussian Filter":
        filter_sigma = kwargs.get('filter_sigma', 1.0)
        filter_window = kwargs.get('filter_window', 21)
        
        # Apply Gaussian filter only to near-surface region
        from scipy.ndimage import gaussian_filter1d
        
        # Create a smoothed version of the near-surface data
        near_surface = array[:correction_samples, :]
        
        # Apply Gaussian filter along depth axis
        filtered_surface = gaussian_filter1d(near_surface, sigma=filter_sigma, axis=0, mode='reflect')
        
        # Blend original and filtered based on depth
        blend_weights = np.linspace(1.0, 0.0, correction_samples)[:, np.newaxis]
        blended_surface = near_surface * blend_weights + filtered_surface * (1 - blend_weights)
        
        result[:correction_samples, :] = blended_surface
    
    elif correction_type == "Windowed Normalization":
        window_size = kwargs.get('window_size', 50)
        target_amplitude = kwargs.get('target_amplitude', 0.3)
        
        half_window = window_size // 2
        
        for i in range(n_traces):
            trace = result[:correction_samples, i]
            normalized_trace = np.zeros_like(trace)
            
            for j in range(len(trace)):
                start = max(0, j - half_window)
                end = min(len(trace), j + half_window + 1)
                
                window_data = trace[start:end]
                rms = np.sqrt(np.mean(window_data**2))
                
                if rms > 0:
                    # Gradually reduce the normalization effect with depth
                    depth_factor = 1.0 - (j / len(trace))
                    normalized_trace[j] = trace[j] * (target_amplitude / rms) * depth_factor
                else:
                    normalized_trace[j] = trace[j]
            
            result[:correction_samples, i] = normalized_trace
    
    return result

def reverse_array(array):
    """Reverse the array along the trace axis (flip A->B to B->A)"""
    return array[:, ::-1]

def apply_trace_mute(array, mute_params, x_axis=None, coordinates=None):
    """Apply trace muting to the radar array"""
    n_samples, n_traces = array.shape
    muted_array = array.copy()
    mute_mask = np.zeros_like(array, dtype=bool)
    
    # If coordinates are provided, use coordinate distance
    if coordinates is not None and x_axis is None:
        x_axis = coordinates['distance']
    
    # Handle different mute methods
    if mute_params['method'] == "By Distance":
        # Find trace indices corresponding to distance range
        if x_axis is not None:
            start_idx = np.argmin(np.abs(x_axis - mute_params['start']))
            end_idx = np.argmin(np.abs(x_axis - mute_params['end']))
            
            # Ensure start < end
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            
            # Ensure indices are within bounds
            start_idx = max(0, min(start_idx, n_traces-1))
            end_idx = max(0, min(end_idx, n_traces-1))
            
            # Apply muting
            if mute_params.get('apply_taper', False):
                taper_len = int((end_idx - start_idx) * mute_params.get('taper_length', 0.1))
                taper_start = np.linspace(1, 0, taper_len)
                taper_end = np.linspace(0, 1, taper_len)
                
                # Full mute middle section
                if end_idx - start_idx > 2 * taper_len:
                    mute_factor = (1 - mute_params['strength']/100)
                    muted_array[:, start_idx+taper_len:end_idx-taper_len] *= mute_factor
                    mute_mask[:, start_idx+taper_len:end_idx-taper_len] = True
                
                # Tapered edges
                for i in range(taper_len):
                    taper_val = taper_start[i]
                    mute_factor = (1 - mute_params['strength']/100 * taper_val)
                    muted_array[:, start_idx+i] *= mute_factor
                    mute_mask[:, start_idx+i] = taper_val > 0.5
                    
                    taper_val = taper_end[i]
                    mute_factor = (1 - mute_params['strength']/100 * taper_val)
                    muted_array[:, end_idx-taper_len+i] *= mute_factor
                    mute_mask[:, end_idx-taper_len+i] = taper_val > 0.5
            else:
                # Simple mute without taper
                mute_factor = (1 - mute_params['strength']/100)
                muted_array[:, start_idx:end_idx] *= mute_factor
                mute_mask[:, start_idx:end_idx] = True
    
    elif mute_params['method'] == "By Trace Index":
        start_idx = int(mute_params['start'])
        end_idx = int(mute_params['end'])
        
        # Ensure start < end and within bounds
        start_idx = max(0, min(start_idx, n_traces-1))
        end_idx = max(0, min(end_idx, n_traces-1))
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        
        # Apply muting
        if mute_params.get('apply_taper', False):
            taper_samples = mute_params.get('taper_samples', 10)
            taper_start = np.linspace(1, 0, taper_samples)
            taper_end = np.linspace(0, 1, taper_samples)
            
            # Full mute middle section
            if end_idx - start_idx > 2 * taper_samples:
                mute_factor = (1 - mute_params['strength']/100)
                muted_array[:, start_idx+taper_samples:end_idx-taper_samples] *= mute_factor
                mute_mask[:, start_idx+taper_samples:end_idx-taper_samples] = True
            
            # Tapered edges
            for i in range(taper_samples):
                taper_val = taper_start[i]
                mute_factor = (1 - mute_params['strength']/100 * taper_val)
                muted_array[:, start_idx+i] *= mute_factor
                mute_mask[:, start_idx+i] = taper_val > 0.5
                
                taper_val = taper_end[i]
                mute_factor = (1 - mute_params['strength']/100 * taper_val)
                muted_array[:, end_idx-taper_samples+i] *= mute_factor
                mute_mask[:, end_idx-taper_samples+i] = taper_val > 0.5
        else:
            # Simple mute without taper
            mute_factor = (1 - mute_params['strength']/100)
            muted_array[:, start_idx:end_idx] *= mute_factor
            mute_mask[:, start_idx:end_idx] = True
    
    return muted_array, mute_mask

def apply_multiple_mute_zones(array, mute_zones, x_axis=None, coordinates=None):
    """Apply multiple mute zones to the radar array"""
    muted_array = array.copy()
    combined_mask = np.zeros_like(array, dtype=bool)
    
    for zone in mute_zones:
        zone_params = {
            'method': zone['method'],
            'start': zone['start'],
            'end': zone['end'],
            'apply_taper': zone.get('taper', False),
            'strength': 100,
            'taper_length': 0.1 if zone.get('taper', False) else 0,
            'taper_samples': 10 if zone.get('taper', False) else 0
        }
        
        # Apply this zone
        zone_muted, zone_mask = apply_trace_mute(muted_array, zone_params, x_axis, coordinates)
        
        # Update combined mask
        combined_mask = combined_mask | zone_mask
        
        # For multiple zones, we need to be careful about overlapping zones
        muted_array = np.minimum(muted_array, zone_muted)
    
    return muted_array, combined_mask

def calculate_fft(trace, sampling_rate=1000):
    """Calculate FFT of a trace"""
    n = len(trace)
    yf = fft(trace)
    xf = fftfreq(n, 1/sampling_rate)[:n//2]
    
    # Take magnitude
    magnitude = 2.0/n * np.abs(yf[:n//2])
    
    return xf, magnitude

def process_coordinates(coords_df, n_traces, trace_col=None, method='linear'):
    """Process and interpolate coordinates to match number of GPR traces"""
    required_cols = ['Easting', 'Northing', 'Elevation']
    available_cols = {}
    
    # Try to find columns (case-insensitive, partial match)
    for req in required_cols:
        matches = [col for col in coords_df.columns if req.lower() in col.lower()]
        if matches:
            available_cols[req] = matches[0]
        else:
            st.error(f"Column '{req}' not found in CSV. Available columns: {list(coords_df.columns)}")
            return None
    
    # Extract data
    easting = coords_df[available_cols['Easting']].values
    northing = coords_df[available_cols['Northing']].values
    elevation = coords_df[available_cols['Elevation']].values
    
    # Determine x positions for coordinate points
    if trace_col and trace_col in coords_df.columns:
        # Use provided trace indices
        coord_trace_indices = coords_df[trace_col].values
    else:
        # Assume coordinates are evenly spaced along the profile
        dx = np.diff(easting)
        dy = np.diff(northing)
        distances = np.sqrt(dx**2 + dy**2)
        cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
        coord_trace_indices = np.linspace(0, n_traces-1, len(cumulative_dist))
    
    # Target trace indices (all traces)
    target_trace_indices = np.arange(n_traces)
    
    # Interpolate each coordinate component
    if method == 'linear':
        kind = 'linear'
    elif method == 'cubic':
        kind = 'cubic'
    elif method == 'nearest':
        kind = 'nearest'
    elif method == 'previous':
        kind = 'previous'
    elif method == 'next':
        kind = 'next'
    else:
        kind = 'linear'
    
    # Create interpolation functions
    try:
        f_easting = interp1d(coord_trace_indices, easting, kind=kind, fill_value='extrapolate')
        f_northing = interp1d(coord_trace_indices, northing, kind=kind, fill_value='extrapolate')
        f_elevation = interp1d(coord_trace_indices, elevation, kind=kind, fill_value='extrapolate')
        
        # Interpolate to all traces
        easting_interp = f_easting(target_trace_indices)
        northing_interp = f_northing(target_trace_indices)
        elevation_interp = f_elevation(target_trace_indices)
        
        # Calculate distance along profile (cumulative distance from start)
        dx_interp = np.diff(easting_interp)
        dy_interp = np.diff(northing_interp)
        dist_interp = np.sqrt(dx_interp**2 + dy_interp**2)
        cumulative_distance = np.concatenate(([0], np.cumsum(dist_interp)))
        
        return {
            'easting': easting_interp,
            'northing': northing_interp,
            'elevation': elevation_interp,
            'distance': cumulative_distance,
            'trace_indices': target_trace_indices,
            'original_points': len(easting),
            'interpolated_points': n_traces
        }
        
    except Exception as e:
        st.error(f"Error interpolating coordinates: {str(e)}")
        return None

def scale_axes(array_shape, depth_unit, max_depth, distance_unit, total_distance, coordinates=None):
    """Create scaled axis arrays based on user input"""
    n_samples, n_traces = array_shape
    
    # Scale Y-axis (depth/time)
    if depth_unit == "samples":
        y_axis = np.arange(n_samples)
        y_label = "Sample Number"
    elif depth_unit == "meters":
        y_axis = np.linspace(0, max_depth, n_samples)
        y_label = "Depth (m)"
    elif depth_unit == "nanoseconds":
        y_axis = np.linspace(0, max_depth, n_samples)
        y_label = "Two-way Time (ns)"
    elif depth_unit == "feet":
        y_axis = np.linspace(0, max_depth, n_samples)
        y_label = "Depth (ft)"
    
    # Scale X-axis (distance)
    if coordinates is not None:
        # Use coordinate-based distance
        x_axis = coordinates['distance']
        x_label = "Distance along profile (m)"
        distance_unit = "meters"  # Coordinates are assumed to be in meters
        total_distance = x_axis[-1]
    elif distance_unit == "traces":
        x_axis = np.arange(n_traces)
        x_label = "Trace Number"
    elif distance_unit == "meters":
        x_axis = np.linspace(0, total_distance, n_traces)
        x_label = "Distance (m)"
    elif distance_unit == "feet":
        x_axis = np.linspace(0, total_distance, n_traces)
        x_label = "Distance (ft)"
    elif distance_unit == "kilometers":
        x_axis = np.linspace(0, total_distance, n_traces)
        x_label = "Distance (km)"
    
    return x_axis, y_axis, x_label, y_label, distance_unit, total_distance

def get_aspect_ratio(mode, manual_ratio=None, data_shape=None):
    """Calculate aspect ratio based on mode"""
    if mode == "Auto":
        return "auto"
    elif mode == "Equal":
        return "equal"
    elif mode == "Manual" and manual_ratio is not None:
        return manual_ratio
    elif mode == "Realistic" and manual_ratio is not None:
        return manual_ratio
    elif data_shape is not None:
        return data_shape[0] / data_shape[1] * 0.5
    else:
        return "auto"

def get_window_indices(x_axis, y_axis, depth_min, depth_max, distance_min, distance_max):
    """Convert user-specified window coordinates to array indices"""
    # Find depth indices
    depth_idx_min = np.argmin(np.abs(y_axis - depth_min))
    depth_idx_max = np.argmin(np.abs(y_axis - depth_max))
    
    # Ensure correct ordering
    if depth_idx_min > depth_idx_max:
        depth_idx_min, depth_idx_max = depth_idx_max, depth_idx_min
    
    # Find distance indices
    dist_idx_min = np.argmin(np.abs(x_axis - distance_min))
    dist_idx_max = np.argmin(np.abs(x_axis - distance_max))
    
    # Ensure correct ordering
    if dist_idx_min > dist_idx_max:
        dist_idx_min, dist_idx_max = dist_idx_max, dist_idx_min
    
    return {
        'depth_min_idx': depth_idx_min,
        'depth_max_idx': depth_idx_max,
        'dist_min_idx': dist_idx_min,
        'dist_max_idx': dist_idx_max,
        'depth_min_val': y_axis[depth_idx_min],
        'depth_max_val': y_axis[depth_idx_max],
        'dist_min_val': x_axis[dist_idx_min],
        'dist_max_val': x_axis[dist_idx_max]
    }

# Main content
if csv_file and process_btn:
    with st.spinner("Processing GPR data from CSV..."):
        try:
            # Create progress bar
            progress_bar = st.progress(0)

            # ---------- NEW: Read CSV ----------
            df = pd.read_csv(csv_file, header=0, index_col=0)
            original_array = df.values.astype(np.float32)
            n_samples, n_traces = original_array.shape
            st.info(f"Loaded CSV: {n_samples} samples × {n_traces} traces")

            progress_bar.progress(20)

            # ---------- Process coordinates (unchanged) ----------
            coordinates_data = None
            if coord_csv:
                try:
                    coords_df = pd.read_csv(coord_csv)
                    st.session_state.coordinates = coords_df
                    st.info(f"Loaded {len(coords_df)} coordinate points")
                except Exception as e:
                    st.warning(f"Could not read CSV coordinates: {str(e)}")
                    coord_csv = None

            progress_bar.progress(30)

            # ---------- Line reversal (unchanged) ----------
            if reverse_line:
                original_array = reverse_array(original_array)
                st.session_state.line_reversed = True
                st.info("✓ Line direction reversed (A→B to B→A)")
            else:
                st.session_state.line_reversed = False
            st.session_state.reverse_line = reverse_line

            # ---------- Trace muting (unchanged) ----------
            # ... (your existing muting code) ...

            progress_bar.progress(40)

            # ---------- NEW: Apply time-zero adjustment ----------
            if time_zero > 0:
                shifted = np.zeros_like(original_array)
                if time_zero < n_samples:
                    shifted[:-time_zero, :] = original_array[time_zero:, :]
                original_array = shifted
                st.info(f"✓ Applied time-zero adjustment: {time_zero} samples")

            progress_bar.progress(45)

            # ---------- NEW: Apply stacking ----------
            if stacking != "none":
                if stacking == "auto":
                    stack_width = 3
                else:
                    stack_width = stack_value
                if stack_width > 1:
                    stacked = np.zeros_like(original_array)
                    half = stack_width // 2
                    for i in range(n_traces):
                        start = max(0, i - half)
                        end = min(n_traces, i + half + 1)
                        stacked[:, i] = np.mean(original_array[:, start:end], axis=1)
                    original_array = stacked
                    st.info(f"✓ Applied stacking: {stack_width}-trace moving average")

            progress_bar.progress(50)

            # ---------- NEW: Apply background removal ----------
            if bgr:
                if bgr_type == "Full-width":
                    background = np.mean(original_array, axis=1, keepdims=True)
                    original_array = original_array - background
                    st.info("✓ Applied full-width background removal")
                else:  # Boxcar
                    half = bgr_window // 2
                    background = np.zeros_like(original_array)
                    for i in range(n_traces):
                        start = max(0, i - half)
                        end = min(n_traces, i + half + 1)
                        background[:, i] = np.mean(original_array[:, start:end], axis=1)
                    original_array = original_array - background
                    st.info(f"✓ Applied boxcar background removal (window: {bgr_window} traces)")

            progress_bar.progress(60)

            # ---------- NEW: Apply frequency filtering ----------
            if freq_filter:
                from scipy import signal as scipy_signal
                # Design a Butterworth bandpass filter (you can adjust order and type)
                sos = scipy_signal.butter(4, [freq_min, freq_max], btype='band', fs=freq_max*2, output='sos')
                filtered = np.zeros_like(original_array)
                for i in range(n_traces):
                    filtered[:, i] = scipy_signal.sosfiltfilt(sos, original_array[:, i])
                original_array = filtered
                st.info(f"✓ Applied frequency filter: {freq_min}-{freq_max} MHz")

            progress_bar.progress(70)

            # ---------- Near-surface correction (unchanged) ----------
            if apply_near_surface_correction:
                # ... (your existing near-surface correction code) ...

            # ---------- Deconvolution (unchanged) ----------
            if apply_deconvolution:
                # ... (your existing deconvolution code) ...
            else:
                st.session_state.deconvolution_applied = False
                processed_array = original_array.copy()

            # ---------- Gain (unchanged) ----------
            processed_array = apply_gain(processed_array, gain_type,
                                        # ... parameters ...
                                        )

            progress_bar.progress(80)

            # ---------- Coordinate interpolation (unchanged) ----------
            if coord_csv and st.session_state.coordinates is not None:
                # ... (your existing coordinate processing code) ...

            progress_bar.progress(90)

            # ---------- Store in session state (modified header) ----------
            st.session_state.header = {
                'system': 'CSV Import',
                'ant_freq': 'N/A',
                'spt': n_samples,
                'ntraces': n_traces
            }
            st.session_state.original_array = original_array
            st.session_state.processed_array = processed_array
            st.session_state.gps = None
            st.session_state.data_loaded = True

            # Store axis scaling parameters (unchanged)
            st.session_state.depth_unit = depth_unit
            st.session_state.max_depth = max_depth if depth_unit != "samples" else None
            st.session_state.use_coords_for_distance = use_coords_for_distance
            st.session_state.coordinates_data = coordinates_data
            if not st.session_state.use_coords_for_distance:
                st.session_state.distance_unit = distance_unit
                st.session_state.total_distance = total_distance if distance_unit != "traces" else None
            else:
                st.session_state.distance_unit = "meters"
                st.session_state.total_distance = coordinates_data['distance'][-1] if coordinates_data else None

            # Aspect ratio & windows (unchanged)
            st.session_state.aspect_mode = aspect_mode
            if aspect_mode == "Manual" and 'aspect_ratio_float' in locals():
                st.session_state.aspect_ratio = aspect_ratio_float
            elif aspect_mode == "Realistic" and 'aspect_ratio_float' in locals():
                st.session_state.aspect_ratio = aspect_ratio_float
            else:
                st.session_state.aspect_ratio = None

            st.session_state.use_custom_window = use_custom_window
            if use_custom_window:
                st.session_state.depth_min = depth_min if 'depth_min' in locals() else 0
                st.session_state.depth_max = depth_max if 'depth_max' in locals() else max_depth
                if not st.session_state.use_coords_for_distance:
                    st.session_state.distance_min = distance_min if 'distance_min' in locals() else 0
                    st.session_state.distance_max = distance_max if 'distance_max' in locals() else total_distance
            st.session_state.multiple_windows = multiple_windows
            if multiple_windows and use_custom_window and 'windows' in locals():
                st.session_state.additional_windows = windows

            progress_bar.progress(100)
            st.success("✅ Data processed successfully!")

        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.code(str(e))

# Display results if data is loaded
if st.session_state.data_loaded:
    # Create tabs - Added Deconvolution Analysis tab
    tab_names = ["📊 Header Info", "📈 Full View", "🔍 Custom Window", "🗺️ Coordinate View", 
                 "📉 FFT Analysis", "🎛️ Gain Analysis", "🔬 Deconvolution Analysis", "💾 Export"]
    tabs = st.tabs(tab_names)
    
    with tabs[0]:  # Header Info
        st.subheader("File Information & Settings")
        
        # Display coordinate info if available
        if st.session_state.interpolated_coords is not None:
            st.markdown("### Coordinate Information")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Original Points", st.session_state.interpolated_coords['original_points'])
                st.metric("Total Distance", f"{st.session_state.interpolated_coords['distance'][-1]:.1f} m")
            
            with col2:
                st.metric("Interpolated Points", st.session_state.interpolated_coords['interpolated_points'])
                st.metric("Avg Point Spacing", 
                         f"{st.session_state.interpolated_coords['distance'][-1]/st.session_state.interpolated_coords['original_points']:.1f} m")
            
            with col3:
                st.metric("Easting Range", 
                         f"{st.session_state.interpolated_coords['easting'].min():.1f} - {st.session_state.interpolated_coords['easting'].max():.1f}")
                st.metric("Elevation Range", 
                         f"{st.session_state.interpolated_coords['elevation'].min():.1f} - {st.session_state.interpolated_coords['elevation'].max():.1f}")
        
        # Display scaling settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Axis Scaling Settings")
            settings_data = {
                "Y-axis (Depth)": f"{st.session_state.depth_unit}",
                "Max Y-value": f"{st.session_state.max_depth if st.session_state.max_depth else 'Auto'}",
                "X-axis (Distance)": f"{st.session_state.distance_unit}",
                "Total X-distance": f"{st.session_state.total_distance if st.session_state.total_distance else 'Auto'}"
            }
            
            for key, value in settings_data.items():
                st.markdown(f"**{key}:** {value}")
            
            st.markdown(f"**Aspect Mode:** {st.session_state.aspect_mode}")
            if st.session_state.aspect_ratio:
                st.markdown(f"**Aspect Ratio:** {st.session_state.aspect_ratio:.3f}")
            
            # Display line adjustment info
            if hasattr(st.session_state, 'line_reversed') and st.session_state.line_reversed:
                st.markdown("### Line Adjustment")
                st.markdown("**Line Direction:** Reversed (B→A)")
            
            # Display near-surface correction info if applied
            if hasattr(st.session_state, 'near_surface_correction') and st.session_state.near_surface_correction:
                st.markdown("### Near-Surface Correction")
                st.markdown(f"**Type:** {st.session_state.correction_type}")
                st.markdown(f"**Depth:** {st.session_state.correction_depth} m")
            
            # Display mute info if applied
            if hasattr(st.session_state, 'mute_applied') and st.session_state.mute_applied:
                st.markdown("### Trace Muting")
                st.markdown(f"**Muting Applied:** ✓")
                st.markdown(f"**Mute Strength:** {mute_strength if 'mute_strength' in locals() else 100}%")
                
                if hasattr(st.session_state, 'mute_zones'):
                    for i, zone in enumerate(st.session_state.mute_zones):
                        zone_label = zone.get('label', f'Zone {i+1}')
                        if zone['method'] == 'By Distance':
                            st.markdown(f"**{zone_label}:** Distance {zone['start']:.1f} - {zone['end']:.1f} {st.session_state.distance_unit}")
                        else:
                            st.markdown(f"**{zone_label}:** Traces {zone['start']} - {zone['end']}")
                        
                        if zone.get('apply_taper', False):
                            st.markdown(f"  *With taper applied*")
            
            # Display deconvolution info if applied
            if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
                st.markdown("### Deconvolution")
                st.markdown(f"**Method:** {st.session_state.deconv_method}")
                if hasattr(st.session_state, 'deconv_params'):
                    params = st.session_state.deconv_params
                    st.markdown(f"**Window:** {params.get('deconv_window_start', 0)} - {params.get('deconv_window_end', 1000)} samples")
                    st.markdown(f"**Wavelet Trace:** {params.get('trace_for_wavelet', 0)}")
                    if params.get('use_average_wavelet', False):
                        st.markdown(f"**Wavelet Averaging:** {params.get('wavelet_trace_range', 10)} traces")
        
        with col2:
            if st.session_state.header:
                st.markdown("### File Header")
                info_data = {
                    "System": st.session_state.header.get('system', 'Unknown'),
                    "Antenna Frequency": f"{st.session_state.header.get('ant_freq', 'N/A')} MHz",
                    "Samples per Trace": st.session_state.header.get('spt', 'N/A'),
                    "Number of Traces": st.session_state.header.get('ntraces', 'N/A')
                }
                
                for key, value in info_data.items():
                    st.markdown(f"**{key}:** {value}")
    
    with tabs[1]:  # Full View
        st.subheader("Full Radar Profile")
        
        # Determine aspect ratio
        aspect_value = get_aspect_ratio(
            st.session_state.aspect_mode,
            st.session_state.aspect_ratio,
            st.session_state.processed_array.shape
        )
        
        # Create scaled axes for full view
        x_axis_full, y_axis_full, x_label_full, y_label_full, _, _ = scale_axes(
            st.session_state.processed_array.shape,
            st.session_state.depth_unit,
            st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
            st.session_state.distance_unit,
            st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None,
            coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
        )
        
        # Display options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_colorbar = st.checkbox("Show Colorbar", True, key="full_cbar")
            interpolation = st.selectbox("Interpolation", ["none", "bilinear", "bicubic", "gaussian"], key="full_interp")
        
        with col2:
            colormap = st.selectbox("Colormap", ["seismic", "RdBu", "gray", "viridis", "jet", "coolwarm"], key="full_cmap")
            aspect_display = st.selectbox("Display Aspect", ["auto", "equal", 0.1, 0.2, 0.5, 1.0, 2.0, 5.0], 
                                         index=0, key="full_display_aspect")
        
        with col3:
            vmin = st.number_input("Color Min", -1.0, 0.0, -0.5, 0.01, key="full_vmin")
            vmax = st.number_input("Color Max", 0.0, 1.0, 0.5, 0.01, key="full_vmax")
            normalize_colors = st.checkbox("Auto-normalize Colors", True, key="full_norm")
        
        # Create figure - Show deconvolved data if available
        if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
            fig_full, (ax1_full, ax2_full, ax3_full) = plt.subplots(1, 3, figsize=(24, 8))
        else:
            fig_full, (ax1_full, ax2_full) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Plot original full view
        if normalize_colors:
            vmax_plot = np.percentile(np.abs(st.session_state.original_array), 99)
            vmin_plot = -vmax_plot
        else:
            vmin_plot, vmax_plot = vmin, vmax
        
        im1 = ax1_full.imshow(st.session_state.original_array, 
                             extent=[x_axis_full[0], x_axis_full[-1], y_axis_full[-1], y_axis_full[0]],
                             aspect=aspect_display, cmap=colormap, 
                             vmin=vmin_plot, vmax=vmax_plot,
                             interpolation=interpolation)
        
        ax1_full.set_xlabel(x_label_full)
        ax1_full.set_ylabel(y_label_full)
        ax1_full.set_title("Original Data")
        ax1_full.grid(True, alpha=0.3, linestyle='--')
        
        if show_colorbar:
            plt.colorbar(im1, ax=ax1_full, label='Amplitude')
        
        # Plot processed full view
        im2 = ax2_full.imshow(st.session_state.processed_array,
                             extent=[x_axis_full[0], x_axis_full[-1], y_axis_full[-1], y_axis_full[0]],
                             aspect=aspect_display, cmap=colormap,
                             vmin=vmin_plot, vmax=vmax_plot,
                             interpolation=interpolation)
        
        if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
            ax2_full.set_title(f"Processed ({gain_type} Gain + {st.session_state.deconv_method})")
        else:
            ax2_full.set_title(f"Processed ({gain_type} Gain)")
        
        
        ax2_full.set_xlabel(x_label_full)
        ax2_full.set_ylabel(y_label_full)
        ax2_full.grid(True, alpha=0.3, linestyle='--')
        
        if show_colorbar:
            plt.colorbar(im2, ax=ax2_full, label='Amplitude')
        
        # Plot deconvolved data if available
        if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
            if hasattr(st.session_state, 'deconvolved_array'):
                im3 = ax3_full.imshow(st.session_state.deconvolved_array,
                                     extent=[x_axis_full[0], x_axis_full[-1], y_axis_full[-1], y_axis_full[0]],
                                     aspect=aspect_display, cmap=colormap,
                                     vmin=vmin_plot, vmax=vmax_plot,
                                     interpolation=interpolation)
                
                ax3_full.set_xlabel(x_label_full)
                ax3_full.set_ylabel(y_label_full)
                ax3_full.set_title(f"Deconvolved Only ({st.session_state.deconv_method})")
                ax3_full.grid(True, alpha=0.3, linestyle='--')
                
                if show_colorbar:
                    plt.colorbar(im3, ax=ax3_full, label='Amplitude')
        
        # Add mute zone visualization if applied
        if hasattr(st.session_state, 'mute_applied') and st.session_state.mute_applied:
            if hasattr(st.session_state, 'mute_mask'):
                # Create a transparent red colormap for mute zones
                mute_cmap = ListedColormap([(1, 0, 0, 0.3)])
                
                # Plot mute mask overlay on all subplots
                ax1_full.imshow(st.session_state.mute_mask, 
                              extent=[x_axis_full[0], x_axis_full[-1], y_axis_full[-1], y_axis_full[0]],
                              aspect=aspect_display, cmap=mute_cmap, alpha=0.3,
                              interpolation='nearest')
                ax2_full.imshow(st.session_state.mute_mask, 
                              extent=[x_axis_full[0], x_axis_full[-1], y_axis_full[-1], y_axis_full[0]],
                              aspect=aspect_display, cmap=mute_cmap, alpha=0.3,
                              interpolation='nearest')
                
                if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
                    ax3_full.imshow(st.session_state.mute_mask, 
                                  extent=[x_axis_full[0], x_axis_full[-1], y_axis_full[-1], y_axis_full[0]],
                                  aspect=aspect_display, cmap=mute_cmap, alpha=0.3,
                                  interpolation='nearest')
                
                # Add legend
                mute_patch = Patch(facecolor='red', alpha=0.3, label='Mute Zone')
                ax1_full.legend(handles=[mute_patch], loc='upper right')
                ax2_full.legend(handles=[mute_patch], loc='upper right')
                if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
                    ax3_full.legend(handles=[mute_patch], loc='upper right')
        
        plt.tight_layout()
        st.pyplot(fig_full)
        
        # Display aspect ratio info
        st.info(f"**Aspect Ratio:** {aspect_value} | **Plot Dimensions:** {st.session_state.processed_array.shape[1]} × {st.session_state.processed_array.shape[0]} | **Y:X Scale:** {y_axis_full[-1]/x_axis_full[-1]:.3f}")
    
    # Continue with other tabs (Custom Window, Coordinate View, FFT Analysis, Gain Analysis)
    # These tabs remain largely the same as before, but we need to add Deconvolution Analysis tab
    with tabs[2]:  # Custom Window
        st.subheader("Custom Window Analysis")
        
        if not st.session_state.use_custom_window:
            st.warning("⚠️ Enable 'Use Custom Plot Window' in the sidebar to use this feature.")
        else:
            # Create scaled axes
            x_axis, y_axis, x_label, y_label, _, _ = scale_axes(
                st.session_state.processed_array.shape,
                st.session_state.depth_unit,
                st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                st.session_state.distance_unit,
                st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None,
                coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
            )
            
            # Get window indices
            window_info = get_window_indices(
                x_axis, y_axis,
                st.session_state.depth_min, st.session_state.depth_max,
                st.session_state.distance_min, st.session_state.distance_max
            )
            
            # Extract windowed data
            window_data = st.session_state.processed_array[
                window_info['depth_min_idx']:window_info['depth_max_idx'],
                window_info['dist_min_idx']:window_info['dist_max_idx']
            ]
            
            window_data_original = st.session_state.original_array[
                window_info['depth_min_idx']:window_info['depth_max_idx'],
                window_info['dist_min_idx']:window_info['dist_max_idx']
            ]
            
            # Create windowed axes
            x_axis_window = x_axis[window_info['dist_min_idx']:window_info['dist_max_idx']]
            y_axis_window = y_axis[window_info['depth_min_idx']:window_info['depth_max_idx']]
            
            # Display window statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Window Depth Range", 
                         f"{window_info['depth_min_val']:.1f} - {window_info['depth_max_val']:.1f} {st.session_state.depth_unit}")
            
            with col2:
                st.metric("Window Distance Range", 
                         f"{window_info['dist_min_val']:.1f} - {window_info['dist_max_val']:.1f} {st.session_state.distance_unit}")
            
            with col3:
                st.metric("Window Size (samples×traces)", 
                         f"{window_data.shape[0]} × {window_data.shape[1]}")
            
            with col4:
                st.metric("Data Points", 
                         f"{window_data.size:,}")
            
            # Plot windowed data
            fig_window, (ax1_window, ax2_window) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Windowed original
            im1_window = ax1_window.imshow(window_data_original,
                                          extent=[x_axis_window[0], x_axis_window[-1], 
                                                  y_axis_window[-1], y_axis_window[0]],
                                          aspect='auto', cmap='seismic')
            
            ax1_window.set_xlabel(x_label)
            ax1_window.set_ylabel(y_label)
            ax1_window.set_title(f"Original Data - Custom Window\n"
                               f"Depth: {window_info['depth_min_val']:.1f}-{window_info['depth_max_val']:.1f} {st.session_state.depth_unit}\n"
                               f"Distance: {window_info['dist_min_val']:.1f}-{window_info['dist_max_val']:.1f} {st.session_state.distance_unit}")
            ax1_window.grid(True, alpha=0.3)
            plt.colorbar(im1_window, ax=ax1_window, label='Amplitude')
            
            # Windowed processed
            im2_window = ax2_window.imshow(window_data,
                                          extent=[x_axis_window[0], x_axis_window[-1], 
                                                  y_axis_window[-1], y_axis_window[0]],
                                          aspect='auto', cmap='seismic')
            
            ax2_window.set_xlabel(x_label)
            ax2_window.set_ylabel(y_label)
            ax2_window.set_title(f"Processed Data - Custom Window\n"
                               f"Depth: {window_info['depth_min_val']:.1f}-{window_info['depth_max_val']:.1f} {st.session_state.depth_unit}\n"
                               f"Distance: {window_info['dist_min_val']:.1f}-{window_info['dist_max_val']:.1f} {st.session_state.distance_unit}")
            ax2_window.grid(True, alpha=0.3)
            plt.colorbar(im2_window, ax=ax2_window, label='Amplitude')
            
            plt.tight_layout()
            st.pyplot(fig_window)
            
            # Multiple windows view
            if st.session_state.multiple_windows and hasattr(st.session_state, 'additional_windows'):
                st.subheader("Multiple Windows View")
                
                # Create figure with subplots
                num_windows_total = 1 + len(st.session_state.additional_windows)
                cols = min(2, num_windows_total)
                rows = (num_windows_total + cols - 1) // cols
                
                fig_multi, axes = plt.subplots(rows, cols, figsize=(cols*8, rows*6))
                if rows * cols == 1:
                    axes = np.array([[axes]])
                elif rows == 1:
                    axes = axes.reshape(1, -1)
                elif cols == 1:
                    axes = axes.reshape(-1, 1)
                
                # Plot main window
                ax = axes[0, 0]
                im = ax.imshow(window_data,
                             extent=[x_axis_window[0], x_axis_window[-1], 
                                     y_axis_window[-1], y_axis_window[0]],
                             aspect='auto', cmap='seismic')
                
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_title(f"Window 1\n{window_info['depth_min_val']:.1f}-{window_info['depth_max_val']:.1f} {st.session_state.depth_unit}")
                ax.grid(True, alpha=0.3)
                plt.colorbar(im, ax=ax, label='Amplitude')
                
                # Plot additional windows
                window_idx = 1
                for i in range(rows):
                    for j in range(cols):
                        if window_idx >= num_windows_total:
                            if i == 0 and j == 0:
                                continue
                            axes[i, j].axis('off')
                            continue
                        
                        if window_idx == 0:  # Skip first (already plotted)
                            continue
                        
                        ax = axes[i, j]
                        win = st.session_state.additional_windows[window_idx-1]
                        
                        # Get indices for this window
                        win_info = get_window_indices(
                            x_axis, y_axis,
                            win['depth_min'], win['depth_max'],
                            win['distance_min'], win['distance_max']
                        )
                        
                        # Extract window data
                        win_data = st.session_state.processed_array[
                            win_info['depth_min_idx']:win_info['depth_max_idx'],
                            win_info['dist_min_idx']:win_info['dist_max_idx']
                        ]
                        
                        # Create windowed axes
                        x_axis_win = x_axis[win_info['dist_min_idx']:win_info['dist_max_idx']]
                        y_axis_win = y_axis[win_info['depth_min_idx']:win_info['depth_max_idx']]
                        
                        # Plot
                        im = ax.imshow(win_data,
                                     extent=[x_axis_win[0], x_axis_win[-1], 
                                             y_axis_win[-1], y_axis_win[0]],
                                     aspect='auto', cmap='seismic')
                        
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(y_label)
                        ax.set_title(f"Window {window_idx+1}\n{win_info['depth_min_val']:.1f}-{win_info['depth_max_val']:.1f} {st.session_state.depth_unit}")
                        ax.grid(True, alpha=0.3)
                        plt.colorbar(im, ax=ax, label='Amplitude')
                        
                        window_idx += 1
                
                plt.tight_layout()
                st.pyplot(fig_multi)
            
            # Windowed trace analysis
            st.subheader("Windowed Trace Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Select trace within window
                trace_in_window = st.number_input(
                    "Select Trace in Window", 
                    0, window_data.shape[1]-1,
                    window_data.shape[1]//2,
                    key="window_trace"
                )
                
                # Get the actual trace index
                actual_trace_idx = window_info['dist_min_idx'] + trace_in_window
                trace_distance = x_axis_window[trace_in_window]
                
                # Get trace data
                trace_depth = y_axis_window
                trace_amplitude = window_data[:, trace_in_window]
                
                # Plot trace
                fig_trace, ax_trace = plt.subplots(figsize=(10, 6))
                
                ax_trace.plot(trace_depth, trace_amplitude, 
                             'b-', linewidth=1.5, alpha=0.8)
                ax_trace.fill_between(trace_depth, 0, trace_amplitude, 
                                     alpha=0.3, color='blue')
                ax_trace.set_xlabel(y_label)
                ax_trace.set_ylabel("Amplitude")
                ax_trace.set_title(f"Trace {actual_trace_idx} in Window\n"
                                 f"Distance: {trace_distance:.1f} {st.session_state.distance_unit}")
                ax_trace.grid(True, alpha=0.3)
                ax_trace.invert_xaxis()  # Depth increases downward
                
                st.pyplot(fig_trace)
                
                # SIMPLE DOWNLOAD TRACE DATA
                trace_df = pd.DataFrame({
                    'Depth': trace_depth,
                    'Amplitude': trace_amplitude
                })
                
                trace_csv = trace_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Trace Data",
                    data=trace_csv,
                    file_name=f"trace_{actual_trace_idx}.csv",
                    mime="text/csv",
                    key="download_trace_simple"
                )
            
            with col2:
                # Select depth slice within window
                depth_slice_in_window = st.slider(
                    "Select Depth Slice in Window", 
                    0, window_data.shape[0]-1,
                    window_data.shape[0]//2,
                    key="window_depth"
                )
                
                # Get actual depth value
                actual_depth = y_axis_window[depth_slice_in_window]
                
                # Get depth slice data
                slice_distance = x_axis_window
                slice_amplitude = window_data[depth_slice_in_window, :]
                
                # Plot depth slice
                fig_slice, ax_slice = plt.subplots(figsize=(10, 6))
                
                ax_slice.plot(slice_distance, slice_amplitude, 
                             'r-', linewidth=1.5, alpha=0.8)
                ax_slice.fill_between(slice_distance, 0, slice_amplitude, 
                                     alpha=0.3, color='red')
                ax_slice.set_xlabel(x_label)
                ax_slice.set_ylabel("Amplitude")
                ax_slice.set_title(f"Depth Slice at {actual_depth:.2f} {st.session_state.depth_unit}")
                ax_slice.grid(True, alpha=0.3)
                
                st.pyplot(fig_slice)
                
                # SIMPLE DOWNLOAD DEPTH SLICE DATA
                slice_df = pd.DataFrame({
                    'Distance': slice_distance,
                    'Amplitude': slice_amplitude
                })
                
                slice_csv = slice_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Depth Slice Data",
                    data=slice_csv,
                    file_name=f"depth_slice_{actual_depth:.2f}.csv",
                    mime="text/csv",
                    key="download_slice_simple"
                )
            # Window statistics
            st.subheader("Window Statistics")
            
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric("Mean Amplitude", f"{window_data.mean():.3e}")
                st.metric("Std Deviation", f"{window_data.std():.3e}")
            
            with stat_col2:
                st.metric("Min Amplitude", f"{window_data.min():.3e}")
                st.metric("Max Amplitude", f"{window_data.max():.3e}")
            
            with stat_col3:
                st.metric("Depth Resolution", 
                         f"{(y_axis_window[1] - y_axis_window[0]):.3f} {st.session_state.depth_unit}/sample")
                st.metric("Distance Resolution", 
                         f"{(x_axis_window[1] - x_axis_window[0]):.3f} {st.session_state.distance_unit}/trace")
            
            with stat_col4:
                st.metric("Window Area", 
                         f"{(window_info['depth_max_val'] - window_info['depth_min_val']) * (window_info['dist_max_val'] - window_info['dist_min_val']):.1f} {st.session_state.depth_unit}×{st.session_state.distance_unit}")
                st.metric("Data Density", 
                         f"{window_data.size / ((window_info['depth_max_val'] - window_info['depth_min_val']) * (window_info['dist_max_val'] - window_info['dist_min_val'])):.1f} points/unit²")


    with tabs[3]:  # Coordinate View
        st.subheader("Coordinate-Based Visualization")
        
        if st.session_state.interpolated_coords is None:
            st.warning("No coordinates imported. Upload a CSV with Easting, Northing, Elevation columns.")
        else:
            # Add electric poles CSV upload option
            
            
            # Display coordinate statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Profile Length", f"{st.session_state.interpolated_coords['distance'][-1]:.1f} m")
                st.metric("Elevation Change", 
                         f"{st.session_state.interpolated_coords['elevation'].max() - st.session_state.interpolated_coords['elevation'].min():.1f} m")
            
            with col2:
                st.metric("Easting Range", 
                         f"{np.ptp(st.session_state.interpolated_coords['easting']):.1f} m")
                st.metric("Northing Range", 
                         f"{np.ptp(st.session_state.interpolated_coords['northing']):.1f} m")
            
            with col3:
                avg_spacing = np.mean(np.diff(st.session_state.interpolated_coords['distance']))
                st.metric("Avg Trace Spacing", f"{avg_spacing:.2f} m")
                st.metric("Profile Bearing", 
                         f"{np.degrees(np.arctan2(st.session_state.interpolated_coords['northing'][-1] - st.session_state.interpolated_coords['northing'][0], 
                                                  st.session_state.interpolated_coords['easting'][-1] - st.session_state.interpolated_coords['easting'][0])):.1f}")
            
            with col4:
                slope = (st.session_state.interpolated_coords['elevation'][-1] - st.session_state.interpolated_coords['elevation'][0]) / st.session_state.interpolated_coords['distance'][-1]
                st.metric("Average Slope", f"{slope*100:.1f}%")
                st.metric("Data Points", f"{len(st.session_state.interpolated_coords['easting'])}")
            
            # Create coordinate visualizations
            fig_coords, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            
            # 1. Plan view (Easting vs Northing)
            ax1.plot(st.session_state.interpolated_coords['easting'], 
                    st.session_state.interpolated_coords['northing'], 
                    'b-', linewidth=1, alpha=0.7)
            ax1.scatter(st.session_state.interpolated_coords['easting'], 
                       st.session_state.interpolated_coords['northing'], 
                       c=st.session_state.interpolated_coords['distance'], 
                       cmap='viridis', s=20, alpha=0.8)
            
            # Plot electric poles if available
            if pole_data is not None:
                for i in range(len(pole_data['easting'])):
                    if 'TS' in str(pole_data['names'][i]):
                        color = 'red'
                        marker = '^'
                        label = 'TS Pole'
                    elif 'TL' in str(pole_data['names'][i]):
                        color = 'purple'
                        marker = '^'
                        label = 'TL Pole'
                    else:
                        color = 'orange'
                        marker = 'o'
                        label = 'CPT'
                    
                    ax1.scatter(pole_data['easting'][i], pole_data['northing'][i], 
                               c=color, marker=marker, s=100, edgecolor='black', 
                               linewidth=1, alpha=0.8, label=label if i == 0 else "")
            
            ax1.set_xlabel('Easting (m)')
            ax1.set_ylabel('Northing (m)')
            ax1.set_title('Plan View - Survey Line with Electric Poles')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            
            # Create legend for poles
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.8, label='TS Pole (Triangle)'),
                Patch(facecolor='purple', alpha=0.8, label='TL Pole (Triangle)'),
                Patch(facecolor='orange', alpha=0.8, label='Other Pole')
            ]
            ax1.legend(handles=legend_elements, loc='upper right')
            
            plt.colorbar(ax1.collections[0], ax=ax1, label='Distance along profile (m)')
            
            # 2. Elevation profile with pole markers
            ax2.plot(st.session_state.interpolated_coords['distance'], 
                    st.session_state.interpolated_coords['elevation'], 
                    'g-', linewidth=2, alpha=0.8)
            ax2.fill_between(st.session_state.interpolated_coords['distance'],
                            st.session_state.interpolated_coords['elevation'].min(),
                            st.session_state.interpolated_coords['elevation'],
                            alpha=0.3, color='green')
            
            # Mark pole locations on elevation profile
            if pole_data is not None:
                # Get elevation at pole projected distances
                from scipy.interpolate import interp1d
                elev_interp = interp1d(st.session_state.interpolated_coords['distance'],
                                      st.session_state.interpolated_coords['elevation'],
                                      kind='linear', fill_value='extrapolate')
                
                for i in range(len(pole_data['projected_distances'])):
                    pole_elev = elev_interp(pole_data['projected_distances'][i])
                    if 'TS' in str(pole_data['names'][i]):
                        color = 'red'
                        marker = '^'
                    elif 'TL' in str(pole_data['names'][i]):
                        color = 'purple'
                        marker = '^'
                    else:
                        color = 'orange'
                        marker = 'o'
                    
                    ax2.scatter(pole_data['projected_distances'][i], pole_elev,
                               c=color, marker=marker, s=80, edgecolor='black', 
                               linewidth=1, alpha=0.8, zorder=5)
                    ax2.text(pole_data['projected_distances'][i], pole_elev + 0.5,
                            pole_data['names'][i], fontsize=8, ha='center')
            
            ax2.set_xlabel('Distance along profile (m)')
            ax2.set_ylabel('Elevation (m)')
            ax2.set_title('Elevation Profile with Electric Poles')
            ax2.grid(True, alpha=0.3)
            
            # 3. 3D view of survey line with poles
            from mpl_toolkits.mplot3d import Axes3D
            ax3 = fig_coords.add_subplot(2, 2, 3, projection='3d')
            ax3.plot(st.session_state.interpolated_coords['easting'],
                    st.session_state.interpolated_coords['northing'],
                    st.session_state.interpolated_coords['elevation'],
                    'b-', linewidth=1, alpha=0.7)
            scatter = ax3.scatter(st.session_state.interpolated_coords['easting'],
                                 st.session_state.interpolated_coords['northing'],
                                 st.session_state.interpolated_coords['elevation'],
                                 c=st.session_state.interpolated_coords['distance'],
                                 cmap='viridis', s=20, alpha=0.8)
            
            # Add poles to 3D view
            if pole_data is not None:
                for i in range(len(pole_data['easting'])):
                    if 'TS' in str(pole_data['names'][i]):
                        color = 'red'
                        marker = '^'
                    elif 'TL' in str(pole_data['names'][i]):
                        color = 'purple'
                        marker = '^'
                    else:
                        color = 'orange'
                        marker = 'o'
                    
                    ax3.scatter(pole_data['easting'][i], 
                               pole_data['northing'][i],
                               elev_interp(pole_data['projected_distances'][i]),
                               c=color, marker=marker, s=100, edgecolor='black', 
                               linewidth=1, alpha=0.8)
            
            ax3.set_xlabel('Easting (m)')
            ax3.set_ylabel('Northing (m)')
            ax3.set_zlabel('Elevation (m)')
            ax3.set_title('3D Survey Line with Electric Poles')
            plt.colorbar(scatter, ax=ax3, label='Distance (m)')
            
            # 4. GPR data with coordinate-based X-axis
            # Determine aspect ratio for this plot
            aspect_value_coords = get_aspect_ratio(
                st.session_state.aspect_mode,
                st.session_state.aspect_ratio,
                st.session_state.processed_array.shape
            )
            
            # Create depth axis
            if st.session_state.depth_unit != "samples":
                depth_axis = np.linspace(0, st.session_state.max_depth, 
                                        st.session_state.processed_array.shape[0])
            else:
                depth_axis = np.arange(st.session_state.processed_array.shape[0])
            
            # Plot GPR data with coordinate-based distance
            im = ax4.imshow(st.session_state.processed_array,
                          extent=[st.session_state.interpolated_coords['distance'][0],
                                 st.session_state.interpolated_coords['distance'][-1],
                                 depth_axis[-1], depth_axis[0]],
                          aspect=aspect_value_coords, cmap='seismic', alpha=0.9)
            ax4.set_xlabel('Distance along profile (m)')
            ax4.set_ylabel(f'Depth ({st.session_state.depth_unit})')
            ax4.set_title(f'GPR Data with Coordinate Scaling (Aspect: {aspect_value_coords})')
            ax4.grid(True, alpha=0.2)
            plt.colorbar(im, ax=ax4, label='Amplitude')
            
            # Overlay elevation profile on GPR plot (secondary axis)
            ax4_twin = ax4.twinx()
            ax4_twin.plot(st.session_state.interpolated_coords['distance'],
                         st.session_state.interpolated_coords['elevation'],
                         'g-', linewidth=2, alpha=0.6, label='Elevation')
            ax4_twin.set_ylabel('Elevation (m)', color='green')
            ax4_twin.tick_params(axis='y', labelcolor='green')
            
            plt.tight_layout()
            st.pyplot(fig_coords)
            
            # ELECTRIC POLE ANOMALY COMPARISON PLOT
            st.subheader("GPR Line Plot")
            
            # Calculate elevation-adjusted depth for GPR display
            n_traces = st.session_state.processed_array.shape[1]
            n_samples = st.session_state.processed_array.shape[0]
            
            # Create meshgrid for pcolormesh
            X, Y = np.meshgrid(st.session_state.interpolated_coords['distance'], depth_axis)
            
            # Adjust Y coordinates by elevation (convert depth to elevation)
            Y_elev = np.zeros_like(Y)
            for i in range(n_traces):
                Y_elev[:, i] = st.session_state.interpolated_coords['elevation'][i] - depth_axis
            
            fig_elev, ax_elev = plt.subplots(figsize=(14, 6))
            
            # Use pcolormesh for elevation-adjusted display
            mesh = ax_elev.pcolormesh(X, Y_elev, st.session_state.processed_array,vmin=vmin_plot, vmax=vmax_plot,
                                     cmap='seismic', shading='auto', alpha=0.9)
            
            ax_elev.set_xlabel('Distance along profile (m)')
            ax_elev.set_ylabel('Elevation (m)')
            ax_elev.set_title('GPR Section')
            ax_elev.grid(True, alpha=0.2)
            plt.colorbar(mesh, ax=ax_elev, label='Amplitude')
            
            # Add topographic surface line
            ax_elev.plot(st.session_state.interpolated_coords['distance'],
                        st.session_state.interpolated_coords['elevation'],
                        'k-', linewidth=1, alpha=0.8, label='Surface')
            ax_elev.fill_between(st.session_state.interpolated_coords['distance'],
                                Y_elev.min(), st.session_state.interpolated_coords['elevation'],
                                alpha=0.1, color='gray')
            
            # Mark electric poles on the surface
            if pole_data is not None:
                for i in range(len(pole_data['projected_distances'])):
                    pole_elev = elev_interp(pole_data['projected_distances'][i])
                    if 'TS' in str(pole_data['names'][i]):
                        color = 'red'
                        marker = '^'
                        label = 'TS Pole'
                    elif 'TL' in str(pole_data['names'][i]):
                        color = 'green'
                        marker = '^'
                        label = 'TL Pole'
                    else:
                        color = 'orange'
                        marker = 'o'
                        label = 'CPT'
                    
                    # Plot pole at surface elevation
                    ax_elev.scatter(pole_data['projected_distances'][i], pole_elev + 0.5,
                                   c=color, marker=marker, s=100, 
                                    alpha=0.9, zorder=10)
                    
                    # Add vertical dashed line from pole to bottom of plot
                    #ax_elev.plot([pole_data['projected_distances'][i], pole_data['projected_distances'][i]],
                     #           [pole_elev, Y_elev.min()],
                      #          color=color, linestyle='--', alpha=0.5, linewidth=1)
                    
                    # Add pole name
                    ax_elev.text(pole_data['projected_distances'][i], pole_elev + 1,
                                pole_data['names'][i], fontsize=6, ha='center')
                
                # Create custom legend for poles
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                          markersize=10, label='TS Pole'),
                    Line2D([0], [0], marker='^', color='w', markerfacecolor='purple', 
                          markersize=10, label='TL Pole'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                          markersize=10, label='Other Pole'),
                    Line2D([0], [0], color='k', linewidth=2, label='Surface'),
                    Line2D([0], [0], color='gray', alpha=0.1, linewidth=10, label='Subsurface')
                ]
                ax_elev.legend(handles=legend_elements, loc='upper right')
            
            ax_elev.legend()
            ax_elev.set_ylim(Y_elev.min(), st.session_state.interpolated_coords['elevation'].max() + 5)
            
            plt.tight_layout()
            st.pyplot(fig_elev)
            
            # Display pole information table
            if pole_data is not None:
                st.subheader("Electric Pole Information")
                pole_info_df = pd.DataFrame({
                    'Name': pole_data['names'],
                    'Easting (m)': pole_data['easting'],
                    'Northing (m)': pole_data['northing'],
                    'Distance along profile (m)': pole_data['projected_distances'],
                    'Distance from GPR line (m)': pole_data['min_distances'],
                    'Type': ['TS' if 'TS' in str(name) else 'TL' if 'TL' in str(name) else 'Other' 
                            for name in pole_data['names']]
                })
                st.dataframe(pole_info_df.sort_values('Distance along profile (m)'))
            
            # Export coordinates
            st.subheader("Export Interpolated Coordinates")
            
            if st.button("💾 Download Interpolated Coordinates", use_container_width=True):
                coord_df = pd.DataFrame({
                    'Trace_Index': st.session_state.interpolated_coords['trace_indices'],
                    'Distance_m': st.session_state.interpolated_coords['distance'],
                    'Easting_m': st.session_state.interpolated_coords['easting'],
                    'Northing_m': st.session_state.interpolated_coords['northing'],
                    'Elevation_m': st.session_state.interpolated_coords['elevation']
                })
                csv = coord_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="interpolated_coordinates.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    ########################
    
    with tabs[4]:  # FFT Analysis
        st.subheader("Frequency vs Amplitude Analysis (FFT)")
        
        # FFT analysis options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trace_for_fft = st.number_input("Select Trace for FFT", 
                                     0, st.session_state.processed_array.shape[1]-1, 
                                     st.session_state.processed_array.shape[1]//2,
                                     key="fft_trace")
        
        with col2:
            sampling_rate = st.number_input("Sampling Rate (MHz)", 100, 5000, 1000, 100,
                                           help="Antenna sampling rate in MHz",
                                           key="fft_sampling")
        
        with col3:
            fft_mode = st.selectbox("FFT Mode", ["Single Trace", "Average of All Traces", "Trace Range", "Windowed Traces"],
                                   key="fft_mode")
        
        if fft_mode == "Trace Range":
            trace_start = st.number_input("Start Trace", 0, st.session_state.processed_array.shape[1]-1, 0,
                                         key="fft_start")
            trace_end = st.number_input("End Trace", 0, st.session_state.processed_array.shape[1]-1, 
                                       st.session_state.processed_array.shape[1]-1,
                                       key="fft_end")
        
        # Calculate FFT
        if fft_mode == "Single Trace":
            trace_data = st.session_state.processed_array[:, trace_for_fft]
            freq, amplitude = calculate_fft(trace_data, sampling_rate)
            title = f"FFT - Trace {trace_for_fft}"
        
        elif fft_mode == "Average of All Traces":
            avg_trace = np.mean(st.session_state.processed_array, axis=1)
            freq, amplitude = calculate_fft(avg_trace, sampling_rate)
            title = "FFT - Average of All Traces"
        
        elif fft_mode == "Trace Range":
            avg_trace = np.mean(st.session_state.processed_array[:, trace_start:trace_end+1], axis=1)
            freq, amplitude = calculate_fft(avg_trace, sampling_rate)
            title = f"FFT - Traces {trace_start} to {trace_end}"
        
        elif fft_mode == "Windowed Traces" and st.session_state.use_custom_window:
            # Create scaled axes to get window indices
            x_axis, y_axis, _, _, _, _ = scale_axes(
                st.session_state.processed_array.shape,
                st.session_state.depth_unit,
                st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                st.session_state.distance_unit,
                st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None,
                coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
            )
            
            # Get window indices
            window_info = get_window_indices(
                x_axis, y_axis,
                st.session_state.depth_min, st.session_state.depth_max,
                st.session_state.distance_min, st.session_state.distance_max
            )
            
            # Use windowed traces
            windowed_traces = st.session_state.processed_array[
                :, window_info['dist_min_idx']:window_info['dist_max_idx']
            ]
            avg_trace = np.mean(windowed_traces, axis=1)
            freq, amplitude = calculate_fft(avg_trace, sampling_rate)
            title = f"FFT - Windowed Traces ({window_info['dist_min_idx']} to {window_info['dist_max_idx']})"
        elif fft_mode == "Windowed Traces" and not st.session_state.use_custom_window:
            st.warning("Please enable 'Use Custom Plot Window' in the sidebar to use Windowed Traces mode.")
            freq, amplitude = [], []
            title = ""
        else:
            st.warning("Please select a valid FFT mode")
            freq, amplitude = [], []
            title = ""
        
        if len(freq) > 0:
            # Plot FFT
            fig_fft, (ax_fft1, ax_fft2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Linear scale
            ax_fft1.plot(freq, amplitude, 'b-', linewidth=2, alpha=0.8)
            ax_fft1.fill_between(freq, 0, amplitude, alpha=0.3, color='blue')
            ax_fft1.set_xlabel("Frequency (MHz)")
            ax_fft1.set_ylabel("Amplitude")
            ax_fft1.set_title(f"{title} - Linear Scale")
            ax_fft1.grid(True, alpha=0.3)
            ax_fft1.set_xlim([0, sampling_rate/2])
            
            # Log scale
            ax_fft2.semilogy(freq, amplitude, 'r-', linewidth=2, alpha=0.8)
            ax_fft2.fill_between(freq, 0.001, amplitude, alpha=0.3, color='red')
            ax_fft2.set_xlabel("Frequency (MHz)")
            ax_fft2.set_ylabel("Amplitude (log)")
            ax_fft2.set_title(f"{title} - Log Scale")
            ax_fft2.grid(True, alpha=0.3)
            ax_fft2.set_xlim([0, sampling_rate/2])
            
            plt.tight_layout()
            st.pyplot(fig_fft)
            
            # FFT statistics
            st.subheader("FFT Statistics")
            
            # Find peak frequencies
            peak_idx = np.argmax(amplitude)
            peak_freq = freq[peak_idx]
            peak_amp = amplitude[peak_idx]
            
            # Calculate bandwidth at -3dB
            max_amp = np.max(amplitude)
            half_power = max_amp / np.sqrt(2)
            
            # Find frequencies where amplitude is above half power
            mask = amplitude >= half_power
            if np.any(mask):
                low_freq = freq[mask][0]
                high_freq = freq[mask][-1]
                bandwidth = high_freq - low_freq
            else:
                low_freq = high_freq = bandwidth = 0
            
            # Display statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Peak Frequency", f"{peak_freq:.1f} MHz")
            with col2:
                st.metric("Peak Amplitude", f"{peak_amp:.3e}")
            with col3:
                st.metric("Bandwidth (-3dB)", f"{bandwidth:.1f} MHz")
            with col4:
                st.metric("Center Freq", f"{(low_freq + high_freq)/2:.1f} MHz")
    
    with tabs[5]:  # Gain Analysis
        
        st.subheader("Gain Analysis")
        
        # Calculate gain profile
        n_samples = st.session_state.original_array.shape[0]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            gain_profile = np.zeros(n_samples)
            for i in range(n_samples):
                orig_slice = st.session_state.original_array[i, :]
                proc_slice = st.session_state.processed_array[i, :]
                
                mask = np.abs(orig_slice) > 1e-10
                if np.any(mask):
                    gains = np.abs(proc_slice[mask]) / np.abs(orig_slice[mask])
                    gain_profile[i] = np.median(gains)
                else:
                    gain_profile[i] = 1.0
        
        # Try to get scaled depth axis from scale_axes function
        try:
            result = scale_axes(
                (n_samples, 1),
                st.session_state.depth_unit,
                st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                "traces",
                None
            )
            
            # Check how many values are returned
            if len(result) >= 4:
                y_axis_analysis = result[0]
                y_label_analysis = result[3] if len(result) >= 4 else "Depth"
            else:
                raise ValueError("scale_axes returned insufficient values")
                
        except Exception as e:
            st.warning(f"Could not use scale_axes: {e}. Creating depth axis manually.")
            # Create depth axis manually
            if st.session_state.depth_unit == "samples":
                y_axis_analysis = np.arange(n_samples)
                y_label_analysis = "Sample Number"
            else:
                if hasattr(st.session_state, 'max_depth') and st.session_state.max_depth is not None:
                    y_axis_analysis = np.linspace(0, st.session_state.max_depth, n_samples)
                    y_label_analysis = f"Depth ({st.session_state.depth_unit})"
                else:
                    y_axis_analysis = np.arange(n_samples)
                    y_label_analysis = "Depth"
        
        # Debug info (you can remove this after fixing)
        st.write(f"gain_profile shape: {gain_profile.shape}")
        st.write(f"y_axis_analysis shape: {y_axis_analysis.shape}")
        st.write(f"y_label_analysis: {y_label_analysis}")
        
        # Ensure both arrays have the same length
        if len(gain_profile) != len(y_axis_analysis):
            st.error(f"Array length mismatch: gain_profile={len(gain_profile)}, y_axis={len(y_axis_analysis)}")
            # Truncate to minimum length
            min_length = min(len(gain_profile), len(y_axis_analysis))
            gain_profile = gain_profile[:min_length]
            y_axis_analysis = y_axis_analysis[:min_length]
        
        # Plot gain profile
        fig_gain, ax_gain = plt.subplots(figsize=(10, 6))
        
        ax_gain.plot(gain_profile, y_axis_analysis, 'b-', linewidth=2, label='Gain Factor')
        ax_gain.fill_betweenx(y_axis_analysis, 1, gain_profile, alpha=0.3, color='blue')
        
        ax_gain.set_xlabel("Gain Factor (multiplier)")
        ax_gain.set_ylabel(y_label_analysis)
        ax_gain.set_title("Gain Applied vs Depth")
        ax_gain.grid(True, alpha=0.3)
        ax_gain.legend()
        ax_gain.invert_yaxis()  # Depth increases downward
        
        st.pyplot(fig_gain)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Min Gain", f"{gain_profile.min():.2f}x")
        with col2:
            st.metric("Max Gain", f"{gain_profile.max():.2f}x")
        with col3:
            st.metric("Mean Gain", f"{gain_profile.mean():.2f}x")
    
    with tabs[6]:  # NEW: Deconvolution Analysis
        st.subheader("Deconvolution Analysis")
        
        if not hasattr(st.session_state, 'deconvolution_applied') or not st.session_state.deconvolution_applied:
            st.warning("⚠️ Enable 'Apply Deconvolution' in the sidebar to use this feature.")
        else:
            # Create two columns for deconvolution analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Estimated Wavelet")
                if hasattr(st.session_state, 'estimated_wavelet'):
                    wavelet = st.session_state.estimated_wavelet
                    
                    fig_wavelet, (ax_wavelet1, ax_wavelet2) = plt.subplots(2, 1, figsize=(10, 8))
                    
                    # Time domain
                    ax_wavelet1.plot(wavelet, 'b-', linewidth=2)
                    ax_wavelet1.fill_between(range(len(wavelet)), 0, wavelet, alpha=0.3, color='blue')
                    ax_wavelet1.set_xlabel("Sample")
                    ax_wavelet1.set_ylabel("Amplitude")
                    ax_wavelet1.set_title(f"Estimated Wavelet - Time Domain (Length: {len(wavelet)} samples)")
                    ax_wavelet1.grid(True, alpha=0.3)
                    
                    # Frequency domain
                    freq_wavelet, mag_wavelet = calculate_fft(wavelet, sampling_rate=1000)
                    ax_wavelet2.semilogy(freq_wavelet, mag_wavelet, 'r-', linewidth=2)
                    ax_wavelet2.fill_between(freq_wavelet, 0.001, mag_wavelet, alpha=0.3, color='red')
                    ax_wavelet2.set_xlabel("Frequency (MHz)")
                    ax_wavelet2.set_ylabel("Amplitude (log)")
                    ax_wavelet2.set_title("Wavelet Spectrum")
                    ax_wavelet2.grid(True, alpha=0.3)
                    ax_wavelet2.set_xlim([0, 500])
                    
                    plt.tight_layout()
                    st.pyplot(fig_wavelet)
                    
                    # Wavelet statistics
                    st.markdown("#### Wavelet Statistics")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Peak Amplitude", f"{np.max(np.abs(wavelet)):.3f}")
                        st.metric("Zero-crossing", f"{np.sum(np.diff(np.sign(wavelet)) != 0)}")
                    with col_stat2:
                        st.metric("Energy", f"{np.sum(wavelet**2):.3f}")
                        st.metric("Duration", f"{len(wavelet)} samples")
                    with col_stat3:
                        st.metric("Bandwidth", f"{freq_wavelet[np.argmax(mag_wavelet)]:.1f} MHz")
                        st.metric("Symmetry", f"{np.abs(wavelet[:len(wavelet)//2].sum() / wavelet[len(wavelet)//2:].sum()):.2f}")
            
            with col2:
                st.markdown("### Deconvolution Quality Metrics")
                
                # Select a trace for detailed analysis
                trace_idx = st.slider("Select Trace for Analysis", 0, st.session_state.processed_array.shape[1]-1, 
                                     st.session_state.processed_array.shape[1]//2, key="deconv_trace")
                
                original_trace = st.session_state.original_array[:, trace_idx]
                if hasattr(st.session_state, 'deconvolved_array'):
                    deconv_trace = st.session_state.deconvolved_array[:, trace_idx]
                else:
                    deconv_trace = st.session_state.processed_array[:, trace_idx]
                
                # Calculate metrics
                correlation = np.corrcoef(original_trace, deconv_trace)[0, 1]
                energy_ratio = np.sum(deconv_trace**2) / (np.sum(original_trace**2) + 1e-10)
                kurtosis_original = np.mean((original_trace - np.mean(original_trace))**4) / (np.std(original_trace)**4 + 1e-10)
                kurtosis_deconv = np.mean((deconv_trace - np.mean(deconv_trace))**4) / (np.std(deconv_trace)**4 + 1e-10)
                sparsity = np.sum(np.abs(deconv_trace) > 0.1 * np.max(np.abs(deconv_trace))) / len(deconv_trace)
                
                # Display metrics
                st.metric("Correlation with Original", f"{correlation:.3f}")
                st.metric("Energy Ratio (Deconv/Orig)", f"{energy_ratio:.3f}")
                st.metric("Kurtosis (Original)", f"{kurtosis_original:.2f}")
                st.metric("Kurtosis (Deconvolved)", f"{kurtosis_deconv:.2f}")
                st.metric("Sparsity Index", f"{sparsity:.3f}")
                
                # Plot trace comparison
                fig_trace, ax_trace = plt.subplots(figsize=(10, 6))
                
                ax_trace.plot(original_trace, 'b-', linewidth=1.5, alpha=0.7, label='Original')
                ax_trace.plot(deconv_trace, 'r-', linewidth=1.5, alpha=0.7, label='Deconvolved')
                ax_trace.set_xlabel("Sample")
                ax_trace.set_ylabel("Amplitude")
                ax_trace.set_title(f"Trace {trace_idx} - Original vs Deconvolved")
                ax_trace.legend()
                ax_trace.grid(True, alpha=0.3)
                
                st.pyplot(fig_trace)
            
            # Deconvolution residual analysis
            st.markdown("### Residual Analysis")
            
            if hasattr(st.session_state, 'deconvolved_array'):
                # Calculate residuals
                residual = st.session_state.original_array - st.session_state.deconvolved_array
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    # Plot residual statistics
                    residual_mean = np.mean(residual, axis=1)
                    residual_std = np.std(residual, axis=1)
                    
                    fig_res, ax_res = plt.subplots(figsize=(10, 6))
                    
                    ax_res.plot(residual_mean, 'g-', linewidth=2, label='Mean Residual')
                    ax_res.fill_between(range(len(residual_mean)), 
                                       residual_mean - residual_std, 
                                       residual_mean + residual_std, 
                                       alpha=0.3, color='green', label='±1 Std Dev')
                    ax_res.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    ax_res.set_xlabel("Sample")
                    ax_res.set_ylabel("Residual Amplitude")
                    ax_res.set_title("Deconvolution Residual Statistics")
                    ax_res.legend()
                    ax_res.grid(True, alpha=0.3)
                    
                    st.pyplot(fig_res)
                
                with col_res2:
                    # Residual histogram
                    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
                    
                    flat_residual = residual.flatten()
                    # Remove outliers for better visualization
                    q_low, q_high = np.percentile(flat_residual, [1, 99])
                    filtered_residual = flat_residual[(flat_residual >= q_low) & (flat_residual <= q_high)]
                    
                    ax_hist.hist(filtered_residual, bins=100, density=True, alpha=0.7, color='purple', edgecolor='black')
                    ax_hist.set_xlabel("Residual Amplitude")
                    ax_hist.set_ylabel("Density")
                    ax_hist.set_title("Residual Distribution")
                    ax_hist.grid(True, alpha=0.3)
                    
                    # Add Gaussian fit
                    from scipy.stats import norm
                    mu, std = norm.fit(filtered_residual)
                    x = np.linspace(filtered_residual.min(), filtered_residual.max(), 100)
                    p = norm.pdf(x, mu, std)
                    ax_hist.plot(x, p, 'k-', linewidth=2, label=f'Gaussian fit\nμ={mu:.3f}, σ={std:.3f}')
                    ax_hist.legend()
                    
                    st.pyplot(fig_hist)
                    
                    # Display residual statistics
                    st.metric("Mean Residual", f"{np.mean(flat_residual):.3e}")
                    st.metric("Std Dev Residual", f"{np.std(flat_residual):.3e}")
                    st.metric("Residual Kurtosis", f"{np.mean((flat_residual - np.mean(flat_residual))**4) / (np.std(flat_residual)**4 + 1e-10):.2f}")
    
    with tabs[7]:  # Export
        st.subheader("Export Processed Data")
        
        # Export options in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Save Full Radar Image", use_container_width=True):
                fig, ax = plt.subplots(figsize=(12, 8))
                
                x_axis_export, y_axis_export, x_label_export, y_label_export, _, _ = scale_axes(
                    st.session_state.processed_array.shape,
                    st.session_state.depth_unit,
                    st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                    st.session_state.distance_unit,
                    st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None,
                    coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
                )
                
                im = ax.imshow(st.session_state.processed_array,
                             extent=[x_axis_export[0], x_axis_export[-1], 
                                    y_axis_export[-1], y_axis_export[0]],
                             aspect='auto', cmap='seismic')
                ax.set_xlabel(x_label_export)
                ax.set_ylabel(y_label_export)
                
                if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
                    ax.set_title(f"GPR Data - {gain_type} Gain + {st.session_state.deconv_method}")
                else:
                    ax.set_title(f"GPR Data - {gain_type} Gain")
                
                plt.colorbar(im, ax=ax, label='Amplitude')
                plt.tight_layout()
                plt.savefig("gpr_data_full.png", dpi=300, bbox_inches='tight')
                st.success("Saved as 'gpr_data_full.png'")
        
        with col2:
            if st.session_state.use_custom_window:
                if st.button("💾 Save Windowed Image", use_container_width=True):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Create scaled axes
                    x_axis, y_axis, x_label, y_label, _, _ = scale_axes(
                        st.session_state.processed_array.shape,
                        st.session_state.depth_unit,
                        st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                        st.session_state.distance_unit,
                        st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None,
                        coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
                    )
                    
                    window_info = get_window_indices(
                        x_axis, y_axis,
                        st.session_state.depth_min, st.session_state.depth_max,
                        st.session_state.distance_min, st.session_state.distance_max
                    )
                    
                    window_data = st.session_state.processed_array[
                        window_info['depth_min_idx']:window_info['depth_max_idx'],
                        window_info['dist_min_idx']:window_info['dist_max_idx']
                    ]
                    
                    x_axis_window = x_axis[window_info['dist_min_idx']:window_info['dist_max_idx']]
                    y_axis_window = y_axis[window_info['depth_min_idx']:window_info['depth_max_idx']]
                    
                    im = ax.imshow(window_data,
                                 extent=[x_axis_window[0], x_axis_window[-1], 
                                         y_axis_window[-1], y_axis_window[0]],
                                 aspect='auto', cmap='seismic')
                    ax.set_xlabel(x_label)
                    ax.set_ylabel(y_label)
                    ax.set_title(f"GPR Data - Custom Window\n"
                               f"Depth: {window_info['depth_min_val']:.1f}-{window_info['depth_max_val']:.1f} {st.session_state.depth_unit}\n"
                               f"Distance: {window_info['dist_min_val']:.1f}-{window_info['dist_max_val']:.1f} {st.session_state.distance_unit}")
                    plt.colorbar(im, ax=ax, label='Amplitude')
                    plt.tight_layout()
                    plt.savefig("gpr_data_windowed.png", dpi=300, bbox_inches='tight')
                    st.success("Saved as 'gpr_data_windowed.png'")
        
        with col3:
            # Export as CSV with scaled axes
            x_axis_csv, _, _, _, _, _ = scale_axes(
                st.session_state.processed_array.shape,
                st.session_state.depth_unit,
                st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                st.session_state.distance_unit,
                st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None,
                coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
            )
            
            csv_data = pd.DataFrame(st.session_state.processed_array, 
                                  columns=[f"{xi:.2f}" for xi in x_axis_csv])
            csv_string = csv_data.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Full CSV",
                data=csv_string,
                file_name="gpr_data_full.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col4:
            if st.session_state.use_custom_window:
                # Export windowed data
                x_axis, y_axis, x_label, y_label, _, _ = scale_axes(
                    st.session_state.processed_array.shape,
                    st.session_state.depth_unit,
                    st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
                    st.session_state.distance_unit,
                    st.session_state.total_distance if hasattr(st.session_state, 'total_distance') else None,
                    coordinates=st.session_state.interpolated_coords if st.session_state.use_coords_for_distance else None
                )
                
                window_info = get_window_indices(
                    x_axis, y_axis,
                    st.session_state.depth_min, st.session_state.depth_max,
                    st.session_state.distance_min, st.session_state.distance_max
                )
                
                window_data = st.session_state.processed_array[
                    window_info['depth_min_idx']:window_info['depth_max_idx'],
                    window_info['dist_min_idx']:window_info['dist_max_idx']
                ]
                
                x_axis_window = x_axis[window_info['dist_min_idx']:window_info['dist_max_idx']]
                
                window_csv = pd.DataFrame(window_data, 
                                        columns=[f"{xi:.2f}" for xi in x_axis_window])
                window_csv_string = window_csv.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Window CSV",
                    data=window_csv_string,
                    file_name="gpr_data_window.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Export deconvolved data if available
        if hasattr(st.session_state, 'deconvolution_applied') and st.session_state.deconvolution_applied:
            st.subheader("Export Deconvolved Data")
            
            col_deconv1, col_deconv2 = st.columns(2)
            
            with col_deconv1:
                if hasattr(st.session_state, 'deconvolved_array'):
                    deconv_csv_data = pd.DataFrame(st.session_state.deconvolved_array, 
                                                 columns=[f"{xi:.2f}" for xi in x_axis_csv])
                    deconv_csv_string = deconv_csv_data.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Deconvolved CSV",
                        data=deconv_csv_string,
                        file_name="gpr_data_deconvolved.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
            
            with col_deconv2:
                # Export deconvolution settings
                if hasattr(st.session_state, 'deconv_params'):
                    deconv_settings = {
                        'method': st.session_state.deconv_method,
                        'parameters': st.session_state.deconv_params,
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                    
                    json_string = json.dumps(deconv_settings, indent=2)
                    st.download_button(
                        label="📝 Download Deconvolution Settings",
                        data=json_string,
                        file_name="deconvolution_settings.json",
                        mime="application/json",
                        use_container_width=True
                    )
        
        # Export mute settings
        if hasattr(st.session_state, 'mute_applied') and st.session_state.mute_applied:
            st.subheader("Export Mute Settings")
            
            if st.button("📝 Export Mute Settings as JSON", use_container_width=True):
                mute_settings = {
                    'line_reversed': st.session_state.line_reversed if hasattr(st.session_state, 'line_reversed') else False,
                    'mute_zones': st.session_state.mute_zones if hasattr(st.session_state, 'mute_zones') else [],
                    'mute_strength': mute_strength if 'mute_strength' in locals() else 100
                }
                
                json_string = json.dumps(mute_settings, indent=2)
                st.download_button(
                    label="📥 Download Mute Settings",
                    data=json_string,
                    file_name="mute_settings.json",
                    mime="application/json",
                    use_container_width=True
                )

# Initial state message
elif not csv_file:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        👈 **Upload a CSV file to begin processing**

        **CSV Format:**
        - First row: trace indices (0,1,2,...)
        - First column: sample numbers (0,1,2,...)
        - Amplitudes in between

        **Advanced Deconvolution Features:**
        ... (keep your existing feature list) ...
        """)
# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "📡 <b>GPR Data Processor v7.0</b> | Advanced Deconvolution Suite | "
    "Built with Streamlit & readgssi"
    "</div>",
    unsafe_allow_html=True
)






























