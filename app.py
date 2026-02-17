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
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.interpolate import interp1d, griddata
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="GPR Data Processor",
    page_icon="📡",
    layout="wide"
)

# Title
st.title("📡 GPR Data Processor with Coordinate Import & Aspect Control")
st.markdown("Process GPR data with CSV coordinate import, interpolation, and aspect ratio control")

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'original_array' not in st.session_state:
    st.session_state.original_array = None
if 'processed_array' not in st.session_state:
    st.session_state.processed_array = None
if 'coordinates' not in st.session_state:
    st.session_state.coordinates = None
if 'interpolated_coords' not in st.session_state:
    st.session_state.interpolated_coords = None

# Sidebar
with st.sidebar:
    st.header("📂 File Upload")
    
    dzt_file = st.file_uploader("Upload DZT file", type=['dzt', 'DZT', '.dzt'])
    dzg_file = st.file_uploader("Upload DZG file (GPS data)", type=['dzg', 'DZG'], 
                                help="Optional: Required for GPS-based distance normalization")
    
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

# Helper functions
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

def calculate_fft(trace, sampling_rate=1000):
    """Calculate FFT of a trace"""
    n = len(trace)
    yf = fft(trace)
    xf = fftfreq(n, 1/sampling_rate)[:n//2]
    
    # Take magnitude
    magnitude = 2.0/n * np.abs(yf[:n//2])
    
    return xf, magnitude

def process_coordinates(coords_df, n_traces, trace_col=None, method='linear'):
    """
    Process and interpolate coordinates to match number of GPR traces
    
    Parameters:
    - coords_df: DataFrame with Easting, Northing, Elevation columns
    - n_traces: Number of traces in GPR data
    - trace_col: Column name for trace indices in CSV (optional)
    - method: Interpolation method ('linear', 'cubic', 'nearest', 'previous', 'next')
    
    Returns:
    - Dictionary with interpolated coordinates and distance along profile
    """
    # Check required columns
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
        # Use the cumulative distance along the profile
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
        # Auto-calculate based on data dimensions
        return data_shape[0] / data_shape[1] * 0.5  # Default aspect
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
if dzt_file and process_btn:
    with st.spinner("Processing radar data..."):
        try:
            # Try to import readgssi
            try:
                from readgssi import readgssi
            except ImportError as e:
                st.error(f"Failed to import readgssi: {e}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Save files to temp location
            with tempfile.TemporaryDirectory() as tmpdir:
                progress_bar.progress(10)
                
                # Save DZT
                dzt_path = os.path.join(tmpdir, "input.dzt")
                with open(dzt_path, "wb") as f:
                    f.write(dzt_file.getbuffer())
                
                # Save DZG if provided
                dzg_path = None
                if dzg_file:
                    dzg_path = os.path.join(tmpdir, "input.dzg")
                    with open(dzg_path, "wb") as f:
                        f.write(dzg_file.getbuffer())
                
                progress_bar.progress(30)
                
                # Process coordinates if provided
                coordinates_data = None
                if coord_csv:
                    try:
                        coords_df = pd.read_csv(coord_csv)
                        st.session_state.coordinates = coords_df
                        st.info(f"Loaded {len(coords_df)} coordinate points")
                    except Exception as e:
                        st.warning(f"Could not read CSV coordinates: {str(e)}")
                        coord_csv = None
                
                progress_bar.progress(40)
                
                # Build parameters for readgssi
                params = {
                    'infile': dzt_path,
                    'zero': [time_zero],
                    'verbose': False
                }
                
                # Add stacking
                if stacking == "auto":
                    params['stack'] = 'auto'
                elif stacking == "manual":
                    params['stack'] = stack_value
                
                # Add BGR
                if bgr:
                    if bgr_type == "Full-width":
                        params['bgr'] = 0
                    else:
                        params['bgr'] = bgr_window
                
                # Add frequency filter
                if freq_filter:
                    params['freqmin'] = freq_min
                    params['freqmax'] = freq_max
                
                progress_bar.progress(50)
                
                # Read data
                header, arrays, gps = readgssi.readgssi(**params)
                
                progress_bar.progress(70)
                
                # Store original array
                if arrays and len(arrays) > 0:
                    original_array = arrays[0]
                    
                    # Apply time-varying gain
                    processed_array = original_array.copy()
                    
                    # Apply near-surface correction if requested
                    if apply_near_surface_correction:
                        # Store correction parameters
                        st.session_state.near_surface_correction = True
                        st.session_state.correction_type = correction_type
                        st.session_state.correction_depth = correction_depth
                        
                        # Apply the correction
                        processed_array = apply_near_surface_correction(
                            processed_array, 
                            correction_type, 
                            correction_depth, 
                            max_depth if depth_unit != "samples" else None,
                            **locals()
                        )
                    
                    # Apply selected gain
                    if gain_type == "Constant":
                        processed_array = apply_gain(processed_array, "Constant", 
                                                    const_gain=const_gain)
                    elif gain_type == "Linear":
                        processed_array = apply_gain(processed_array, "Linear",
                                                    min_gain=min_gain, max_gain=max_gain)
                    elif gain_type == "Exponential":
                        processed_array = apply_gain(processed_array, "Exponential",
                                                    base_gain=base_gain, exp_factor=exp_factor)
                    elif gain_type == "AGC (Automatic Gain Control)":
                        processed_array = apply_gain(processed_array, "AGC (Automatic Gain Control)",
                                                    window_size=window_size, target_amplitude=target_amplitude)
                    elif gain_type == "Spherical":
                        processed_array = apply_gain(processed_array, "Spherical",
                                                    power_gain=power_gain, attenuation=attenuation)
                    
                    progress_bar.progress(80)
                    
                    # Process coordinates if provided
                    if coord_csv and st.session_state.coordinates is not None:
                        try:
                            coordinates_data = process_coordinates(
                                st.session_state.coordinates,
                                processed_array.shape[1],
                                trace_col=trace_col if 'trace_col' in locals() else None,
                                method=interp_method.lower() if 'interp_method' in locals() else 'linear'
                            )
                            st.session_state.interpolated_coords = coordinates_data
                            if coordinates_data:
                                st.success(f"✓ Interpolated {coordinates_data['original_points']} coordinate points to {coordinates_data['interpolated_points']} traces")
                        except Exception as e:
                            st.warning(f"Coordinate processing failed: {str(e)}")
                    
                    progress_bar.progress(90)
                    
                    # Store in session state
                    st.session_state.header = header
                    st.session_state.original_array = original_array
                    st.session_state.processed_array = processed_array
                    st.session_state.gps = gps
                    st.session_state.data_loaded = True
                    
                    # Store axis scaling parameters
                    st.session_state.depth_unit = depth_unit
                    st.session_state.max_depth = max_depth if depth_unit != "samples" else None
                    
                    # Store coordinate usage
                    st.session_state.use_coords_for_distance = 'use_coords_for_distance' in locals() and use_coords_for_distance
                    st.session_state.coordinates_data = coordinates_data
                    
                    if not st.session_state.use_coords_for_distance:
                        st.session_state.distance_unit = distance_unit
                        st.session_state.total_distance = total_distance if distance_unit != "traces" else None
                    else:
                        st.session_state.distance_unit = "meters"  # Default for coordinates
                        st.session_state.total_distance = coordinates_data['distance'][-1] if coordinates_data else None
                    
                    # Store aspect ratio
                    st.session_state.aspect_mode = aspect_mode
                    if aspect_mode == "Manual" and 'aspect_ratio_float' in locals():
                        st.session_state.aspect_ratio = aspect_ratio_float
                    elif aspect_mode == "Realistic" and 'aspect_ratio_float' in locals():
                        st.session_state.aspect_ratio = aspect_ratio_float
                    else:
                        st.session_state.aspect_ratio = None
                    
                    # Store window parameters
                    st.session_state.use_custom_window = use_custom_window
                    if use_custom_window:
                        st.session_state.depth_min = depth_min if 'depth_min' in locals() else 0
                        st.session_state.depth_max = depth_max if 'depth_max' in locals() else max_depth
                        if not st.session_state.use_coords_for_distance:
                            st.session_state.distance_min = distance_min if 'distance_min' in locals() else 0
                            st.session_state.distance_max = distance_max if 'distance_max' in locals() else total_distance
                    
                    # Store multiple windows
                    st.session_state.multiple_windows = multiple_windows
                    if multiple_windows and use_custom_window and 'windows' in locals():
                        st.session_state.additional_windows = windows
                    
                    progress_bar.progress(100)
                    st.success("✅ Data processed successfully!")
                    
                else:
                    st.error("No radar data found in file")
                    
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.code(str(e))

# Display results if data is loaded
if st.session_state.data_loaded:
    # Create tabs
    tab_names = ["📊 Header Info", "📈 Full View", "🔍 Custom Window", "🗺️ Coordinate View", "📉 FFT Analysis", "🎛️ Gain Analysis", "💾 Export"]
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
            
            # Display near-surface correction info if applied
            if hasattr(st.session_state, 'near_surface_correction') and st.session_state.near_surface_correction:
                st.markdown("### Near-Surface Correction")
                st.markdown(f"**Type:** {st.session_state.correction_type}")
                st.markdown(f"**Depth:** {st.session_state.correction_depth} m")
        
        with col2:
            if st.session_state.header:
                st.markdown("### File Header")
                info_data = {
                    "System": st.session_state.header.get('system', 'Unknown'),
                    "Antenna Frequency": f"{st.session_state.header.get('ant_freq', 'N/A')} MHz",
                    "Samples per Trace": st.session_state.header.get('spt', 'N/A'),
                    "Number of Traces": st.session_state.header.get('ntraces', 'N/A'),
                    "Sampling Depth": f"{st.session_state.header.get('depth', 'N/A'):.2f} m"
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
        
        # Create figure
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
        ax1_full.set_title("Original Data - Full View")
        ax1_full.grid(True, alpha=0.3, linestyle='--')
        
        if show_colorbar:
            plt.colorbar(im1, ax=ax1_full, label='Amplitude')
        
        # Plot processed full view
        im2 = ax2_full.imshow(st.session_state.processed_array,
                             extent=[x_axis_full[0], x_axis_full[-1], y_axis_full[-1], y_axis_full[0]],
                             aspect=aspect_display, cmap=colormap,
                             vmin=vmin_plot, vmax=vmax_plot,
                             interpolation=interpolation)
        
        ax2_full.set_xlabel(x_label_full)
        ax2_full.set_ylabel(y_label_full)
        ax2_full.set_title(f"Processed ({gain_type} Gain) - Full View")
        ax2_full.grid(True, alpha=0.3, linestyle='--')
        
        if show_colorbar:
            plt.colorbar(im2, ax=ax2_full, label='Amplitude')
        
        plt.tight_layout()
        st.pyplot(fig_full)
        
        # Display aspect ratio info
        st.info(f"**Aspect Ratio:** {aspect_value} | **Plot Dimensions:** {st.session_state.processed_array.shape[1]} × {st.session_state.processed_array.shape[0]} | **Y:X Scale:** {y_axis_full[-1]/x_axis_full[-1]:.3f}")
    
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
                trace_in_window = st.slider(
                    "Select Trace in Window", 
                    0, window_data.shape[1]-1,
                    window_data.shape[1]//2,
                    key="window_trace"
                )
                
                # Get the actual trace index
                actual_trace_idx = window_info['dist_min_idx'] + trace_in_window
                
                # Plot trace
                fig_trace, ax_trace = plt.subplots(figsize=(10, 6))
                
                ax_trace.plot(y_axis_window, window_data[:, trace_in_window], 
                             'b-', linewidth=1.5, alpha=0.8)
                ax_trace.fill_between(y_axis_window, 0, window_data[:, trace_in_window], 
                                     alpha=0.3, color='blue')
                ax_trace.set_xlabel(y_label)
                ax_trace.set_ylabel("Amplitude")
                ax_trace.set_title(f"Trace {actual_trace_idx} in Window\n"
                                 f"Distance: {x_axis_window[trace_in_window]:.1f} {st.session_state.distance_unit}")
                ax_trace.grid(True, alpha=0.3)
                ax_trace.invert_xaxis()
                
                st.pyplot(fig_trace)
            
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
                
                # Plot depth slice
                fig_slice, ax_slice = plt.subplots(figsize=(10, 6))
                
                ax_slice.plot(x_axis_window, window_data[depth_slice_in_window, :], 
                             'r-', linewidth=1.5, alpha=0.8)
                ax_slice.fill_between(x_axis_window, 0, window_data[depth_slice_in_window, :], 
                                     alpha=0.3, color='red')
                ax_slice.set_xlabel(x_label)
                ax_slice.set_ylabel("Amplitude")
                ax_slice.set_title(f"Depth Slice at {actual_depth:.2f} {st.session_state.depth_unit}")
                ax_slice.grid(True, alpha=0.3)
                
                st.pyplot(fig_slice)
            
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
            # Display coordinate statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Profile Length", f"{st.session_state.interpolated_coords['distance'][-1]:.1f} m")
                st.metric("Elevation Change", 
                         f"{st.session_state.interpolated_coords['elevation'].max() - st.session_state.interpolated_coords['elevation'].min():.1f} m")
            
            with col2:
                st.metric("Easting Range", 
                         f"{st.session_state.interpolated_coords['easting'].ptp():.1f} m")
                st.metric("Northing Range", 
                         f"{st.session_state.interpolated_coords['northing'].ptp():.1f} m")
            
            with col3:
                avg_spacing = np.mean(np.diff(st.session_state.interpolated_coords['distance']))
                st.metric("Avg Trace Spacing", f"{avg_spacing:.2f} m")
                st.metric("Profile Bearing", 
                         f"{np.degrees(np.arctan2(st.session_state.interpolated_coords['northing'][-1] - st.session_state.interpolated_coords['northing'][0], 
                                                  st.session_state.interpolated_coords['easting'][-1] - st.session_state.interpolated_coords['easting'][0])):.1f}°")
            
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
            ax1.set_xlabel('Easting (m)')
            ax1.set_ylabel('Northing (m)')
            ax1.set_title('Plan View - Survey Line')
            ax1.grid(True, alpha=0.3)
            ax1.axis('equal')
            plt.colorbar(ax1.collections[0], ax=ax1, label='Distance along profile (m)')
            
            # 2. Elevation profile
            ax2.plot(st.session_state.interpolated_coords['distance'], 
                    st.session_state.interpolated_coords['elevation'], 
                    'g-', linewidth=2, alpha=0.8)
            ax2.fill_between(st.session_state.interpolated_coords['distance'],
                            st.session_state.interpolated_coords['elevation'].min(),
                            st.session_state.interpolated_coords['elevation'],
                            alpha=0.3, color='green')
            ax2.set_xlabel('Distance along profile (m)')
            ax2.set_ylabel('Elevation (m)')
            ax2.set_title('Elevation Profile')
            ax2.grid(True, alpha=0.3)
            
            # 3. 3D view of survey line
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
            ax3.set_xlabel('Easting (m)')
            ax3.set_ylabel('Northing (m)')
            ax3.set_zlabel('Elevation (m)')
            ax3.set_title('3D Survey Line')
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
            
            # Coordinate-based GPR with elevation adjustment
            st.subheader("Elevation-Adjusted GPR Display")
            
            # Calculate elevation-adjusted depth
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
            mesh = ax_elev.pcolormesh(X, Y_elev, st.session_state.processed_array,
                                     cmap='seismic', shading='auto', alpha=0.9)
            
            ax_elev.set_xlabel('Distance along profile (m)')
            ax_elev.set_ylabel('Elevation (m)')
            ax_elev.set_title('GPR Data with Elevation Adjustment')
            ax_elev.grid(True, alpha=0.2)
            plt.colorbar(mesh, ax=ax_elev, label='Amplitude')
            
            # Add topographic surface line
            ax_elev.plot(st.session_state.interpolated_coords['distance'],
                        st.session_state.interpolated_coords['elevation'],
                        'k-', linewidth=2, alpha=0.8, label='Surface')
            ax_elev.fill_between(st.session_state.interpolated_coords['distance'],
                                Y_elev.min(), st.session_state.interpolated_coords['elevation'],
                                alpha=0.1, color='gray')
            
            ax_elev.legend()
            ax_elev.set_ylim(Y_elev.min(), st.session_state.interpolated_coords['elevation'].max() + 5)
            
            plt.tight_layout()
            st.pyplot(fig_elev)
            
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
    
    with tabs[4]:  # FFT Analysis
        st.subheader("Frequency vs Amplitude Analysis (FFT)")
        
        # FFT analysis options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            trace_for_fft = st.slider("Select Trace for FFT", 
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
        
        # Create scaled depth axis
        y_axis_analysis, _, _, y_label_analysis = scale_axes(
            (n_samples, 1),
            st.session_state.depth_unit,
            st.session_state.max_depth if hasattr(st.session_state, 'max_depth') else None,
            "traces",
            None
        )[0:4]  # Only get first 4 values
        
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
    
    with tabs[6]:  # Export
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

# Initial state message
elif not dzt_file:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("""
        👈 **Upload a DZT file to begin processing**
        
        **New Coordinate Features:**
        1. **CSV Coordinate Import:** Upload CSV with Easting, Northing, Elevation
        2. **Automatic Interpolation:** Interpolates coordinates to match GPR traces
        3. **Aspect Ratio Control:** Adjust Y:X scale for realistic visualization
        4. **Coordinate-Based Visualization:** Plan view, elevation profile, 3D view
        
        **New Near-Surface Correction:**
        1. **Amplitude Normalization:** Reduce high amplitudes in 0-2.5m region
        2. **Multiple Methods:** Linear, Exponential, Gaussian, or Windowed normalization
        3. **Preserve Depth:** No need to adjust time zero excessively
        4. **Better Visualization:** Normalized amplitudes across entire profile
        
        **Coordinate CSV Format:**
        ```
        Easting, Northing, Elevation
        100.5, 200.3, 50.2
        101.0, 201.0, 50.1
        101.5, 201.7, 50.0
        ...
        ```
        
        **Aspect Ratio Examples:**
        - 1:1 (Square)
        - 1:10 (Standard GPR display)
        - 1:50 (Very stretched for deep investigations)
        - Auto (Matplotlib default)
        
        **Realistic Display:** Choose aspect ratios that match your survey conditions!
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "📡 <b>GPR Data Processor v5.1</b> | Coordinate Import & Near-Surface Correction | "
    "Built with Streamlit & readgssi"
    "</div>",
    unsafe_allow_html=True
)

