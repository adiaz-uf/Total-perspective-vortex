import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.pyplot import matplotlib

def plot_channels(raw, title_suffix=""):
    n_seconds = 0.5 # Seconds to plot
    n_channels = 15 # Number of channels to plot
    spacing = 100e-6
    sfreq = raw.info['sfreq']
    
    data, times = raw[:n_channels, :int(sfreq * n_seconds)] 
    
    fig = go.Figure()
    
    for i in range(n_channels):
        # Remove mean (detrend) and add offset
        trace_data = data[i] - np.mean(data[i]) 
        fig.add_trace(go.Scatter(
            x=times, 
            y=trace_data + (i * spacing), 
            name=raw.ch_names[i],
            line=dict(width=1)
        ))
    
    fig.update_layout(
        title=f"EEG - {title_suffix} (First 15 channels)",
        xaxis_title="Time (s)",
        yaxis_title="Channels with spacing",
        showlegend=True,
        template="plotly_white"
    )
    fig.show()

def plot_psd_compare(raw_before, raw_after):
    # 1. Calculate spectra
    spec_before = raw_before.compute_psd(fmin=1, fmax=60)
    spec_after = raw_after.compute_psd(fmin=1, fmax=60)
    
    # Get data (averaging all channels for a clear view)
    psds_before, freqs_before = spec_before.get_data(return_freqs=True)
    psds_after, freqs_after = spec_after.get_data(return_freqs=True)
    
    # Average and conversion to dB (Logarithmic scale)
    psd_before_db = 10 * np.log10(psds_before.mean(axis=0))
    psd_after_db = 10 * np.log10(psds_after.mean(axis=0))

    # 2. Create Subplots
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Original Spectrum (Note noise and low freqs)", 
                                        "Filtered Spectrum (Focus on Alpha/Beta bands)"))

    # Top plot (Before)
    fig.add_trace(go.Scatter(x=freqs_before, y=psd_before_db, 
                             name="Raw", line=dict(color='firebrick')), row=1, col=1)

    # Bottom plot (After)
    fig.add_trace(go.Scatter(x=freqs_after, y=psd_after_db, 
                             name="Filtered", line=dict(color='royalblue')), row=2, col=1)

    fig.update_layout(height=800, title_text="Frequency Analysis for Feature Extraction",
                      template="plotly_white", showlegend=False)
    
    # Axis labels
    fig.update_yaxes(title_text="Power (dB)", row=1, col=1)
    fig.update_yaxes(title_text="Power (dB)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    
    fig.show()


def heatmap_psd(raw):
	raw.pick("eeg")

	# Compute PSD for alpha band (8-12 Hz)
	spectrum_alpha = raw.compute_psd(fmin=8, fmax=12) 

	# Plot topomap for alpha band
	spectrum_alpha.plot_topomap(bands={'Alpha': (8, 12)}, cmap='Spectral_r', contours=5, sensors=True)

	# Compute PSD for beta band (12-30 Hz)
	spectrum_beta = raw.compute_psd(fmin=12, fmax=30) 

	# Plot topomap for beta band
	spectrum_beta.plot_topomap(bands={'Beta': (12, 30)}, cmap='Spectral_r', contours=5, sensors=True)
