import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

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
    raw_eeg = raw.copy().pick("eeg")
    plt.rcParams.update({'font.size': 10})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ALPHA (8-12 Hz)
    spec_alpha = raw_eeg.compute_psd(fmin=8, fmax=12)
    data_alpha = spec_alpha.get_data() * 1e12
    vmax_alpha = np.nanpercentile(data_alpha, 85)
    
    if vmax_alpha <= 0: vmax_alpha = 1.0

    spec_alpha.plot_topomap(
        bands={'Alpha': (8, 12)},
        cmap='Spectral_r',
        sensors=True,
        show_names=True,
        axes=ax1,
        vlim=(0, vmax_alpha),
        mask_params=dict(markerfacecolor='black', markersize=4),
        show=False
    )

    # BETA (12-30 Hz)
    spec_beta = raw_eeg.compute_psd(fmin=12, fmax=30)
    data_beta = spec_beta.get_data() * 1e12
    vmax_beta = np.nanpercentile(data_beta, 85)
    
    if vmax_beta <= 0: vmax_beta = 1.0

    spec_beta.plot_topomap(
        bands={'Beta': (12, 30)},
        cmap='Spectral_r',
        sensors=True,
        show_names=True,
        axes=ax2,
        vlim=(0, vmax_beta),
        mask_params=dict(markerfacecolor='black', markersize=4),
        show=False
    )

    plt.tight_layout()
    output_path = "./V.1.1-process_data/Topomap_alpha_beta.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)