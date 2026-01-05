import pandas as pd
import mne
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATA_FILE = "data/S001/S001R01.edf"

def plot_channels(raw, title_suffix=""):
    n_seconds = 1 # Seconds to plot
    n_channels = 20 # Number of channels to plot
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
        title=f"EEG - {title_suffix} (First 20 channels)",
        xaxis_title="Time (s)",
        yaxis_title="Channels with spacing",
        showlegend=True,
        template="plotly_white"
    )
    fig.show()

def plot_psd_compare_plotly(raw_before, raw_after):
    # 1. Calculate spectra
    spec_before = raw_before.compute_psd(fmax=60)
    spec_after = raw_after.compute_psd(fmax=60)
    
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


def main(): 
	raw = mne.io.read_raw_edf(DATA_FILE, preload=True)
	raw.rename_channels(lambda x: x.strip('.'))
	raw.set_montage('standard_1005', match_case=False)

	# Filter
	raw_filtered = raw.copy().filter(1, 40)

	# Plot raw channels
	plot_channels(raw)

	# Plot filtered channels
	plot_channels(raw_filtered, "Filtered")

	# Plot PSD comparison
	plot_psd_compare_plotly(raw, raw_filtered)


if __name__ == "__main__":
	main()
	