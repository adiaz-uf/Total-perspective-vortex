import mne 
import numpy as np

def prepare_epochs(raw):
    # Extract the events from the internal annotations
    events, event_id = mne.events_from_annotations(raw)

    if events[0, 0] == 0:
        # Add 0.2 seconds of silence at the beginning of the raw data
        
        raw.padding = 0.2
        print("!! Detectado evento en t=0. Aplicando padding para evitar descartes.")
        # Shift events by 0.2 seconds
        events[:, 0] += int(0.2 * raw.info['sfreq'])

    mapping = {}

    if 'T0' in event_id: mapping['T0'] = event_id['T0']
    if 'T1' in event_id: mapping['T1'] = event_id['T1']
    if 'T2' in event_id: mapping['T2'] = event_id['T2']
    
    # Define the time window (2 seconds)
    tmin, tmax = -0.2, 1.8

    # Create the Epochs
    # Turn (channels, time) data into (events, channels, time)
    epochs = mne.Epochs(
        raw, 
        events, 
        event_id=mapping, 
        tmin=tmin,         
        tmax=tmax,            
        proj=True, 
        picks='eeg', 
        baseline=(None, 0), 
        preload=True,
        on_missing='warn'
    )

    if len(epochs) == 0:
        print("WARNING: Events found but dropped (possibly at t=0). Retrying without pre-stimulus...")
        epochs = mne.Epochs(raw, events, event_id=mapping, tmin=0, tmax=2.0, 
                            baseline=None, preload=True, on_missing='warn')

    print(f"Created {len(epochs)} data chunks for classification.")
    return epochs


def extract_features(raw, method='fourier'):
    target_channels = ['FC3', 'FCZ', 'FC4', 'C3', 'CZ', 'C4', 'CP3', 'CPZ', 'CP4']
    raw_picks = raw.copy().pick_channels(target_channels)

    # Slice the data into epochs
    epochs = prepare_epochs(raw_picks)

    if method == 'wavelet':
        freqs = np.arange(8, 31, 1)
        n_cycles = freqs / 2.

        # Wavelet Transform (Continuous Wavelet Transform)
        tfr = mne.time_frequency.tfr_morlet(
            epochs, freqs=freqs, 
            n_cycles=n_cycles, 
            return_itc=False,
            average=False
        )
        
        features = np.mean(tfr.data, axis=-1)

    else:
        # Transform to frequency domain using Fourier Transform
        features = epochs.compute_psd(fmin=8, fmax=30).get_data()

    # Convert to dB
    features = 10 * np.log10(features)
    return features, epochs.events[:, -1]