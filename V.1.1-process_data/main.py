import mne
from plot_data import plot_channels, plot_psd_compare, heatmap_psd
from extract_features import extract_features

DATA_FILE = "data/S001/S001R08.edf"

def print_info(features, events):
	print("Features shape:", features.shape)
	print("Features:", features[0])
	print("Events shape:", events.shape)
	print("Events:", events)

def main(): 
	raw = mne.io.read_raw_edf(DATA_FILE, preload=True)
	raw.rename_channels(lambda x: x.strip('.').upper())
	raw.set_montage('standard_1005', match_case=False)

	# Filter (quit electrical network and drift frequencies)
	raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=50.0)

	# Plot raw channels
	plot_channels(raw)

	# Plot filtered channels
	plot_channels(raw_filtered, "Filtered")

	# Plot PSD comparison
	plot_psd_compare(raw, raw_filtered)

	# Plot PSD heatmap
	heatmap_psd(raw)

	# Extract features
	features, events = extract_features(raw)
	#print_info(features, events)

if __name__ == "__main__":
	main()
	