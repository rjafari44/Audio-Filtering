import numpy as np # fundamental library for numerical operations such as arrays, FFT, noise generation, etc
import matplotlib # plotting library, I call it first for using the 'Agg' backend
matplotlib.use('Agg')  # 'Agg' backend is used to save plots to files without requiring a GUI. Must be set before importing pyplot to avoid GUI errors
import matplotlib.pyplot as plt # import sub library for plotting in matplotlib
from scipy import signal # signal processing library
import soundfile as sf # library to read and write .wav audo files, supports many formats

# load clean audio from samples directory
audio_file = 'samples/clean_audio.wav' 

# read the audio file into a NumPy array
# signal_clean contains audio sample amplitudes
# sample rate in Hz (samples per second)
signal_clean, sample_rate = sf.read(audio_file)

# some audio files are stereo (2 channels) so convert stereo to mono by averaging both channels
# this ensures the signal is 1D and compatible with our simple filter
if signal_clean.ndim == 2:
    signal_clean = signal_clean.mean(axis=1)

# print basic info about the audio file
print(f"Audio loaded: {len(signal_clean)} samples, {sample_rate} Hz sample rate")

# generate high frequency noise
noise_amp = 0.05                                       # controls loudness of background static, higher values make noise louder
noise = noise_amp * np.random.randn(len(signal_clean)) # np.random.randn(len(signal_clean)) creates random numbers with mean 0 and standard deviation of 1

signal_noisy = signal_clean + noise # add the noise to the clean audio
signal_noisy = np.clip(signal_noisy, -1.0, 1.0) # clip the noisy signal to [-1, 1] to ensure audio remains valid for .wav files & prevent distortion

# save the noisy audio as a wav file in samples/
sf.write('samples/noisy_audio.wav', signal_noisy, sample_rate)
print("Noisy audio saved as 'noisy_audio.wav'")

# create a new figure for plotting time domain plot of clean vs noisy signal
plt.figure(figsize=(12, 4))                             # width=12 inches, height=4 inches
plt.plot(signal_clean, label='Clean Signal')            # plot the clean signal
plt.plot(signal_noisy, label='Noisy Signal', alpha=0.7) # plot the noisy signal on the same axes, alpha is the tranparency (1 opaque, 0 transparent)
plt.title("Time-Domain: Clean vs Noisy")                # add plot title and axis labels
plt.xlabel("Sample Index")                              # x-axis: sample number (not time)
plt.ylabel("Amplitude")                                 # y-axis: audio amplitude
plt.legend()                                            # add legend to distinguish curves
plt.grid(True)                                          # add major grid lines for readability
plt.savefig("assets/time_domain_clean_vs_noisy.png")    # save the plot to assets/ as .png
plt.close()                                             # close the figure to free memory (even if no gui)
print("Time-domain plot (clean vs noisy) saved as PNG")

fft_noisy = np.fft.fft(signal_noisy) # compute the FFT of the noisy signal and convert from time domain to frequency domain
freqs = np.fft.fftfreq(len(signal_noisy), 1/sample_rate) # Compute frequency values for each FFT bin

# create a figure for plotting the frequency domain of noisy signal
# plot only the positive half of the frequency spectrum (real signals are symmetric)
plt.figure(figsize=(12, 4))
plt.plot(freqs[:len(freqs)//2], np.abs(fft_noisy)[:len(freqs)//2])
plt.title("Frequency-Domain: Noisy Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.savefig("assets/frequency_domain_noisy.png")
plt.close()
print("Frequency-domain plot (noisy) saved as PNG")

# applying moving average LTI low pass filter
N = 5             # filter length (number of samples to average), higher = more high frequency attenuation
h = np.ones(N)/N  # impulse response of moving-average filter; h[n] = 1/N for n = 0, 1, ..., N-1

signal_filtered = np.convolve(signal_noisy, h, mode='same') # apply the LTI filter using convolution, mode=same ensures output length equals input length
signal_filtered = np.clip(signal_filtered, -1.0, 1.0) # clip again to [-1, 1] after filtering

# save the filtered audio as a wav file in samples/
sf.write('samples/filtered_audio.wav', signal_filtered, sample_rate)
print("Filtered audio saved as 'filtered_audio.wav'")

# create a figure for plotting the time domain of clean vs filtered signal
plt.figure(figsize=(12, 4))
plt.plot(signal_clean, label='Clean Signal')
plt.plot(signal_filtered, label='Filtered Signal', alpha=0.7)
plt.title("Time-Domain: Clean vs Filtered")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.savefig("assets/time_domain_clean_vs_filtered.png")
plt.close()
print("Time-domain plot (clean vs filtered) saved as PNG")

fft_filtered = np.fft.fft(signal_filtered) # FFT of filtered signal

# create a figure for plotting the frequency domain of filtered signal
plt.figure(figsize=(12, 4))
plt.plot(freqs[:len(freqs)//2], np.abs(fft_filtered)[:len(freqs)//2])
plt.title("Frequency-Domain: Filtered Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)
plt.savefig("assets/frequency_domain_filtered.png")
plt.close()
print("Frequency-domain plot (filtered) saved as PNG")

# done
print("All processing complete.")
