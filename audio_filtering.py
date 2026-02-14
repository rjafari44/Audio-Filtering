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

signal_noisy = signal_clean + noise             # add the noise to the clean audio
signal_noisy = np.clip(signal_noisy, -1.0, 1.0) # clip the noisy signal to [-1, 1] to ensure audio remains valid for .wav files & prevent distortion

# save the noisy audio as a wav file in samples/
sf.write('samples/noisy_audio.wav', signal_noisy, sample_rate)
print("Noisy audio saved as 'noisy_audio.wav'")

# =============== TIME-DOMAIN PLOT: CLEAN VS NOISY ===============
# create a new figure for plotting time-domain plot of clean vs noisy signal (side-by-side)
plt.figure(figsize=(14, 4))  # create a new figure window with specified size

# left subplot: clean signal
plt.subplot(1, 2, 1)  # create 1x2 grid, select first subplot
plt.plot(signal_clean)  # plot clean signal waveform
plt.title("Time-Domain: Clean Signal")  # set title of subplot
plt.xlabel("Sample Index (Hz)")  # label x-axis
plt.ylabel("Amplitude")  # label y-axis
plt.ylim(-1.0, 1.0)  # set y-axis limits for consistent scaling
plt.grid(True)  # enable grid lines

# right subplot: noisy signal
plt.subplot(1, 2, 2)  # select second subplot
plt.plot(signal_noisy)  # plot noisy signal waveform
plt.title("Time-Domain: Noisy Signal")  # set title
plt.xlabel("Sample Index (Hz)")  # label x-axis
plt.ylim(-1.0, 1.0)  # keep same amplitude limits
plt.grid(True)  # enable grid

plt.tight_layout()  # automatically adjust spacing between subplots
plt.savefig("assets/time_domain_clean_vs_noisy.png")  # save figure as PNG file
plt.close()  # close figure to free memory
print("Time-domain side by side plot (clean vs noisy) saved as PNG")
# ===========================================================================================

# compute FFTs of clean and noisy signals
fft_clean = np.fft.fft(signal_clean)
fft_noisy = np.fft.fft(signal_noisy)

freqs = np.fft.fftfreq(len(signal_noisy), 1/sample_rate) # Compute frequency values for each FFT bin

# =============== FREQUENCY-DOMAIN PLOT: CLEAN VS NOISY ===============
# create a figure for plotting the frequency-domain of clean vs noisy signal (side-by-side)
plt.figure(figsize=(14, 4))  # create new figure for frequency plots

# left subplot: clean spectrum in dB
plt.subplot(1, 2, 1)  # select first subplot
plt.plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft_clean)[:len(freqs)//2] + 1e-12))  # magnitude in dB
plt.title("Frequency-Domain: Clean Signal")  # set title
plt.xlabel("Frequency (Hz)")  # label x-axis in Hz
plt.ylabel("Magnitude (dB)")  # label magnitude axis
plt.grid(True)  # enable grid

# right subplot: noisy spectrum in dB
plt.subplot(1, 2, 2)  # select second subplot
plt.plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft_noisy)[:len(freqs)//2] + 1e-12))  # plot noisy magnitude spectrum in dB
plt.title("Frequency-Domain: Noisy Signal")  # set title
plt.xlabel("Frequency (Hz)")  # label x-axis
plt.grid(True)  # enable grid

plt.tight_layout()  # adjust subplot spacing
plt.savefig("assets/frequency_domain_clean_vs_noisy.png")  # save frequency comparison
plt.close()  # close figure
print("Frequency-domain side by side plot (clean vs noisy) saved as PNG")
# ===========================================================================================

# applying moving average LTI low-pass filter
N = 5            # filter length (number of samples to average), higher = more high frequency attenuation
h = np.ones(N)/N # impulse response of moving-average filter; h[n] = 1/N for n = 0, 1, ..., N-1

signal_filtered = np.convolve(signal_noisy, h, mode='same')
signal_filtered = np.clip(signal_filtered, -1.0, 1.0)

# save the filtered audio as a wav file in samples/
sf.write('samples/filtered_audio.wav', signal_filtered, sample_rate)
print("Filtered audio saved as 'filtered_audio.wav'")

# =============== TIME-DOMAIN PLOT: CLEAN VS FILTERED ===============
# create a new figure for plotting time-domain plot of clean vs filtered signal (side-by-side)
plt.figure(figsize=(14, 4))  # create new figure

# left subplot: clean signal
plt.subplot(1, 2, 1)  # select first subplot
plt.plot(signal_clean)  # plot clean waveform
plt.title("Time-Domain: Clean Signal")  # set title
plt.xlabel("Sample Index (Hz)")  # label x-axis
plt.ylabel("Amplitude")  # label y-axis
plt.ylim(-1.0, 1.0)  # consistent amplitude scaling
plt.grid(True)  # enable grid

# right subplot: filtered signal
plt.subplot(1, 2, 2)  # select second subplot
plt.plot(signal_filtered)  # plot filtered waveform
plt.title("Time-Domain: Filtered Signal")  # set title
plt.xlabel("Sample Index (Hz)")  # label x-axis
plt.ylim(-1.0, 1.0)  # same y-axis limits
plt.grid(True)  # enable grid

plt.tight_layout()  # adjust spacing
plt.savefig("assets/time_domain_clean_vs_filtered.png")  # save figure
plt.close()  # close figure
print("Time-domain side by side plot (clean vs filtered) saved as PNG")
# ===========================================================================================

fft_filtered = np.fft.fft(signal_filtered) # FFT of filtered signal

# =============== FREQUENCY-DOMAIN PLOT: CLEAN VS FILTERED ===============
# create a figure for plotting the frequency-domain of clean vs filtered signal (side-by-side)
plt.figure(figsize=(14, 4))  # create new figure

# left subplot: clean spectrum in dB
plt.subplot(1, 2, 1)  # select first subplot
plt.plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft_clean)[:len(freqs)//2] + 1e-12))  # magnitude in dB
plt.title("Frequency-Domain: Clean Signal")  # set title
plt.xlabel("Frequency (Hz)")  # label x-axis
plt.ylabel("Magnitude (dB)")  # label magnitude axis
plt.grid(True)  # enable grid

# right subplot: filtered spectrum in dB
plt.subplot(1, 2, 2)  # select second subplot
plt.plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft_filtered)[:len(freqs)//2] + 1e-12))  # plot filtered magnitude spectrum in dB
plt.title("Frequency-Domain: Filtered Signal")  # set title
plt.xlabel("Frequency (Hz)")  # label x-axis
plt.grid(True)  # enable grid

plt.tight_layout()  # adjust subplot spacing
plt.savefig("assets/frequency_domain_clean_vs_filtered.png")  # save frequency comparison
plt.close()  # close figure
print("Frequency-domain side by side plot (clean vs filtered) saved as PNG")
# ===========================================================================================

# done
print("All processing complete!")
