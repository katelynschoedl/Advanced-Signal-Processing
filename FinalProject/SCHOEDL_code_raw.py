# %% [markdown]
# Sources for this project:
# 
# FinalProject\DAS4Whales-DAS4Whales-3cfc942 - "The DAS python package"
# FinalProject\Ocean-Data-Lab-Goestchel_JASA_2025-6dcb6db - "The author's code"

# %%
### Import the DAS4Whales module and dependencies from GitHub or local download folder

# %pip install das4whales

import das4whales as dw
import h5py
import scipy.signal as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# ## DATA 
# From the author's code.
# 
# The data used in this code comes from the 2021 OOI RCA dataset. 
# Experiment information can be found on the [OOI website](https://oceanobservatories.org/pi-instrument/rapid-a-community-test-of-distributed-acoustic-sensing-on-the-ocean-observatories-initiative-regional-cabled-array/).
# 
# >Wilcock, W., & Ocean Observatories Initiative. (2023). Rapid: A Community Test of Distributed Acoustic Sensing on the Ocean Observatories Initiative Regional Cabled Array [Data set]. Ocean Observatories Initiative. https://doi.org/10.58046/5J60-FJ89
# 
# 
# ### Download some DAS data
# If the file stored at the chosen `url` already exists it will work, otherwise we download a `.hdf5` file from the OOI DAS experiment (Wilcock & OOI, 2023). The file is ~850 MB so the download might take a few seconds.
# 
# The dataset of this example is constituted of 60s time series along 66 km of cable
# 
# ## Warning: high RAM usage
# The scripts are memory intensive, and at least 32GB of RAM is recommended. Otherwise, the number of channels:
# 
# -`selected_channels_m_north = [12000, 66000, 5]`
# 
# -`selected_channels_m_north = [12000, 66000, 5]`
# 
# This can be reduced by increasing the channel spacing in meters. The channel spacing of the raw data is 2m, the channel spacing used in these scripts is 4m and can be changed by increasing the last value in the lists above (e.g. 5). 

# %% [markdown]
# Recall: when it comes to downsampling a signal, what kind of pre-processing must be applied prior to decimation?
# A low-pass filter should ALWAYS be applied prior to decimation to avoid aliasing of all the physical signals higher than the Nyquist frequency.
# 
# Choosing every 2nd channel is decimation along the cable, which is why the anti-alias question applies to channels, not time. The gauge length’s averaging helps a little there.

# %%
# The dataset of this example is constituted of 60s time series along the north and south cables

######################
## North cable plots
######################
url_north = ['http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
                'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
                'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5'] 

selected_channels_m_north = [12000, 66000, 10]  # list of values in meters corresponding to the starting,
                                                # ending and step wanted channels along the FO Cable
                                                # selected_channels_m = [ChannelStart_m, ChannelStop_m, ChannelStep_m]
                                                # in meters


# Download North DAS data
url = url_north[0]
filepath, filename = dw.data_handle.dl_file(url)


######################
## Read HDF5 files and access metadata
######################

# Get the acquisition parameters for the data folder
with h5py.File(filepath, 'r') as f: print(list(f['Acquisition'].attrs.keys()))
# ['AcquisitionId', 'GaugeLength', 'GaugeLengthUnit', 'MaximumFrequency', 'MinimumFrequency', 'NumberOfLoci', 'PulseRate', 'PulseWidth', 'PulseWidthUnit', 'SpatialSamplingInterval', 'SpatialSamplingIntervalUnit', 'StartLocusIndex', 'TriggeredMeasurement', 'VendorCode', 'schemaVersion', 'uuid']
with h5py.File(filepath, 'r') as f: print(list(f['Acquisition/Raw[0]'].attrs.keys()))
# ['NumberOfLoci', 'OutputDataRate', 'RawDataUnit', 'RawDescription', 'StartLocusIndex', 'uuid']
with h5py.File(filepath, 'r') as f: print(list(f['Acquisition/Custom'].attrs.keys()))
# ['ADC Clock (MHz)', 'Acquisition Start (CSU)', 'CSU Resolution (ns)', 'Carrier Separation (CSU)', 'CustomSchemaVersion', 'Data Compromised', 'Data Width (Bits)', 'Decimation Factor', 'FPGA Data Sub Type', 'FPGA Data Type', 'FPGA Drawing Number', 'FPGA Image in Set', 'FPGA Version', 'Fibre Refractive Index', 'Firmware Version', 'GPS Enabled', 'GPS Sync Guaranteed', 'Gauge Length (CSU)', 'Header Schema', 'Input Channel Pitch (CSU)', 'Input Mux. Position', 'Laser Wavelength (nm)', 'Num Carriers', 'Num Elements Per Channel', 'Num Output Channels', 'Output Channel Pitch (CSU)', 'Output Channel Start (CSU)', 'Phase Fractional Width (Bits)', 'Ping Period (CSU)', 'Product Drawing Number', 'Pulse Layout', 'Pulse Width (CSU)', 'Synthetic Gauge Length (CSU)', 'Unit Serial Number', 'User Data Type', 'Variant']
# Can also see these parameters in h5 viewer 

metadata = dw.data_handle.get_acquisition_parameters(filepath, interrogator='optasense')
fs, dx, nx, ns, gauge_length, scale_factor = metadata["fs"], metadata["dx"], metadata["nx"], metadata["ns"], metadata["GL"], metadata["scale_factor"]

    #metadata : dict
    #    dictionary filled with metadata, key's breakdown:\n
    #    fs: the sampling frequency (Hz)\n
    #    dx: interval between two virtual sensing points also called channel spacing (m)\n
    #    nx: the number of spatial samples also called channels\n
    #    ns: the number of time samples\n
    #    n: refractive index of the fiber\n
    #    GL: the gauge length (m)\n
    #    scale_factor: the value to convert DAS data from unwrapped optical phase (rad) to strain 
    #       scale_factor = (2 * np.pi) / 2 ** 16 * (1550.12 * 1e-9) / (0.78 * 4 * np.pi * n * GL)

print(f'Sampling frequency: {metadata["fs"]} Hz')
print(f'Channel spacing: {metadata["dx"]} m')
print(f'Gauge length: {metadata["GL"]} m')
print(f'File duration: {metadata["ns"] / metadata["fs"]} s')
print(f'Cable max distance: {metadata["nx"] * metadata["dx"]/1e3:.1f} km')
print(f'Number of channels: {metadata["nx"]}')
print(f'Number of time samples: {metadata["ns"]}')

### Select the desired channels and channel interval

selected_channels = [int(m // dx) for m in selected_channels_m_north]  
                                                        # list of values in channel number (spatial sample) corresponding to the starting, ending and step wanted
                                                        # channels along the FO Cable
                                                        # selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
                                                        # numbers

print('Begin channel #:', selected_channels[0], 
', End channel #: ',selected_channels[1], 
', step: ',selected_channels[2], 
'equivalent to ',selected_channels[2]*dx,' m')

######################
### Load raw DAS data using the pre-defined selected channels. 
######################

tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)
trf = dw.dsp.bp_filt(tr, fs, 14, 28) # bandpass filter the data between 14 and 28 Hz, which is the frequency range of interest for this dataset

cable_name = 'North' 

#######
# view the raw data in the time domain
#######

# What are the minimum and maximum values of the raw strain data?
#print(f'min: {tr.min():.3e}   max: {tr.max():.3e}   (strain)')
#print(f'min: {tr.min()*1e9:.1f}   max: {tr.max()*1e9:.1f}   (x$10^{-9}$ strain)')

# What does most of the data look like / percentiles of the absolute value of the raw strain data?
# for p in (1, 50, 95, 97, 99, 99.9):
    #print(f'{p:5}th percentile of |tr|: {np.percentile(np.abs(tr), p)*1e9:.1f} nε')

    #1th percentile of |tr|: 0.8 nε
    #50th percentile of |tr|: 66.4 nε
    #95th percentile of |tr|: 576.3 nε
    #97th percentile of |tr|: 684.2 nε
    #99th percentile of |tr|: 929.7 nε
    #99.9th percentile of |tr|: 1591.9 nε


# What the loader already does (load_das_data):
# Reads the metadata (get_acquisition_parameters), including scale_factor, which depends on n and GL.
# Converts metres to channel indices (m // dx) and slices start:stop:step from disk, so only those channels are read.
# Casts to float64 and removes each channel’s mean (raw2strain).
# Multiplies by scale_factor to get strain.

# Returns tr (raw strain, signed, dimensionless), time vector (seconds) and dist vector (metres from the interrogator).
# and fileBeginTimeUTC is the start time of the file in UTC.

tr_ne = tr * 1e9                                  # nanostrain
lim = np.percentile(np.abs(tr_ne), 99.9)          # display limit only

plt.figure(figsize=(12, 10))
plt.imshow(tr_ne, aspect='auto', origin='lower', cmap='RdBu', vmin=-lim, vmax=lim,
           extent=[0, tr.shape[1]/fs, dist[0]/1e3, dist[-1]/1e3])
plt.xlabel('Time (s)')
plt.ylabel('Distance along cable (km)')
plt.colorbar(label='Strain [-] x$10^{-9}$')
plt.title('Raw Nanostrain, North cable')
plt.show()

# The plot from the das4whales module is a more compact way to do the same thing:
dw.plot.plot_rawdata(tr, time, dist)
# Notice that the strain is limited to +/- 500 nanostrain


# What are the minimum and maximum values of the filtered strain data?
print('The maximum and minimum values of the filtered strain data are:')
print(f'min: {trf.min():.3e}   max: {trf.max():.3e}   (strain)')
print(f'min: {trf.min()*1e9:.1f}   max: {trf.max()*1e9:.1f}   (nanostrain)')


# Try the raw data plotter with the filtered data
dw.plot.plot_rawdata(trf, time, dist)
# The strain is limited to +/- 500 nanostrain

# Lets make a plot closer to the min and max strain
trf_ne = trf * 1e9                                  # nanostrain      
plt.figure(figsize=(12, 10))
plt.imshow(trf_ne, aspect='auto', origin='lower', cmap='RdBu', vmin=-12, vmax=12,
           extent=[0, tr.shape[1]/fs, dist[0]/1e3, dist[-1]/1e3])
plt.xlabel('Time (s)')
plt.ylabel('Distance along cable (km)')
plt.colorbar(label='Strain (nε)')
plt.title('Bandpass filtered raw strain data, North cable')
plt.show()

# Shows nothing... limit it even more

# NOW we can see. 

plt.figure(figsize=(12, 10))
plt.imshow(trf_ne, aspect='auto', origin='lower', cmap='RdBu', vmin=-.5, vmax=.5,
           extent=[0, tr.shape[1]/fs, dist[0]/1e3, dist[-1]/1e3])
plt.xlabel('Time (s)')
plt.ylabel('Distance along cable (km)')
plt.colorbar(label='Strain (nε)')
plt.title('Bandpass filtered raw strain data, North cable')
plt.show()

# how to know the right frequency range to filter?  
# Use a periodogram to see the frequency content of the raw data.  
# The Welch method is a good way to do this.

#######
# view the frequency content in the raw data and in the filtered data
#######

# FROM WELCH LECTURE
# In lecture the input was one long 1D signal. Here tr is 2D: one row per channel, one column per time sample.
# axis=1 means "work along time", so every channel gets its own periodogram.
# nperseg=4096 cuts each channel's 60 s into pieces of 4096 samples (about 20 s), computes a spectrum
# for each piece and averages them. The frequency step is fs/4096 = 0.05 Hz.
frequencies, welch_psd = sp.welch(tr, fs, nperseg=4096, axis=1)          # raw data      -> (channels, frequencies), strain²/Hz
# detrend=False: do not subtract each piece's average first. 
frequencies, welch_psd_filt = sp.welch(trf, fs, nperseg=4096, axis=1, detrend=False)    # filtered data -> same shape

# Average over all channels: one curve for the whole selection of cable.
mean_psd = welch_psd.mean(axis=0)
mean_psd_filt = welch_psd_filt.mean(axis=0)

# Plain axes: the ocean swell is so much larger than everything else that it is a thin spike
# on the left edge, and the rest looks like zero.
plt.figure(figsize=(16, 7))
plt.plot(frequencies, mean_psd, 'k', lw=0.75, label='raw strain')
plt.plot(frequencies, mean_psd_filt, 'r', lw=0.75, label='bandpass filtered 14-28 Hz')
plt.xlim(0, 100)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (strain²/Hz)')
plt.title(f'Welch Periodogram on plain axes, {cable_name} cable')
plt.legend()
plt.grid()
plt.show()


# logarithmic axis
plt.figure(figsize=(12, 10))
plt.loglog(frequencies[1:], mean_psd[1:], 'k', lw=0.75, label='raw strain')
plt.loglog(frequencies[1:], mean_psd_filt[1:], 'r', lw=0.75, label='bandpass filtered 14-28 Hz')
plt.axvspan(0.05, 0.15, color='tab:blue', alpha=0.15, label='ocean swell (around 0.1 Hz)')
plt.axvspan(0.15, 3, color='tab:green', alpha=0.15, label='microseism (0.15-3 Hz)')
plt.axvspan(14, 28, color='tab:orange', alpha=0.3, label='fin whale band (14-28 Hz)')
plt.xlim(0.04, 100)
plt.ylim(1e-26, 1e-11)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power/Frequency (strain²/Hz)')
plt.title(f'Welch Periodogram, average over all channels, {cable_name} cable')
plt.legend()
plt.grid()
plt.show()

# Power, variance and standard deviation:
# Power is the average of the signal squared. 
# Variance is the power after removing the average value.
# Standard deviation is the square root of the variance. 
# 
# The loader already removed each channel's average,
# so power and variance come out the same.
print('RAW strain, all channels')
print(f'  power               np.mean(tr**2) = {np.mean(tr**2):.3e} strain²')
print(f'  variance            np.var(tr)     = {np.var(tr):.3e} strain²')
print(f'  standard deviation  np.std(tr)     = {np.std(tr)*1e9:.1f} nε')
print(f'  area under the raw periodogram     = {np.trapezoid(mean_psd, frequencies):.3e} strain²')
print('BANDPASS FILTERED strain, all channels')
print(f'  power               np.mean(trf**2) = {np.mean(trf**2):.3e} strain²')
print(f'  variance            np.var(trf)     = {np.var(trf):.3e} strain²')
print(f'  standard deviation  np.std(trf)     = {np.std(trf)*1e9:.4f} nε')
print(f'  area under the filtered periodogram = {np.trapezoid(mean_psd_filt, frequencies):.3e} strain²')

# Signal-to-noise ratio from the lecture, 
# SNR = 10*log10(P / sigma²), on the channel nearest 28.098 km.
# P is the power of 3 s that contain a whale call. 
# sigma² is the variance of 3 s with no call.
ch = np.argmin(np.abs(dist - 28098))                 # row of that channel
call = slice(int(5.5*fs), int(8.5*fs))               # columns for 5.5 s to 8.5 s
quiet = slice(int(14*fs), int(17*fs))                # columns for 14 s to 17 s
P_raw = np.mean(tr[ch, call]**2)
sigma2_raw = np.var(tr[ch, quiet])
P_filt = np.mean(trf[ch, call]**2)
sigma2_filt = np.var(trf[ch, quiet])
print(f'Channel at {dist[ch]/1e3:.1f} km, call against quiet')
print(f'  raw:      SNR = {10*np.log10(P_raw/sigma2_raw):.1f} dB')
print(f'  filtered: SNR = {10*np.log10(P_filt/sigma2_filt):.1f} dB')


# Periodograms before averaging, one row per channel, so frequency content is shown along the cable.
# 10*log10 converts power to decibels. [::10] keeps every 10th channel 
plt.figure(figsize=(12, 10))
plt.pcolormesh(frequencies[1:], dist[::10]/1e3, 10*np.log10(welch_psd[::10, 1:]), cmap='viridis', vmin=-230, vmax=-120, shading='auto')
plt.xscale('log')
plt.xlim(0.04, 100)
plt.axvline(14, color='w', ls='--')
plt.axvline(28, color='w', ls='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Distance along cable (km)')
plt.colorbar(label='Power/Frequency (dB re strain²/Hz)')
plt.title(f'Welch Periodogram of every channel, raw strain, {cable_name} cable')
plt.show()

# 10-35 Hz filtered data.
plt.figure(figsize=(12, 10))
plt.pcolormesh(frequencies, dist[::10]/1e3, 10*np.log10(welch_psd_filt[::10]), cmap='viridis', vmin=-230, vmax=-190, shading='auto')
plt.xlim(10, 35)
plt.axvline(14, color='w', ls='--')
plt.axvline(28, color='w', ls='--')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Distance along cable (km)')
plt.colorbar(label='Power/Frequency (dB re strain²/Hz)')
plt.title(f'Welch Periodogram of every channel, bandpass filtered strain, {cable_name} cable')
plt.show()

# How the power is distributed across frequencies. np.trapezoid adds up the area under the average curve between two frequencies.
total_power = np.trapezoid(mean_psd, frequencies)
below_1Hz = frequencies <= 1
fin_band = (frequencies >= 14) & (frequencies <= 28)
print(f'Share of raw power below 1 Hz:  {np.trapezoid(mean_psd[below_1Hz], frequencies[below_1Hz]) / total_power:.4%}')
print(f'Share of raw power in 14-28 Hz: {np.trapezoid(mean_psd[fin_band], frequencies[fin_band]) / total_power:.6%}')


#######
# view the bandpass filtered data in the time domain
#######

# What does most of the data look like / percentiles of the absolute value of the raw strain data?
# for p in (1, 50, 95, 97, 99, 99.9):
# print(f'{p:5}th percentile of |tr|: {np.percentile(np.abs(tr), p)*1e9:.1f} nε')

plt.figure(figsize=(12, 10))
plt.imshow(trf*1e9, aspect='auto', origin='lower', cmap='RdBu', vmin=-.4, vmax=.4,
           extent=[0, tr.shape[1]/fs, dist[0]/1e3, dist[-1]/1e3])
plt.xlabel('Time (s)')
plt.ylabel('Distance along cable (km)')
plt.colorbar(label='Strain [-] x$10^{-9}$')
plt.title('Bandpass (Butterworth) Filtered Nanostrain, North cable')
plt.show()


# The das4whales module uses a Hilbert transform to get the envelope of the bandpass filtered data:
envelope = sp.hilbert(trf, axis=1)

plt.figure(figsize=(12, 10))
plt.imshow(np.abs(envelope)*1e9, aspect='auto', origin='lower', cmap='RdBu', vmin=-.4, vmax=.4,
           extent=[0, tr.shape[1]/fs, dist[0]/1e3, dist[-1]/1e3])
plt.xlabel('Time (s)')
plt.ylabel('Distance along cable (km)')
plt.colorbar(label='Strain [-] x$10^{-9}$')
plt.title('Bandpass (Butterworth) Filtered Nanostrain, Envelope (Hilbert) Magnitude (abs), North cable')
plt.show()


# The plot from the das4whales module is a more compact way to do the same thing:
dw.plot.plot_tx(sp.hilbert(trf, axis=1), time, dist, f'Bandpass filter, {cable_name}', v_max=0.4, fig_size=(12, 10))


#######
# f-k analysis of the bandpass filtered data, the same way as in TP2
#######

# TP2 steps: Fourier transform over time, then over space (here the space samples are the channels).
channel_step = selected_channels[2] * dx                                  # 8.17 m between the channels kept
freqs = np.fft.fftfreq(trf.shape[1], d=1/fs)                              # frequency axis (Hz)
wavenumbers = 2 * np.pi * np.fft.fftfreq(trf.shape[0], d=channel_step)    # wavenumber axis (rad/m), as in TP2

trf_fft_time = np.fft.fft(trf, axis=1)                                    # Fourier transform over time
trf_fft = np.fft.fft(trf_fft_time, axis=0)                                # then over space
fk_mag = np.abs(trf_fft)                                                  # magnitude spectrum
del trf_fft_time, trf_fft                                                 # free about 2.5 GB of memory

k_shifted = np.fft.fftshift(wavenumbers)                                  # wavenumber 0 in the middle
fk_shifted = np.fft.fftshift(fk_mag, axes=0)
del fk_mag

# Keep 0-40 Hz only. imshow instead of TP2's pcolormesh: with 6,600 channels pcolormesh is very slow.
keep = (freqs >= 0) & (freqs <= 40)
fk_band = fk_shifted[:, keep]
plt.figure(figsize=(12, 10))
plt.imshow(fk_band, aspect='auto', origin='lower', cmap='turbo', vmin=0, vmax=np.percentile(fk_band, 99.9),
           extent=[0, 40, k_shifted[0], k_shifted[-1]])

# Speed lines: a wave moving along the cable at speed c lies on wavenumber = 2*pi*frequency / c.
# Positive and negative wavenumbers are the two directions along the cable.
f_line = np.linspace(0, 40, 100)
for c, colour in ((1400, 'tab:pink'), (5000, 'white')):
    plt.plot(f_line,  2*np.pi*f_line/c, '--', color=colour, lw=1.5, label=f'{c} m/s')
    plt.plot(f_line, -2*np.pi*f_line/c, '--', color=colour, lw=1.5)
plt.axvline(14, color='tab:orange', ls='--', lw=1.5, label='14 Hz')
plt.axvline(28, color='tab:red', ls='--', lw=1.5, label='28 Hz')
plt.ylim(-0.2, 0.2)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Wavenumber (rad/m)')
plt.colorbar(label='Magnitude (arbitrary units)')
plt.title(f'f-k analysis of the bandpass filtered data, {cable_name} cable')
plt.legend(loc='upper left')
plt.show()


#######
# basic f-k filtering
#######

# From DAS4Whales 
# Traditional route: Butterworth bandpass, then a separate speed-only f-k filter
trf_2step = dw.dsp.fk_filt(trf, 1, fs, selected_channels[2], dx, 1400, 5000)
dw.plot.plot_tx(sp.hilbert(trf_2step, axis=1), time, dist, f'Butterworth + separate f-k, {cable_name}', v_max=0.4, fig_size=(12, 10))

# SNR map the paper's way, to compare with the hybrid SNR map further down
SNR_2step = dw.dsp.snr_tr_array(trf_2step)
dw.plot.snr_matrix(SNR_2step, time, dist, 20, f'SNR estimation, Butterworth + separate f-k, {cable_name}')


#######
# hybrid f-k filtered data 
#######

# Create the f-k filter 
fk_params = {
'c_min': 1400.,
'c_max': 5000.,
'fmin': 14.,
'fmax': 28.
}

fk_filter = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, fk_params=fk_params, display_filter=True)

# Print the compression ratio given by the sparse matrix usage
dw.tools.disp_comprate(fk_filter)

# Apply the f-k filter to the data, returns spatio-temporal strain matrix
trf_fk = dw.dsp.fk_filter_sparsefilt(tr, fk_filter, tapering=True)
dw.plot.plot_fk_domain(tr, fs, dx, selected_channels, fig_size=(9, 8.8), v_min=0, v_max=0.000025, fk_params=fk_params, ax_lims=[12, 30, 0, 0.025])

# Delete the raw data to free memory
del tr

dw.plot.plot_tx(sp.hilbert(trf_fk, axis=1), time, dist, f'Hybrid filter, {cable_name}', v_max=0.4, fig_size=(12, 10))

# Plot the SNR
SNR = dw.dsp.snr_tr_array(trf_fk)
dw.plot.snr_matrix(SNR, time, dist, 20, "SNR estimation, "+f"{cable_name}")


#######
# SNR the way it was computed in class, for each filtering stage
#######

# Lecture / TP1: SNR = 10*log10(P / sigma²)
#   P      = power of the signal: mean of the squared samples while the call is present
#   sigma² = variance of the noise: variance of the samples when there is no call
# Same channel (28.1 km) and windows (call 5.5-8.5 s, quiet 14-17 s) as in the power section above.
# The raw data was deleted above (del tr), so the raw value reuses P_raw and sigma2_raw from the power section.
print(f'Channel at {dist[ch]/1e3:.1f} km, call (5.5-8.5 s) against quiet (14-17 s)')
print(f'  {"raw strain":28s} SNR = {10*np.log10(P_raw/sigma2_raw):5.1f} dB')
for name, x in (('Butterworth bandpass', trf), ('Butterworth + separate f-k', trf_2step), ('hybrid f-k', trf_fk)):
    P = np.mean(x[ch, call]**2)
    sigma2 = np.var(x[ch, quiet])
    print(f'  {name:28s} SNR = {10*np.log10(P/sigma2):5.1f} dB')

# %% [markdown]
# ## NECESSARY FOR USING SEAFLOOR DAS DATA
# 
# The author's script begins the analysis at 12km. The script converts every value in the list from metres to channel numbers with int(value // dx), where dx or channel spacing is about 2.04 m per channel.
# 
# The signal strength depends on the angle between the wave and the local cable heading, and that heading comes from neighboring channel coordinates on the cable which is not just a straight line.
# 
# The signal from a fin whale would be 20 Hz - 30 Hz. 
# 
# And 5m is a spatial sampling step.
# 
# The data has two sampling rates:
# - Time: 200 Hz. This is the sampling frequency, and its Nyquist limit is 100 Hz. That is well above the 30 Hz call band.
# 
# - Space: one channel every 2.04 m, set by the interrogator. The 5 m step just picks every second channel, giving 4.08 m. Choosing a step cannot make the data finer than 2.04 m.
# 
# 
# ## Provided Files
# 
# \data\GMRT_OOI_RCA_Cables.grd
# - GMRT is the Global Multi-Resolution Topography compilation, and the file is clipped to the cable area: longitude −125.16 to −123.83, latitude 45.00 to 45.50, elevation −1680 m to +671 m. It gives the bathymetry, but the depths required to use the data are already in the cable CSVs.
# 
# 
# \data\north_DAS_multidoord.csv
# \data\south_DAS_multidoord.csv
# - The CSVs give each channel’s latitude, longitude and depth since the H5 files contain no positions.
# 
# The CSV columns are: 
# - chan_idx,lat,lon,depth,chan_m,utm_x,utm_y,x,y
# - raw channel, lat/long/depth cable coordiantes with depth being negative below sea level, distance from interrogator, utm_x and utm_y to account for the curvature of the earth, and x and y which are a different set of meters
# 
# The files start at channel index 942 (spacing is 2.04m per channel, so this is 1.9 km from the interrogator). Nothing closer to shore has coordinates. There are various reasons why the study may have excluded the data closer to shore. Namely, fin whale calls are not typically seen there.
# 
# The paper says "the northern repeater is located at a depth of 600 m, 65 km along the cable, while the southern repeater is at a depth of 1600 m, 95 km along the cable.”
# 

# %%


######################
## South cable plots
######################
url_south = [
        'http://piweb.ooirsn.uw.edu/das/data/Optasense/SouthCable/TransmitFiber/'\
        'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-01T16_09_15-0700/'\
        'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-04T015914Z.h5']         
        
selected_channels_m_south = [12000, 95000, 10]

# Download South DAS data
                # Download the DAS data

url = url_south[0]
filepath, filename = dw.data_handle.dl_file(url)

# Read HDF5 files and access metadata
# Get the acquisition parameters for the data folder
metadata = dw.data_handle.get_acquisition_parameters(filepath, interrogator='optasense')
fs, dx, nx, ns, gauge_length, scale_factor = metadata["fs"], metadata["dx"], metadata["nx"], metadata["ns"], metadata["GL"], metadata["scale_factor"]

print(f'Sampling frequency: {metadata["fs"]} Hz')
print(f'Channel spacing: {metadata["dx"]} m')
print(f'Gauge length: {metadata["GL"]} m')
print(f'File duration: {metadata["ns"] / metadata["fs"]} s')
print(f'Cable max distance: {metadata["nx"] * metadata["dx"]/1e3:.1f} km')
print(f'Number of channels: {metadata["nx"]}')
print(f'Number of time samples: {metadata["ns"]}')

### Select the desired channels and channel interval

selected_channels = [int(selected_channels_m_south // dx) for selected_channels_m_south in selected_channels_m_south]  
                                                        # list of values in channel number (spatial sample) corresponding to the starting, ending and step wanted
                                                        # channels along the FO Cable
                                                        # selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
                                                        # numbers

print('Begin channel #:', selected_channels[0], 
', End channel #: ',selected_channels[1], 
', step: ',selected_channels[2], 
'equivalent to ',selected_channels[2]*dx,' m')

### Load raw DAS data
                
# Loads the data using the pre-defined selected channels. 

tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)
trf = dw.dsp.bp_filt(tr, fs, 14, 28)
cable_name = 'South' 
dw.plot.plot_tx(sp.hilbert(trf, axis=1), time, dist, f'Bandpass filter, {cable_name}', v_max=0.4, fig_size=(12, 10))

# Create the f-k filter 
fk_params = {
'c_min': 1400.,
'c_max': 5000.,
'fmin': 14.,
'fmax': 28.
}

fk_filter = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, fk_params=fk_params, display_filter=True)

# Print the compression ratio given by the sparse matrix usage
dw.tools.disp_comprate(fk_filter)

# Apply the f-k filter to the data, returns spatio-temporal strain matrix
trf_fk = dw.dsp.fk_filter_sparsefilt(tr, fk_filter, tapering=True)
dw.plot.plot_fk_domain(tr, fs, dx, selected_channels, fig_size=(9, 8.8), v_min=0, v_max=0.000025, fk_params=fk_params, ax_lims=[12, 30, 0, 0.025])

# Delete the raw data to free memory
del tr

dw.plot.plot_tx(sp.hilbert(trf_fk, axis=1), time, dist, f'Hybrid filter, {cable_name}', v_max=0.4, fig_size=(12, 10))

# Plot the SNR
SNR = dw.dsp.snr_tr_array(trf_fk)
dw.plot.snr_matrix(SNR, time, dist, 20, "SNR estimation, "+f"{cable_name}")


# %%
#######
# SECTION 4 (author's main_section4.py): matched filter and Gabor filter masking
#######

import cv2

# Section 3 used c_max = 5000 m/s and fmax = 28 Hz
# Channel step: the author uses 5 (every 2nd channel, 4.08 m). 10 (every 4th channel, 8.17 m) is kept here,
# as in the cell above, to keep memory and run time down. 
selected_channels_m_45 = [12000, 66000, 10]
fk_params_45 = {'c_min': 1400., 'c_max': 3300., 'fmin': 14., 'fmax': 30.}

# Reload the North file: the cell above deleted the raw data (del tr), and running the South cell
# changes filepath and metadata to the South cable.
cable_name = 'North'
filepath, filename = dw.data_handle.dl_file(url_north[0])                  # already on disk, no download
metadata = dw.data_handle.get_acquisition_parameters(filepath, interrogator='optasense')
fs, dx = metadata["fs"], metadata["dx"]
selected_channels = [int(m // dx) for m in selected_channels_m_45]
tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)

# Hybrid f-k filter with the section 4-5 settings (not plotted again, it looks like the section 3 one).
# tr is kept for now: section 4c filters the raw data again for its noise band.
fk_filter_45 = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0], tr.shape[1]), selected_channels, dx, fs, fk_params=fk_params_45, display_filter=False)
trf_fk45 = dw.dsp.fk_filter_sparsefilt(tr, fk_filter_45, tapering=True)

#######
# Reference calls on the channel at 28.098 km (paper Fig. 4)
#######
# The author's script picks the channel holding the largest value. The paper names 28.098 km,
# so that channel is chosen directly here.
ch_ref = np.argmin(np.abs(dist - 28098))
print(f'Reference channel at {dist[ch_ref]/1e3:.3f} km')

p, tt, ff = dw.dsp.get_spectrogram(trf_fk45[ch_ref, :], fs, nfft=256, overlap_pct=0.95)   # Fig. 4(a)
dw.plot.plot_spectrogram(p, tt, ff, f_min=10, f_max=35, v_min=-45)
plt.show()
dw.plot.plot_3calls(trf_fk45[ch_ref], time, 6., 27.6, 48.5)                               # Fig. 4(b)

#######
# Template design (paper Fig. 5, Table I): synthetic notes against the measured calls
#######
HF_note_plot = dw.detect.gen_template_fincall(time, fs, fmin=17.8, fmax=28.8, duration=0.68)
LF_note_plot = dw.detect.gen_template_fincall(time, fs, fmin=14.7, fmax=21.8, duration=0.78)
dw.plot.design_mf(trf_fk45[ch_ref], HF_note_plot, LF_note_plot, 6.17, 28., time, fs)

#######
# Matched filter (paper Eq. 2 and 3): hyperbolic downsweeps faded in and out with a Hann window
#######
HF_note = dw.detect.gen_hyperbolic_chirp(17.8, 28.8, 0.68, fs)       # high note (B call): 28.8 -> 17.8 Hz in 0.68 s
HF_note = np.hanning(len(HF_note)) * HF_note
LF_note = dw.detect.gen_hyperbolic_chirp(14.7, 21.8, 0.78, fs)       # low note (A call): 21.8 -> 14.7 Hz in 0.78 s
LF_note = np.hanning(len(LF_note)) * LF_note

nmf_m_HF = dw.detect.calc_nmf_correlogram(trf_fk45, HF_note)         # one output per channel
nmf_m_LF = dw.detect.calc_nmf_correlogram(trf_fk45, LF_note)

# SNR of the matched filter outputs (paper Eq. 1, Fig. 6)
SNR_hf = dw.dsp.snr_tr_array(nmf_m_HF)
SNR_lf = dw.dsp.snr_tr_array(nmf_m_LF)
dw.plot.snr_matrix(SNR_hf, time, dist, 20, title_time_info=f'HF matched filter, {cable_name}')
dw.plot.snr_matrix(SNR_lf, time, dist, 20, title_time_info=f'LF matched filter, {cable_name}')

#######
# Gabor filter masking (paper Fig. 7 and Fig. 8a,b)
#######
c0 = 1500                                                            # speed of sound in water (m/s)
theta_c0 = dw.improcess.angle_fromspeed(c0, fs, dx, selected_channels)   # slope of a call in the image
gabfilt_up, gabfilt_down = dw.improcess.gabor_filt_design(theta_c0, plot=True)   # Fig. 7

def gabor_mask(snr_image, threshold):
    """The author's Gabor masking steps. The package's binning() needs torch, which is not installed,
    so cv2.resize does the same job: shrink the image 10 times in each direction, then enlarge the mask back."""
    im = np.clip(snr_image, 0, None)                                 # negative dB set to 0 (a copy; input unchanged)
    image = dw.improcess.scale_pixels(im) * 255
    small = cv2.resize(image, (image.shape[1] // 10, image.shape[0] // 10), interpolation=cv2.INTER_AREA)
    fimage = cv2.filter2D(small, cv2.CV_64F, gabfilt_up) + cv2.filter2D(small, cv2.CV_64F, gabfilt_down)
    fimage = dw.improcess.scale_pixels(fimage)
    mask = (fimage > threshold).astype(np.float32)                   # keep only strong, line-shaped features
    mask_full = cv2.resize(mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_LINEAR)
    return dw.improcess.apply_smooth_mask(im, mask_full)

for snr_image, note in ((SNR_hf, 'HF'), (SNR_lf, 'LF')):
    dw.plot.snr_matrix(gabor_mask(snr_image, 0.4), time, dist, 20, title_time_info=f'{note} Gabor filter, {cable_name}')

# %%
#######
# SECTION 4c (author's main_section4c.py): noise envelope subtraction
#######

# Noise reference: the same hybrid filter moved to 32-48 Hz, next to the call band but free of 20 Hz calls
fk_params_noise = {'c_min': 1400., 'c_max': 3300., 'fmin': 32., 'fmax': 48.}
fk_filter_noise = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0], tr.shape[1]), selected_channels, dx, fs, fk_params_noise, display_filter=False)
noise = dw.dsp.fk_filter_sparsefilt(tr, fk_filter_noise, tapering=True)
del tr                                                               # raw data no longer needed

SNR_noise = dw.dsp.snr_tr_array(noise)                               # paper Fig. 9(b)
dw.plot.snr_matrix(SNR_noise, time, dist, 20, title_time_info=f'Noise field, {cable_name}')
del SNR_noise

# Noise envelope: divide each channel by its standard deviation, take the envelope,
# then smooth it with a 100-sample (0.5 s) moving average
noise = dw.dsp.normalize_std(noise)
noise = dw.dsp.moving_average_matrix(np.abs(sp.hilbert(noise, axis=1)), 100)

# Denoised SNR (paper Eq. 4): matched filter envelope divided by the noise envelope, in dB.
# Both are divided by their channel's standard deviation first.
nmf_HF_std = dw.dsp.normalize_std(nmf_m_HF)
nmf_LF_std = dw.dsp.normalize_std(nmf_m_LF)
SNR_hf_dn = 20 * np.log10(np.abs(sp.hilbert(nmf_HF_std, axis=1)) / np.abs(sp.hilbert(noise, axis=1)))
SNR_lf_dn = 20 * np.log10(np.abs(sp.hilbert(nmf_LF_std, axis=1)) / np.abs(sp.hilbert(noise, axis=1)))
del nmf_HF_std, nmf_LF_std

# Smoothing as in the author's code. 
SNR_hf_dn = cv2.GaussianBlur(SNR_hf_dn, (9, 73), 0)
SNR_lf_dn = cv2.GaussianBlur(SNR_lf_dn, (9, 73), 0)

dw.plot.snr_matrix(SNR_hf_dn, time, dist, 20, title_time_info=f'HF matched filter denoised, {cable_name}')   # Fig. 9(c)
dw.plot.snr_matrix(SNR_lf_dn, time, dist, 20, title_time_info=f'LF matched filter denoised, {cable_name}')

# Gabor masking on the denoised SNR (paper Fig. 8c,d), with the author's lower threshold of 0.2
for snr_image, note in ((SNR_hf_dn, 'HF'), (SNR_lf_dn, 'LF')):
    dw.plot.snr_matrix(gabor_mask(snr_image, 0.2), time, dist, 20, title_time_info=f'{note} Gabor filter denoised, {cable_name}')

# %%
#######
# SECTION 5 (author's main_section5.py): picking arrival times, before and after denoising
#######

th = 4.      # a pick must reach at least 4 dB
ipi = 2      # and be at least 2 s after the previous pick on the same channel ("inter-pulse interval")

# Before denoising: the matched filter SNR from section 4, smoothed like the author's code.
# After denoising: the section 4c SNR, already smoothed. Negative dB set to 0 in both.
snr_maps = {
    'before': {'HF': np.clip(cv2.GaussianBlur(SNR_hf, (9, 73), 0), 0, None),
               'LF': np.clip(cv2.GaussianBlur(SNR_lf, (9, 73), 0), 0, None)},
    'after':  {'HF': np.clip(SNR_hf_dn, 0, None),
               'LF': np.clip(SNR_lf_dn, 0, None)},
}

def pick_times(snr, threshold, ipi_s):
    """Peaks of each channel's SNR above a threshold and at least ipi_s apart.
    Returns (channel indices, time indices)."""
    distance = max(int(round(ipi_s * fs)), 1)
    peaks = [sp.find_peaks(row, distance=distance, height=threshold)[0] for row in snr]
    return dw.detect.convert_pick_times(peaks)

# Picks with th = 4 dB and ipi = 2 s (paper Fig. 11)
for stage in ('before', 'after'):
    for note, colour in (('HF', 'tab:blue'), ('LF', 'tab:red')):
        snr = snr_maps[stage][note]
        picks = pick_times(snr, th, ipi)
        values = snr[picks[0], picks[1]]                              # SNR of each pick
        sizes = 2 + (values - values.min()) / (values.max() - values.min()) * 138   # dot size from 2 to 140
        print(f'{note} picks {stage} denoising: {len(picks[0])}')
        plt.figure(figsize=(12, 10))
        plt.scatter(picks[1] / fs, dist[picks[0]] / 1e3, color=colour, marker='.', s=sizes, rasterized=True)
        plt.xlabel('Time (s)')
        plt.ylabel('Distance (km)')
        plt.title(f'{note} note picks, {stage} denoising, {cable_name}', loc='right')
        handles = [plt.scatter([], [], s=s, color=colour, label=f'{name} SNR: {v:.1f} dB')
                   for s, name, v in ((2, 'Min', values.min()), (71, 'Median', np.median(values)), (140, 'Max', values.max()))]
        plt.legend(handles=handles, loc='upper right')
        plt.show()

# How the number of picks depends on the threshold and on the minimum spacing (paper supplementary)
thresholds = np.arange(2, 20.5, 0.5)                                  # dB
IPIs = np.arange(0, 5.5, 0.5)                                         # s
IPIs[0] = 1 / fs                                                      # smallest possible spacing: one sample
counts = {(stage, note): ([len(pick_times(snr_maps[stage][note], t, ipi)[0]) for t in thresholds],
                          [len(pick_times(snr_maps[stage][note], th, i)[0]) for i in IPIs])
          for stage in ('before', 'after') for note in ('HF', 'LF')}

plt.figure(figsize=(12, 10))
for row, (x, xlabel) in enumerate(((thresholds, 'Threshold (dB)'), (IPIs, 'Inter-pulse interval (s)'))):
    plt.subplot(2, 1, row + 1)
    for (stage, note), c in counts.items():
        plt.plot(x, c[row], lw=2, ls='-' if stage == 'before' else '--', label=f'{note} {stage} denoising')
    plt.xlabel(xlabel)
    plt.ylabel('Number of picks')
    plt.legend()
    plt.grid()
plt.tight_layout()
plt.show()


