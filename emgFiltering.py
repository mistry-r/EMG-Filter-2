import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
import os

# Function to read EMG data from CSV file
def read_emg_data(file_path):
    with open(file_path, 'r') as input_file:
        lines = input_file.readlines()
    
    blank_line_index = lines.index('\n')
    emg_lines = lines[5:blank_line_index]
    
    emg_data = []
    time = 0
    time_increment = 0.5
    
    for line in emg_lines:
        parts = line.strip().split(',')
        if len(parts) < 2:
            continue
        emg_sensors = {'time': round(time, 5)}
        for i, value in enumerate(parts[2:], start=1):
            if value.strip() == '':
                continue
            emg_sensors[f'emg_sensor_{i}'] = float(value)
        emg_data.append(emg_sensors)
        time += time_increment
    
    emg_df = pd.DataFrame(emg_data)
    
    return emg_df

# Function for full wave rectification
def full_wave_rectification(emg_data):
    for col in emg_data.columns[1:]:
        emg_data[col] = np.abs(emg_data[col])
    return emg_data

# Function to design a Butterworth filter
def butter_filter(cutoff_freq, sampling_freq, btype, order=2):
    nyquist_freq = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype=btype, analog=False)
    return b, a

# Function to apply a Butterworth filter to the EMG signal
def butter_filter_signal(emg_data, cutoff_freq, sampling_freq, btype, order=2):
    emg_data_filtered = emg_data.copy()
    for col in emg_data.columns[1:]:
        b, a = butter_filter(cutoff_freq, sampling_freq, btype, order=order)
        emg_data_filtered[col] = filtfilt(b, a, emg_data[col])
    return emg_data_filtered

# Function for non-inverting amplification
def non_inverting_amplifier(emg_data, gain):
    emg_data_amplified = emg_data.copy()
    for col in emg_data_amplified.columns[1:]:
        emg_data_amplified[col] *= gain
    return emg_data_amplified

# Function for Root Mean Square (RMS) smoothing
def rms_smoothing(emg_data, window_size):
    emg_smoothed = pd.DataFrame({'time': emg_data['time']})  # Create a DataFrame for smoothed data
    for col in emg_data.columns[1:]:
        emg_smoothed[f'RMS_smoothed_{col}'] = emg_data[col].rolling(window=window_size).apply(lambda x: np.sqrt(np.mean(x**2)))
    return emg_smoothed

# Function for Moving Average (MOVAG) smoothing
def movag_smoothing(emg_data, window_size):
    emg_smoothed = pd.DataFrame({'time': emg_data['time']})  # Create a DataFrame for smoothed data
    for col in emg_data.columns[1:]:
        emg_smoothed[f'MOVAG_smoothed_{col}'] = emg_data[col].rolling(window=window_size).mean()
    return emg_smoothed

# Function to plot EMG data
def plot_emg_data(emg_data):
    plt.figure(figsize=(10, 6))
    for col in emg_data.columns[1:3]: 
        plt.plot(emg_data['time'], emg_data[col], label=col)
    plt.xlabel('Time (ms)')
    plt.ylabel('EMG Signal (V)')
    plt.title('EMG Data')
    plt.legend()
    plt.grid(True)
    plt.show()

# Function to save EMG data to CSV file
def save_emg_data(emg_data, file_path):
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    emg_df = pd.DataFrame(emg_data)
    output_file = os.path.join(output_dir, file_path + '.csv')
    emg_df.to_csv(output_file, index=False)

# Main function
def main():
    # File path setup
    input_file_path = 'input/Ehsan Trial 3 EMG.csv'
    output_file_path = 'filtered_EMG'
    emg_data = read_emg_data(input_file_path)

    # Full wave rectifier filter parameters
    emg_rectified = full_wave_rectification(emg_data)

    # Non-inverting amplifier parameters
    emg_amplified = non_inverting_amplifier(emg_rectified, 10000)

    # High-pass filter parameters
    #high_filtered_signal = butter_filter_signal(emg_rectified, 0.6, 2000, btype='high')

    # Low-pass filter parameters
    #low_filtered_signal = butter_filter_signal(high_filtered_signal, 0.4, 2000, btype='low')

    # RMS smoothing parameters
    emg_rms_smoothed = rms_smoothing(emg_amplified, 100)
    rms_output_file_path = 'RMS_smoothed_data'
    plot_emg_data(emg_rms_smoothed)
    save_emg_data(emg_rms_smoothed, rms_output_file_path)

    # MOVAG smoothing parameters
    #emg_movag_smoothed = movag_smoothing(low_filtered_signal, 100)
    #movag_output_file_path = 'MOVAG_smoothed_data'
    #plot_emg_data(emg_movag_smoothed)
    #save_emg_data(emg_movag_smoothed, movag_output_file_path)

    # Plotting parameters
    plot_emg_data(emg_amplified)
    save_emg_data(emg_amplified, output_file_path)

if __name__ == "__main__":
    main()