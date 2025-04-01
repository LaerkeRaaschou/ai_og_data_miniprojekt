import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('/Users/laerkeraaschou/Desktop/semester2/ai_og_data/ai_og_data_miniprojekt/coffeedata.csv')

def dft(signal):
    ''' transforming to frekvens domain'''
    N = len(signal)
    X = []
    for k in range(N):
        sum_val = 0
        for n in range(N):
            sum_val += signal[n] * np.exp(-2j * np.pi * k * n / N)
        X.append(sum_val)
    return np.array(X)

def invdft(X):
    ''' from frekvens to time domain'''
    X = np.asarray(X)
    N = len(X)
    x = []
    for n in range(N):
        sum_val = 0
        for k in range(N):
            sum_val += X[k] * np.exp(2j * np.pi * k * n / N)
        x.append(sum_val / N)
    return np.array(x).real 

def frekvens_domain_filter(data):
    ''' transforming data to frekvens domain and filter and then back '''
    X = dft(data)
    X_final = np.copy(X)
    #X_final = X[:len(X)//2] # TAG HALVDELEN!!!!
    P = 100 # cutoff

    for n in range(len(X_final)):
        if n <= P:
            X_final[n] = X_final[n] * 1
        elif n >= len(X_final) - P:
            X_final[n] = X_final[n] * 1
        else:
            X_final[n] = X_final[n] * 0
    
    transformeret_data = invdft(X_final)
    
    plt.figure(figsize=(12, 5))
    plt.plot(X_final)
    plt.xlabel("Frekvens (Hz)")
    plt.ylabel("Amplitude")
    plt.title("Signal i frekvensdomænet efter lavpas filter (p=100)")
    plt.grid(True)
    plt.show()
    return transformeret_data

data = data[0:1900]
data = data.drop(data.columns[0], axis=1)
data["id"] = range(1, len(data) + 1)

data.set_index("id", inplace=True)
sound_values = data["sound_level"].values
sound_values = np.array(sound_values)
print(sound_values)
average_data = data["sound_level"].rolling(5).mean()
print(average_data)

X = np.fft.fft(sound_values)
freqs = np.fft.fftfreq(len(X), d=1/len(X))
amplitudes = np.abs(X)

cutoff = 100  # fx 30 Hz
X_filtered = X.copy()
X_filtered[np.abs(freqs) > cutoff] = 0

filtered_signal = np.fft.ifft(X_filtered).real

efter_transformation = frekvens_domain_filter(sound_values)



#nyt_data = frekvens_domain_filter(sound_values)
#df = pd.DataFrame(nyt_data, columns=["sound_levelT"])

data["sound_level_smooth"] = data["sound_level"].rolling(window=11).mean()
data["efter_frekvens"] = X
# Step 4: Plot one or more variables
plt.figure(figsize=(12, 5))
#plt.plot(data.index, data["accel_x"], label="X")
#plt.plot(data.index, data["accel_y"], label="Y")
#plt.plot(data.index, data["accel_z"], label="Z")
#plt.plot(data.index, data["sound_level"], label="Original sound")
plt.plot(data.index, data["efter_frekvens"], label="Low pass filter")
plt.xlabel("Time")
plt.ylabel("Sound levels")
plt.title("Time series plot: Sound levels")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()

# Save to working directory
plt.savefig('transformation.jpg')
plt.close()


# vi måler nu 2 ms mellem målinger = 1/0.002 = 500 Hz og så skal vi omregne frekvens på det dobbelte = 1000