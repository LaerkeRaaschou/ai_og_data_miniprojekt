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
    return transformeret_data

data = data[0:1900]
data = data.drop(data.columns[0], axis=1)
data["id"] = range(1, len(data) + 1)
data.set_index("id", inplace=True)

sound_values = data["sound_level"].values
sound_values = np.array(sound_values)

average_data = data["sound_level"].rolling(11).mean()
fourier_trans_low_pass = frekvens_domain_filter(sound_values)

data["sound_level_smooth"] = average_data
data["low_pass"] = fourier_trans_low_pass

# Step 4: Plot one or more variables
plt.figure(figsize=(12, 5))

plt.plot(data.index, data["sound_level"], label="Original sound")
plt.plot(data.index, data["low_pass"], label="Low pass filter")

plt.xlabel("Time")
plt.ylabel("Sound levels")
plt.title("Time series plot: Sound levels")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()