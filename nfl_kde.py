import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import gaussian_kde
import numpy as np

data = pd.read_csv('/Users/laerkeraaschou/Desktop/semester2/ai_og_data/ai_og_data_miniprojekt/NFL.csv')

print(data.info())

def histogram(data, xakse):
    """ making histogram from data """
    data = data.dropna()
    # Plot histogram
    plt.figure(figsize=(8, 5))
    hist, bins, _ = plt.hist(data, bins=25, density=True, alpha=0.5)

    # Labels og titel
    plt.xlabel(xakse)
    plt.ylabel('Frekvens')
    plt.title('Histogram over data')

    # Vis histogrammet
    plt.show()
    return hist

def plot_hist_m_kde(data, density):
    hist, bins, _ = plt.hist(data, bins=25, density=True, alpha=0.5)
    xs = np.linspace(data.min(), data.max(), 200)
    density_values = density(xs)
    # Plot the estimated kernel density
    plt.plot(xs, density_values, label='Kernel Density Estimation')
    # Add labels and title
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.title("Histogram with Kernel Density Estimation")
    plt.legend()
    # Display the plot
    plt.show()


def KDE(data):
    # tæller og fjerner NAN fra datasættet
    nans_in_data = data.isna().sum()
    data = data.dropna()

    # estimerere sandsynlighedstæthedsfunktion basseret på datafordelingen
    density = gaussian_kde(data)

    # vælger værdier ud fra tæthedsfunktionen - ANTAL: så mange som mangler i data
    impute_values = density.resample(nans_in_data, seed=39)
    
    # plot af datafordelingen og den estimerede sandsynlighedtæthedsfunktion
    plot_hist_m_kde(data, density)

    return impute_values.flatten()

def impute_missing_values(data, feature):
    # generere de manglende værdier ud fra en estimeret tæthedsfunktion
    missing_values = KDE(data[feature])

    # finder indexer for de manglende værdier
    nan_indices = data[feature][data[feature].isna()].index

    # sætter genereret værdi ind på mangendes plads
    for i, idx in enumerate(nan_indices):
        data.at[idx, feature] = round(missing_values[i],2)

impute_missing_values(data,"Sprint_40yd")
histogram(data["Sprint_40yd"],"Sprint_40yd")