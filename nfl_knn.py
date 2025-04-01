import pandas as pd
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

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

data = pd.read_csv('/Users/laerkeraaschou/Desktop/semester2/ai_og_data/ai_og_data_miniprojekt/NFL.csv')

data = data.drop(['Year','Player','Age','School','Height','Weight','Drafted..tm.rnd.yr.','BMI','Player_Type','Position_Type','Position','Drafted'],axis=1)

print(data.head())

sprint = data['Sprint_40yd']
data = data.drop(['Sprint_40yd'],axis=1)
data = data.dropna()

data['Sprint_40yd'] = sprint
print(data.info())

knn_imputer = KNNImputer(n_neighbors=3, weights="uniform")
data_imputed = knn_imputer.fit_transform(data)

panda_array = pd.DataFrame(data_imputed,columns=['Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle' ,'Sprint_40yd'])
print(panda_array.info())
histogram(panda_array['Sprint_40yd'], 'Sprint_40yd')

