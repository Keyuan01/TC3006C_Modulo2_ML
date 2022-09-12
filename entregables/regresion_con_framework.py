# Importar librerías necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Importar dataset

# Usar sólo las columnas necesarias
columns = ['Location','MinTemp', 'MaxTemp']
df = pd.read_csv("weatherAUS.csv", usecols= columns)

# Sólo usar la ciudad de Canberra para establecer el modelo
df = df[df['Location'] == 'Canberra']

# df = df.groupby(['Date'])[['MinTemp','MaxTemp']]

# Limpieza de datos - eliminar los datos vacíos
df = df.dropna()

x = np.array(df['MinTemp']).reshape((-1,1))
y = df['MaxTemp']

model = LinearRegression(fit_intercept=True).fit(x,y)

predicts = model.predict(x)

fig, ax = plt.subplots(figsize=(20, 10))
ax.plot(x, y, 'o',alpha=0.2, label="data")
ax.plot(x, predicts, 'r-')
ax.legend(loc='best')
plt.show() 