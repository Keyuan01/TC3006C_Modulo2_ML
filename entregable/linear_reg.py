import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv("weatherAUS.csv")

# Convertir la fecha por el formato datetime
df['Date'] = pd.to_datetime(df['Date'])
#print(df['Date'].info())



average_humidity = df.groupby('Date').sum()['RainToday']
average_humidity.plot(kind='pie', figsize=(20,10))
plt.show()


