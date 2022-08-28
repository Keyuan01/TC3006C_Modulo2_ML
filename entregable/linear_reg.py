import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Usar sólo las columnas necesarias
columns = ['MinTemp', 'MaxTemp']
df = pd.read_csv("weatherAUS.csv", usecols= columns)
# df = df.groupby(['Date'])[['MinTemp','MaxTemp']]
# df.info()

# Valores iniciales
m = 0.1
alfa = 0.001
b = 1
epocas = 100

x = df['MinTemp'].values
y = df['MaxTemp'].values

# print(df.info())

'''
Modelo
m*x+b
'''
def modelo(m,x,b):
    return m*x + b

'''
Mean Square Error
n: cantidad de muestras
y: valores reales
y_p: valores predicha
'''

def mse(y, y_p):

    n = y.shape[0]
    #Sumatoria de los errores al cuadrado
    mean_square_error = np.sum((y-y_p)**2)

    return mean_square_error/n

def reg_lineal_gd(x, y, m, b, alfa):

    n = x.shape[0]
    
    # Derivada
    dm = -(2/n)*np.sum(x*(y-(m*x+b)))
    db = -(2/n)*np.sum(y-(m*x+b))

    m = m - alfa*dm
    b = b - alfa*db

    return m, b


# Crear un arreglo de ceros para almacenar los errores
error = np.zeros((epocas,1))

for i in range(epocas):
    [m, b] = reg_lineal_gd(x, y, m, b, alfa)

    y_p = modelo(m, x, b)

    error[i] = mse(y, y_p)


#print(df.head())
plt.plot(range(epocas), error)
plt.xlabel('Epocas')
plt.ylabel('MSE')
plt.show()

y_regr = modelo(m,b,x)
plt.scatter(x,y)
plt.plot(x,y_regr,'r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()