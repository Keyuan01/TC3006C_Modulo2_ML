import numpy as np
import pandas as pd
import pylab

# Usar sólo las columnas necesarias
columns = ['Location','MinTemp', 'MaxTemp']
df = pd.read_csv("weatherAUS.csv", usecols= columns)

# Sólo usar la ciudad de Canberra para establecer el modelo
df = df[df['Location'] == 'Canberra']

# df = df.groupby(['Date'])[['MinTemp','MaxTemp']]

x = df['MinTemp']
y = df['MaxTemp']

# print(df.info())

# Valores iniciales
alfa = 0.001
b = 0
m = 0
epocas = 5000

# Crear un arreglo de ceros para almacenar los errores
error = np.zeros((epocas,1))

# Mean Square Error
def mse(b, m, x, y):
    n = y.shape[0]

    #Sumatoria de los errores al cuadrado
    mean_square_error = np.sum((y-m*x-b)**2)
    return mean_square_error/n

def optimizer(x, y, b, m, alfa, epocas, error):

    #Gradiente descendente
    for i in range(epocas):
        # Actualizar b y m
        b, m = compute_gradient(b, m, x, y, alfa)
        # Almacenar MSE a un arreglo
        error[i] = mse(b, m, x, y)
        
    return b, m

# Actualizar b y m
def compute_gradient(b, m, x, y, alfa):

    # Derivada
    n = y.shape[0]

    db = -(2/n)*np.sum(y-m*x-b)
    dm = -(2/n)*np.sum(x*(y-m*x-b))
    
    # A través de Alfa(Learning rate) actualizar b y m
    b = b - (alfa * db)
    m = m - (alfa * dm)

    return b, m

def plot_df(x, y, b, m, error, epocas):

    pylab.plot(range(epocas), error)
    pylab.xlabel('Epocas')
    pylab.ylabel('MSE')
    pylab.show()

    y_predict = m*x+b
    pylab.plot(x,y,'o', alpha=0.2)
    pylab.xlabel('Temp Min')
    pylab.ylabel('Temp Max')
    pylab.title('Temperatura de la ciudad Canberra')
    pylab.plot(x,y_predict,'k-')
    pylab.show()

#Optimización de b y m
b, m = optimizer(x, y, b, m, alfa, epocas, error)

#Graficar los resultados
plot_df(x, y, b, m, error, epocas)

print(error[-1])
