# AJUSTES POLINOMIALES
# -----------------------------------------------------------------
# Lección 06
#
# ** Se importan los archivos de trabajo
# ** Se crean las variables
# ** Se generan los modelos
# ** Se grafican las funciones

# Se importa la librería del Sistema Operativo
# Igualmente, la librería utils y numpy
# -----------------------------------------------------------------
from scipy.optimize.minpack import fsolve
import matplotlib.pyplot as plt
import scipy as sp
import os

# Directorios: chart y data en el directorio de trabajo
# -----------------------------------------------------------------
from utils import DATA_DIR, CHART_DIR
import numpy as np

# Se eliminan las advertencias por el uso de funciones que
# en el futuro cambiarán
# -----------------------------------------------------------------
np.seterr(all='ignore')

colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']

# Se importa la librería scipy y matplotlib
# -----------------------------------------------------------------

data = np.loadtxt(os.path.join(DATA_DIR, "DataS.txt"))
# data = np.genfromtxt(os.path.join(DATA_DIR, "C:/Users/Jefferson/Desktop/Computacion_Blanda/DataS.txt"), delimiter="\n")

# se establece el tipo de dato
data = np.array(data, dtype=np.float64)

x = data[0]
y = data[1]


def plot_models(x, y, models, fname, mx=None, ymax=None, xmin=None):
    ''' dibujar datos de entrada '''
    plt.figure(num=None, figsize=(8, 6))
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title("Casos de coronavirus en risaralda")
    plt.xlabel("Tiempo en dias")
    plt.ylabel("Casos diarios")
    """ plt.xticks(
        [w * 7 * 24 for w in range(10)],
        ['semana %i' % w for w in range(10)]) """

    if models:
        if mx is None:
            mx = np.linspace(0, x[-1], 1000)
        for model, style, color in zip(models, linestyles, colors):
            print('mx', mx)
            plt.plot(list(range(250)), model(list(range(250))),
                     linestyle=style, linewidth=2, c=color)

        plt.legend(["d=%i" % m.order for m in models], loc="upper left")

    plt.autoscale(tight=True)
    plt.ylim(ymin=0)
    if ymax:
        plt.ylim(ymax=ymax)
    if xmin:
        plt.xlim(xmin=xmin)
    plt.grid(True, linestyle='-', color='0.75')
    plt.savefig(fname)


def error(f, x, y):
    return np.sum((f(x) - y) ** 2)


# Primera mirada a los datos
# -----------------------------------------------------------------
plot_models(x, y, None, os.path.join(CHART_DIR, "1400_01_01.png"))
# Crea y dibuja los modelos de datos
# -----------------------------------------------------------------
fp1, res1, rank1, sv1, rcond1 = np.polyfit(x, y, 1, full=True)
print("Parámetros del modelo fp1: %s" % fp1)
print("Error del modelo fp1:", res1)
f1 = sp.poly1d(fp1)

fp2, res2, rank2, sv2, rcond2 = np.polyfit(x, y, 2, full=True)
print("Parámetros del modelo fp2: %s" % fp2)
print("Error del modelo fp2:", res2)
f2 = sp.poly1d(fp2)

funcion_definitiva = sp.poly1d(np.polyfit(x, y, 5))

# Se grafican los modelos
# -----------------------------------------------------------------
plot_models(x, y, [funcion_definitiva],
            os.path.join(CHART_DIR, "1400_01_02.png"))

# No se tienen que sacar datos aleatorios de entrenamiento porque los datos que tenemos son pocos

# Haremos la proyeccion de cuando no habran casos de coronavirus
# ---------------------------------------------------------------
prediccion = fsolve(funcion_definitiva, x0=165)
print('Esperamos encontrar 0 casos diarios el dia: ', prediccion)
