import numpy as np
import skfuzzy as sk
import matplotlib.pyplot as plt

# Se defube un array x para el manejo del factor de calidad en un restaurante
x = np.arange(30, 80, 0.1)

# Se define un array para la funcion miembro gauss
lento = sk.trimf(x, [30, 30, 50])
medio = sk.trimf(x, [30, 50, 70])
medio_rapido = sk.trimf(x, [50, 60, 70])
rapido = sk.trimf(x, [60, 80, 780])

# Graficar
plt.figure()
plt.plot(x, rapido, 'b', linewidth=1.5, label='Rapido')
plt.plot(x, medio_rapido, 'k', linewidth=1.5, label='Medio-rapido')
plt.plot(x, medio, 'm', linewidth=1.5, label='Medio')
plt.plot(x, lento, 'r', linewidth=1.5, label='Lento')

plt.title('Pentalti difuso')
plt.ylabel('Membresia')
plt.xlabel('Velocidad (km/h)')
plt.legend(loc='center right', bbox_to_anchor=(
    1.25, 0.5), ncol=1, fancybox=True, shadow=True)
