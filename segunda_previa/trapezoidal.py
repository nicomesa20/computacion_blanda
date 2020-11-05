import numpy as np
import skfuzzy as sk
import matplotlib.pyplot as plt

# Se defube un array x para el manejo del factor de calidad en un restaurante
x = np.arange(0, 11, 1)

# Se defube un array para la funcion miembro trapezoidal
calidad = sk.trapmf(x, [0, 3, 5, 8])

# Graficar
plt.figure()
plt.plot(x, calidad, 'b', linewidth=1.5, label='Servicio')

plt.title('Calidad del servicio en un restaurante')
plt.ylabel('Membresia')
plt.xlabel('Nivel de servicio')
plt.legend(loc='center right', bbox_to_anchor=(
    1.25, 0.5), ncol=1, fancybox=True, shadow=True)
