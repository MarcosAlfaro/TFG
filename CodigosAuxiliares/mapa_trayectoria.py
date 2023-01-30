
"""ESTE PROGRAMA SIRVE PARA GENERAR UNA IMAGEN CON LAS COORDENADAS DE TODAS LAS IMÁGENES DE UN DATASET
    En el mapa se distingue entre:
    -azul: imágenes del dataset
    -rojo: imagen más cercana al centro geométrico de una habitación
    -verde: centro geométrico de cada habitación
    -amarillo: primera imagen tomada de cada habitación
    -naranja: última imagen tomada de cada habitación
    """



import torchvision.datasets as dset
import matplotlib.pyplot as plt
import os
from config import PARAMETERS

# Ruta sobre la cual estamos trabajando
base_dir = os.getcwd()

# Indice de la habitación actual con la que se está trabajando
indice_hab = 0

# Par de variables que guardan las coordenadas x e y de todas las imágenes tomadas
mapa_x = []
mapa_y = []

# Par de variables que guardan las coordenadas x e y de los centros geométricos de cada habitación
x_centro = []
y_centro = []

# Par de variables que guardan las coordenadas x e y de las imágenes más próximas al centro geométrico de cada habitación
cercana_x = []
cercana_y = []

# Par de variables que guardan las coordenadas x e y de la primera imagen tomada de cada habitación
img_inicio_x = []
img_inicio_y = []

# Par de variables que guardan las coordenadas x e y de la última imagen tomada de cada habitación
img_fin_x = []
img_fin_y = []

# Ruta de cada una de las carpetas de cada habitación con todas sus imágenes
img_dir = os.path.join(base_dir, PARAMETERS.training_dir)
folder_dataset = dset.ImageFolder(root=img_dir)
rooms = folder_dataset.classes

#Con este bucle obtenemos las coordenadas x e y de todas las imágenes del centro geométrico de cada habitación
for habitacion in rooms:
    ruta_habitacion = os.path.join(img_dir, habitacion)             
    lista_imagenes = os.listdir(ruta_habitacion)
    centro_x = 0
    centro_y = 0
    indice_img=0


    for img in lista_imagenes:
        ruta_imagen = os.path.join(ruta_habitacion, img)

        x_index = ruta_imagen.index('_x')
        y_index = ruta_imagen.index('_y')
        a_index = ruta_imagen.index('_a')

        coord_x = ruta_imagen[x_index+2:y_index]
        coord_y = ruta_imagen[y_index+2:a_index]

        coord_x = float(coord_x)
        coord_y = float(coord_y)

        centro_x += coord_x
        centro_y += coord_y

        # obtenemos las coordenadas x e y de la primera imagen tomada de cada habitación
        if indice_img == 0:
             img_inicio_x.append(coord_x)
             img_inicio_y.append(coord_y)
        # obtenemos las coordenadas x e y de la última imagen tomada de cada habitación
        elif indice_img == len(lista_imagenes)-1:
            img_fin_x.append(coord_x)
            img_fin_y.append(coord_y)
        # obtenemos las coordenadas x e y del resto de imágenes
        else:
            mapa_x.append(coord_x)
            mapa_y.append(coord_y)
        indice_img += 1


    centro_x /= len(lista_imagenes)
    centro_y /= len(lista_imagenes)

    # obtenemos las coordenadas x e y del centro geométrico de cada habitación
    x_centro.append(centro_x)
    # mapa_x.remove(centro_x)
    y_centro.append(centro_y)
    # mapa_y.remove(centro_y)

# Ruta de cada una de las carpetas de cada habitación con la imagen más próxima al centro geométrico
centro_dir = os.path.join(base_dir, PARAMETERS.centro_geom_dir)
centro_dataset = dset.ImageFolder(root=centro_dir)
rooms_centro = centro_dataset.classes

# En este bucle obtenemos las coordenadas x e y de las imágenes más cercanas al centro geométrico
for habitacion_centro in rooms_centro:
    ruta_habitacion_centro = os.path.join(centro_dir, habitacion_centro)
    lista_imagenes_centro = os.listdir(ruta_habitacion_centro)


    for img in lista_imagenes_centro:
        ruta_imagen_centro = os.path.join(ruta_habitacion_centro, img)


        x_index = ruta_imagen_centro.index('_x')
        y_index = ruta_imagen_centro.index('_y')
        a_index = ruta_imagen_centro.index('_a')

        coord_x_centro = ruta_imagen_centro[x_index+2:y_index]
        coord_y_centro = ruta_imagen_centro[y_index+2:a_index]

        coord_x_centro = float(coord_x_centro)
        coord_y_centro = float(coord_y_centro)

        # obtenemos las coordenadas x e y de las imágenes más cercanas al centro geométrico
        cercana_x.append(coord_x_centro)
        cercana_y.append(coord_y_centro)
        # mapa_x.remove(coord_x_centro)
        # mapa_y.remove(coord_y_centro)


# print(x_centro)
# print(y_centro)

# print(mapa_x)
# print(centro_x)

#Dibujamos en una gráfica el mapa de la habitación con los puntos representados
plt.scatter(mapa_x, mapa_y, color='blue')
plt.scatter(cercana_x, cercana_y, color='red')
plt.scatter(x_centro, y_centro, color='green')
plt.scatter(img_inicio_x, img_inicio_y, color='yellow')
plt.scatter(img_fin_x, img_fin_y, color='orange')

plt.title("Trayectoria robot", fontsize=15)
plt.xlabel("X", fontsize=13)
plt.ylabel("Y", fontsize=13)
plt.legend(['Puntos trayectoria robot', 'Img más cercana al centro geométrico', 'Centro geométrico', 'Primera imagen tomada', 'Última imagen tomada'])
plt.axis([-16, 7, -7, 7])
plt.show()






