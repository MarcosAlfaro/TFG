
"""ESTE PROGRAMA SIRVE PARA GENERAR UN CSV CON LOS LAS IMÁGENES DE ENTRENAMIENTO
    Cada línea del csv incluye:
    -Una imagen anchor
    -Una imagen positive
    -Una imagen negative
    Mediante bucles for y condicionales if, recorremos TODAS las combinaciones posibles de anchor,positive y negative"""


# Importamos las librerías que son necesarias

import os
import torchvision.datasets as dset
import csv
from config import PARAMETERS

# Esta función nos devuelve la ruta del directorio en el que estamos trabajando
base_dir = os.getcwd()

# Creamos una clase en la que definimos las rutas de los directorios de los cuales extraemos las imágenes
class Config():
    training_dir = os.path.join(base_dir, PARAMETERS.training_dir)
    # testing_dir = os.path.join(base_dir, 'TestDataset')


# Creamos un vector en el que cada elemento representa una habitación del edificio
folder_dataset = dset.ImageFolder(root=Config.training_dir)
rooms = folder_dataset.classes


# Generamos el archivo .csv en modo escritura, creando las tres columnas (Anchor,Positive,Negative)
with open(base_dir + PARAMETERS.train_csv_dir, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Anchor", "Positive", "Negative"])


# Creamos varios bucles anidados de forma que se recorran todas las combinaciones posibles de tres imágenes con las restricciones siguientes:
# Las imágenes anchor, positive y negative tienen que ser distintas
# Las imágenes anchor y positive deben pertenecer a la misma habitación
# Las imágenes anchor y negative deben pertenecer a distintas habitaciones

# con los dos primeros bucles recorremos todas las imágenes de cada habitación, y asignamos sus rutas a la columna 'Anchor'
    for habitacion_anc in rooms:
        ruta_habitacion_anc = os.path.join(Config.training_dir, habitacion_anc)             #obtenemos la ruta de la habitación de la iteración actual
        lista_imagenes_anc = os.listdir(ruta_habitacion_anc)                                 #para cada habitación, creamos una lista con las rutas de todas las imágenes de esa habitación


        for anchor in lista_imagenes_anc:
            ruta_imagen_anc = os.path.join(ruta_habitacion_anc, anchor)                     #para cada imagen de la habitación actual, obtenemos su ruta

# con el tercer y cuarto bucle recorremos todas las imágenes pero solo cogemos las que pertenezcan a la misma habitación que la imagen anchor actual, y asignamos sus rutas a la columna 'Positive'
            for habitacion_pos in rooms:
                ruta_habitacion_pos = os.path.join(Config.training_dir, habitacion_pos)
                lista_imagenes_pos = os.listdir(ruta_habitacion_pos)
                if ruta_habitacion_pos == ruta_habitacion_anc:                                # este if nos asegura que cogemos solo las imágenes que pertenezcan a la misma habitación que la imagen anchor actual
                    for positive in lista_imagenes_anc:
                        ruta_imagen_pos = os.path.join(ruta_habitacion_pos, positive)
                        if ruta_imagen_pos != ruta_imagen_anc:                                # este if nos asegura que la imagen positive y la imagen anchor sean distintas

# con el quinto y sexto bucle recorremos todas las imágenes de cada habitación pero solo cogemos las que pertenezcan a una habitación distinta a la de la imagen anchor actual, y asignamos sus rutas a la columna 'Negative'
                            for habitacion_neg in rooms:
                               ruta_habitacion_neg = os.path.join(Config.training_dir, habitacion_neg)
                               lista_imagenes_neg = os.listdir(ruta_habitacion_neg)
                               if ruta_habitacion_neg != ruta_habitacion_anc:               # este if nos asegura que la imagen anchor y la imagen negative pertenecen a habitaciones distintas
                                   for negative in lista_imagenes_neg:
                                        ruta_imagen_neg = os.path.join(ruta_habitacion_neg, negative)


# Para cada combinación creada, añadimos una fila en el archivo .csv con las rutas de las tres imágenes
                                        writer.writerow([ruta_imagen_anc, ruta_imagen_pos, ruta_imagen_neg])

