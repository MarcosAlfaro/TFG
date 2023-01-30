
"""ESTE PROGRAMA SIRVE PARA GENERAR UN CSV CON LOS LAS IMÁGENES DE VALIDACIÓN
   Para cada imagen del dataset de validación utilizado,
   se escribe una línea en el csv por cada habitación del edificio,
   que incluye:
   -la ruta de la imagen de validación
   -la ruta de la imagen representativa de una habitación (suele coincidir con la imagen más próxima al centro geométrico de esa habitación)
   -el índice de la habitación de la imagen de validación"""


import os
import torchvision.datasets as dset
import csv
from config import PARAMETERS



#Esta función nos devuelve la ruta del directorio en el que estamos trabajando
base_dir = os.getcwd()

#Creamos una clase en la que definimos las rutas de los directorios de los cuales extraemos las imágenes
class Config():
   training_dir = os.path.join(base_dir, PARAMETERS.validation_dir)
   testing_dir = os.path.join(base_dir, PARAMETERS.centro_geom_dir)


#Creamos un vector en el que cada elemento representa una habitación del edificio
folder_dataset_val = dset.ImageFolder(root=Config.training_dir)
rooms_val = folder_dataset_val.classes
folder_dataset_comp = dset.ImageFolder(root=Config.testing_dir)
rooms_comp = folder_dataset_comp.classes

index = -1


#Generamos el archivo .csv en modo escritura
with open(base_dir + PARAMETERS.val_csv_dir, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Img0", "Img1", "Indice"])


    for habitacion in rooms_val:
        ruta_habitacion = os.path.join(Config.training_dir, habitacion)             #obtenemos la ruta de la habitación de la iteración actual
        lista_imagenes = os.listdir(ruta_habitacion)                                 #para cada habitación, creamos una lista con las rutas de todas las imágenes de esa habitación
        if index == 8:
            index = 0
        else:
            index = index + 1


        for imagen_val in lista_imagenes:
            ruta_imagen_val = os.path.join(ruta_habitacion, imagen_val)                     #para cada imagen de la habitación actual, obtenemos su ruta

            for habitacion_comp in rooms_comp:
                ruta_habitacion_comp = os.path.join(Config.testing_dir, habitacion_comp)
                lista_imagenes_comp = os.listdir(ruta_habitacion_comp)
                for imagen_comp in lista_imagenes_comp:
                    ruta_imagen_comp = os.path.join(ruta_habitacion_comp, imagen_comp)
                    writer.writerow([ruta_imagen_val,ruta_imagen_comp,index])



