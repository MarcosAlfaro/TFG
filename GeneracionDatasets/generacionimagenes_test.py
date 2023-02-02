


#Importamos las librerías que son necesarias

import os
import torchvision.datasets as dset
import csv
from config import PARAMETERS


# Esta función nos devuelve la ruta del directorio en el que estamos trabajando
base_dir = PARAMETERS.base_dir

# Creamos una clase en la que definimos las rutas de los directorios de los cuales extraemos las imágenes
class Config():
   test_dir = os.path.join(base_dir, 'TestCloudy')
   comp_dir = os.path.join(base_dir, 'ImagenesCentroGeometrico')


# Creamos un vector en el que cada elemento representa una habitación del edificio
folder_dataset_test = dset.ImageFolder(root=Config.test_dir)
rooms_test = folder_dataset_test.classes
folder_dataset_comp = dset.ImageFolder(root=Config.comp_dir)
rooms_comp = folder_dataset_comp.classes

index = -1


#Generamos el archivo .csv en modo escritura
with open(base_dir + '/TestNublado.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["ImgTest", "ImgComp", "Indice"])


    for habitacion_test in rooms_test:
        ruta_habitacion_test = os.path.join(Config.test_dir, habitacion_test)             #obtenemos la ruta de la habitación de la iteración actual
        lista_imagenes = os.listdir(ruta_habitacion_test)                                 #para cada habitación, creamos una lista con las rutas de todas las imágenes de esa habitación
        if index == 8:
            index = 0
        else:
            index = index + 1


        for imagen_test in lista_imagenes:
            ruta_imagen_test = os.path.join(ruta_habitacion_test, imagen_test)                     #para cada imagen de la habitación actual, obtenemos su ruta

            for habitacion_comp in rooms_comp:
                ruta_habitacion_comp = os.path.join(Config.comp_dir, habitacion_comp)
                lista_imagenes_comp = os.listdir(ruta_habitacion_comp)
                for imagen_comp in lista_imagenes_comp:
                    ruta_imagen_comp = os.path.join(ruta_habitacion_comp, imagen_comp)
                    writer.writerow([ruta_imagen_test,ruta_imagen_comp,index])



