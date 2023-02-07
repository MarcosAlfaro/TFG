

# Importamos las librerías que son necesarias

import os
import torchvision.datasets as dset
import csv
from config import PARAMETERS
import numpy as np
import random

# Esta función nos devuelve la ruta del directorio en el que estamos trabajando
base_dir = PARAMETERS.base_dir

# Creamos una clase en la que definimos las rutas de los directorios de los cuales extraemos las imágenes
class Config():
    training_dir = os.path.join(base_dir, PARAMETERS.training_dir)
    # testing_dir = os.path.join(base_dir, 'TestDataset')


# Creamos un vector en el que cada elemento representa una habitación del edificio
folder_dataset = dset.ImageFolder(root=Config.training_dir)
rooms = folder_dataset.classes


rutas_habitacion = []
for habitacion in rooms:
    rutas_habitacion.append(os.path.join(Config.training_dir, habitacion))

n_images = []
for habitacion in rooms:
    ruta_habitacion = os.path.join(Config.training_dir, habitacion)
    n_images.append(len(os.listdir(ruta_habitacion)))
n_images = np.array(n_images)
suma = n_images.sum()
probabilities = n_images / suma

# Generamos el archivo .csv en modo escritura, creando las tres columnas (Anchor,Positive,Negative)
with open(base_dir + PARAMETERS.train_csv_dir, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Img0", "Img1", "Label"])

    for lote in range(0, PARAMETERS.num_iteraciones):

        # obtenemos la ruta de una img anchor escogida al azar entre las carpetas Easy positive de las 9 habitaciones
        number_rooms = np.arange(0, 9)
        room_img0_choice = np.random.choice(number_rooms, 1, p=probabilities)

        room_img0 = (os.path.join(rutas_habitacion[room_img0_choice[0]]))
        list_imgs_img0 = os.listdir(room_img0)

        img0 = random.choice(list_imgs_img0)
        img0 = os.path.join(room_img0, img0)

        select_same_room = np.arange(0, 2)
        same_room_choice = np.random.choice(select_same_room, 1,
                                            p=[1.0 - PARAMETERS.same_room, PARAMETERS.same_room])  # 0:same room, 1:different room

        if same_room_choice == 0:
            room_img1 = (os.path.join(rutas_habitacion[room_img0_choice[0]]))
            list_imgs1 = list_imgs_img0

            img1 = random.choice(list_imgs1)
            img1 = os.path.join(room_img1, img1)

            while img0 == img1:
                img1 = random.choice(list_imgs1)
                img1 = os.path.join(room_img1, img1)

        else:

            number_rooms = np.arange(0, 9)
            room_img1_choice = np.random.choice(number_rooms, 1, p=probabilities)

            while room_img1_choice == room_img0_choice:
                room_img1_choice = np.random.choice(number_rooms, 1, p=probabilities)

            room_img1 = (os.path.join(rutas_habitacion[room_img1_choice[0]]))
            list_imgs1 = os.listdir(room_img0)

            img1 = random.choice(list_imgs1)
            img1 = os.path.join(room_img1, img1)

        label = same_room_choice

        writer.writerow([img0, img1, label])

