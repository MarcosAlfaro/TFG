
# importamos librerías
import os
import torchvision.datasets as dset
import csv
import random
import numpy as np
from config import PARAMETERS
base_dir = os.getcwd()

# definimos rutas con las que vamos a trabajar

class Config():
   training_dir = os.path.join(base_dir, 'Entrenamiento_Hard_Easy')
   testing_dir = os.path.join(base_dir, 'TestDataset')


# creamos un vector con los nombres de las habitaciones
folder_dataset = dset.ImageFolder(root=Config.training_dir)
rooms = folder_dataset.classes

# creamos un vector con las rutas de cada una de las habitaciones
rutas_habitacion=[]
for habitacion in rooms:
    rutas_habitacion.append(os.path.join(Config.training_dir, habitacion))

# obtenemos la proporción de imgs de cada habitación respecto del total de imgs
n_images = []
for habitacion in rooms:
    ruta_habitacion = os.path.join(Config.training_dir, habitacion, habitacion)
    n_images.append(len(os.listdir(ruta_habitacion)))
n_images = np.array(n_images)
suma = n_images.sum()
probabilities_anc = n_images / suma


# creamos un csv que guarde las combinaciones de 3 imgs que vamos a crear (anchor, positive, negative)
with open(base_dir + '/ListaEntrenamiento_HardEasy_v2.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Anchor", "Positive", "Negative", "Index_pos", "Index_neg"])

    for lote in range(0, PARAMETERS.num_iteraciones):

        # obtenemos la ruta de una img anchor escogida al azar entre las carpetas Easy positive de las 9 habitaciones
        number_rooms = np.arange(0, 9)
        room_anc_choice = np.random.choice(number_rooms, 1, p=probabilities_anc)

        room_anc = (os.path.join(rutas_habitacion[room_anc_choice[0]], 'Easy_pos'))
        list_imgs_anc = os.listdir(room_anc)

        anc = random.choice(list_imgs_anc)
        anc = os.path.join(room_anc, anc)

        # para cada img anchor, crearemos 16 combinaciones distintas de imgs hard/easy positive y hard/easy negative
        # escogemos al azar si la imagen positive es hard o easy, y cuál de las imgs escogemos dentro de cada grupo
        # escogemos al azar si la img negative es hard o easy, la habitación de la cual sacamos la img negative y la img de esa habitación
        for i in range(0, 16):

            # ESCOGEMOS LA IMG POSITIVE
            # escogemos al azar si la img es hard o easy, con una proporción definida por el usuario
            select_hard_easy_pos = np.arange(0, 2)
            hard_easy_pos_choice = np.random.choice(select_hard_easy_pos, 1, p=[PARAMETERS.hard_pos, 1-PARAMETERS.hard_pos]) #0:hard, 1:easy

            # si la img es hard, cogemos una img al azar dentro de la carpeta Hard positive de la habitación de la img anchor
            if hard_easy_pos_choice == 0:
                room_pos = (os.path.join(rutas_habitacion[room_anc_choice[0]], 'Hard_pos'))
                list_imgs_pos = os.listdir(room_pos)

                pos = random.choice(list_imgs_pos)
                pos = os.path.join(room_pos, pos)

                index_pos = 0.25

            # si la img es easy, cogemos una img al azar dentro de la carpeta Easy positive de la habitación de la img anchor
            else:
                room_pos = (os.path.join(rutas_habitacion[room_anc_choice[0]], 'Easy_pos'))
                list_imgs_pos = os.listdir(room_pos)

            # nos aseguramos de que no coja la misma img como anchor y positive
                pos = random.choice(list_imgs_pos)
                pos = os.path.join(room_pos, pos)
                while pos == anc:
                    pos = random.choice(list_imgs_pos)
                    pos = os.path.join(room_pos, pos)

                index_pos = 0

            # ESCOGEMOS LA IMG NEGATIVE
            # escogemos al azar si la img es hard o easy, con una proporción definida por el usuario
            # select_hard_easy_neg = np.arange(0, 2)
            # hard_easy_neg_choice = np.random.choice(select_hard_easy_neg, 1, p=[PARAMETERS.hard_neg, 1 - PARAMETERS.hard_neg])  # 0:hard, 1:easy

            # si la img es hard, creamos una lista con las habitaciones que colindan con la habitación de la img anchor
            if i == 0:
                rooms_hard_neg = (os.path.join(rutas_habitacion[room_anc_choice[0]], 'Hard_neg'))
                list_rooms_hard_neg = os.listdir(rooms_hard_neg)

                # escogemos al azar de qué habitación toma el ejemplo negative
                number_rooms_hard_neg = np.arange(0, len(os.listdir(rooms_hard_neg)))
                probabilities_hard_neg = 1 / (len(os.listdir(rooms_hard_neg))) * np.ones(len(os.listdir(rooms_hard_neg)))

                room_neg_choice = np.random.choice(number_rooms_hard_neg, 1, p=probabilities_hard_neg)

                # dentro de la habitación escogida, tomamos una img al azar para el ejemplo negative
                room_neg = list_rooms_hard_neg[room_neg_choice[0]]
                list_imgs_neg = os.listdir(os.path.join(rooms_hard_neg, room_neg))

                neg = random.choice(list_imgs_neg)
                neg = os.path.join(rooms_hard_neg, room_neg, neg)

                index_neg = 0.75

                writer.writerow([anc, pos, neg, index_pos, index_neg])
            # si la img es easy, tomamos una img al azar de la carpeta Easy positive de una cualquiera de las habitaciones restantes
            else:

                # escogemos una habitación al azar, comprobando que no es la misma que la de la img anchor
                number_rooms_easy_neg = np.arange(0, 9)
                room_neg_choice = np.random.choice(number_rooms_easy_neg, 1, p=probabilities_anc)
                while room_neg_choice[0] == room_anc_choice[0]:
                    number_rooms = np.arange(0, 9)
                    room_neg_choice = np.random.choice(number_rooms_easy_neg, 1, p=probabilities_anc)

                # dentro de la habitación escogida, tomamos una img al azar para el ejemplo negative
                room_neg = (os.path.join(rutas_habitacion[room_neg_choice[0]], 'Easy_pos'))
                list_imgs_neg = os.listdir(room_neg)

                neg = random.choice(list_imgs_neg)
                neg = os.path.join(room_neg, neg)

                index_neg = 1

                writer.writerow([anc, pos, neg, index_pos, index_neg])
