
#importamos librerías
import os
import torchvision.datasets as dset
import csv
import random
import numpy as np
from config import PARAMETERS
base_dir=os.getcwd()

#definimos rutas con las que vamos a trabajar
class Config():
   training_dir= os.path.join(base_dir,'Entrenamiento_Completo')
   # testing_dir= os.path.join(base_dir,'TestDataset')

#creamos un vector con los nombres de las habitaciones
folder_dataset = dset.ImageFolder(root=Config.training_dir)
rooms = folder_dataset.classes

#creamos un vector con las rutas de cada una de las habitaciones
rutas_habitacion=[]
for habitacion in rooms:
    rutas_habitacion.append(os.path.join(Config.training_dir, habitacion))

#obtenemos la proporción de imgs de cada habitación respecto del total de imgs
n_images = []
for habitacion in rooms:
    ruta_habitacion = os.path.join(Config.training_dir,habitacion)
    n_images.append(len(os.listdir(ruta_habitacion)))
n_images = np.array(n_images)
suma = n_images.sum()
probabilities_anc = n_images / suma


#creamos un csv que guarde las combinaciones de 3 imgs que vamos a crear (anchor, positive, negative)
with open( base_dir + '/ListaEntrenamiento_Completa.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Anchor","Positive","Negative"])

    for lote in range(0,PARAMETERS.num_iteraciones):

        #obtenemos la ruta de una img anchor escogida al azar entre las carpetas Easy positive de las 9 habitaciones
        number_rooms = np.arange(0, 9)
        room_anc_choice = np.random.choice(number_rooms, 1, p=probabilities_anc)

        room_anc = rutas_habitacion[room_anc_choice[0]]
        list_imgs_anc=os.listdir(room_anc)

        anc = random.choice(list_imgs_anc)
        anc= os.path.join(room_anc,anc)

        #para cada img anchor, crearemos 16 combinaciones distintas de imgs positive y negative

        for i in range(0,16):


                room_pos = rutas_habitacion[room_anc_choice[0]]
                list_imgs_pos = os.listdir(room_pos)

                pos = random.choice(list_imgs_pos)
                pos = os.path.join(room_pos, pos)

                while pos==anc:
                    pos = random.choice(list_imgs_pos)
                    pos = os.path.join(room_pos, pos)



            #ESCOGEMOS LA IMG NEGATIVE





                #escogemos al azar de qué habitación toma el ejemplo negative
                number_rooms_neg = np.arange(0, len(rutas_habitacion))
                probabilities_neg = 1 / (len(rutas_habitacion)) * np.ones(len(rutas_habitacion))

                room_neg_choice = np.random.choice(number_rooms_neg, 1, p=probabilities_neg)
                while room_neg_choice[0] == room_anc_choice[0]:
                    number_rooms = np.arange(0, 9)
                    room_neg_choice = np.random.choice(number_rooms_neg, 1, p=probabilities_neg)
                #dentro de la habitación escogida, tomamos una img al azar para el ejemplo negative
                room_neg = rutas_habitacion[room_neg_choice[0]]
                list_imgs_neg = os.listdir(room_neg)

                neg = random.choice(list_imgs_neg)
                neg = os.path.join(room_neg, neg)



                writer.writerow([anc,pos,neg])

