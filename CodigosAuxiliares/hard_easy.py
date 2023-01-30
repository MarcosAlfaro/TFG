import torchvision.datasets as dset
import os
import csv

base_dir=os.getcwd()

img_dir= os.path.join(base_dir,'Entrenamiento')
folder_dataset = dset.ImageFolder(root=img_dir)
rooms = folder_dataset.classes

def coordenadas_inicio(habitacion):
    ruta_habitacion = os.path.join(img_dir,habitacion)
    lista_imagenes = os.listdir(ruta_habitacion)

    ruta_imagen = os.path.join(ruta_habitacion, lista_imagenes[0])

    x_index = ruta_imagen.index('_x')
    y_index = ruta_imagen.index('_y')
    a_index = ruta_imagen.index('_a')

    coord_x = ruta_imagen[x_index + 2:y_index]
    coord_y = ruta_imagen[y_index + 2:a_index]

    coord_x = float(coord_x)
    coord_y = float(coord_y)
    return coord_x, coord_y

def positives(coord_inicio_x, coord_inicio_y, habitacion):
    ruta_habitacion = os.path.join(img_dir, habitacion)
    lista_imagenes = os.listdir(ruta_habitacion)

    print("NEGATIVES ", habitacion)
    ind = 0
    for img in lista_imagenes:
        ruta_imagen = os.path.join(ruta_habitacion, img)

        x_index = ruta_imagen.index('_x')
        y_index = ruta_imagen.index('_y')
        a_index = ruta_imagen.index('_a')

        coord_x = ruta_imagen[x_index + 2:y_index]
        coord_y = ruta_imagen[y_index + 2:a_index]

        coord_x = float(coord_x)
        coord_y = float(coord_y)

        dist = (coord_x - coord_inicio_x) ** 2 + (coord_y - coord_inicio_y) ** 2

        if dist <= 1:
            print("Index ", ind, ": HARD NEGATIVE")
            writer.writerow([habitacion, ind, "Neg", "Hard", coord_x, coord_y, dist])
        else:
            print("EASY NEGATIVE")
            writer.writerow([habitacion, ind, "Neg", "Easy", coord_x, coord_y, dist])
        ind = ind + 1

def negatives(coord_inicio_x, coord_inicio_y, habitacion):
    ruta_habitacion = os.path.join(img_dir, habitacion)
    lista_imagenes = os.listdir(ruta_habitacion)

    print("NEGATIVES ",habitacion)
    indice=0
    for img in lista_imagenes:
        ruta_imagen = os.path.join(ruta_habitacion, img)

        x_index = ruta_imagen.index('_x')
        y_index = ruta_imagen.index('_y')
        a_index = ruta_imagen.index('_a')

        coord_x = ruta_imagen[x_index + 2:y_index]
        coord_y = ruta_imagen[y_index + 2:a_index]

        coord_x = float(coord_x)
        coord_y = float(coord_y)

        dist = (coord_x - coord_inicio_x) ** 2 + (coord_y - coord_inicio_y) ** 2

        if dist <= 1:
            print("Index ", indice, ": HARD NEGATIVE")
            writer.writerow([habitacion, indice, "Neg", "Hard", coord_x, coord_y, dist])
        else:
            print("EASY NEGATIVE")
            writer.writerow([habitacion, indice, "Neg", "Easy", coord_x, coord_y, dist])
        indice=indice+1

    return

coord_x_ST_A, coord_y_ST_A= coordenadas_inicio('ST-A')
coord_x_TL_A, coord_y_TL_A= coordenadas_inicio('TL-A')


with open( base_dir + '\\ST_A.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Hab","Indice img","Pos_Neg","Hard_Easy","Coord_X","Coord_Y","Distancia"])

    """HABITACIÓN ST-A"""

    ruta_habitacion = os.path.join(img_dir,'ST-A')
    lista_imagenes = os.listdir(ruta_habitacion)
    print("POSITIVES ST-A")
    ind=0
    for img in lista_imagenes:
        ruta_imagen = os.path.join(ruta_habitacion, img)

        x_index = ruta_imagen.index('_x')
        y_index = ruta_imagen.index('_y')
        a_index = ruta_imagen.index('_a')

        coord_x=ruta_imagen[x_index+2:y_index]
        coord_y=ruta_imagen[y_index+2:a_index]

        coord_x=float(coord_x)
        coord_y=float(coord_y)


        dist1 = ((coord_x - coord_x_ST_A) ** 2 + (coord_y - coord_y_ST_A) ** 2) ** 0.5
        dist2 = ((coord_x - coord_x_TL_A) ** 2 + (coord_y - coord_y_TL_A) ** 2) ** 0.5


        if dist1<=1:
            print("Index ",ind,": HARD POSITIVE")
            writer.writerow(['PA-A', ind, "Pos", "Hard", coord_x, coord_y, dist1])
        elif dist2<=1:
            print("Index ", ind, ": HARD POSITIVE")
            writer.writerow(['PA-A', ind, "Pos", "Hard", coord_x, coord_y, dist2])

        else:
            print("EASY POSITIVE")
            writer.writerow(['PA-A', ind, "Pos", "Easy", coord_x, coord_y,min(dist1,dist2)])
        ind = ind + 1






    negatives(coord_x_ST_A,coord_y_ST_A,'CR-A')
    negatives(coord_x_TL_A, coord_y_TL_A, 'TL-A')