import torchvision.datasets as dset
import os
import csv

base_dir=os.getcwd()

img_dir= os.path.join(base_dir,'TrainingDataset3/Entrenamiento')
folder_dataset = dset.ImageFolder(root=img_dir)
rooms = folder_dataset.classes

indice_hab=0

with open( base_dir + '/ImgsRepresentativasDatasetCompleto.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Habitacion","Indice_Habitacion","Imagen","Indice_Imagen","Coord_X","Coord_Y" ])


    for habitacion in rooms:
        ruta_habitacion = os.path.join(img_dir, habitacion)
        lista_imagenes = os.listdir(ruta_habitacion)
        centro_x=0
        centro_y=0

        for img in lista_imagenes:
            ruta_imagen = os.path.join(ruta_habitacion, img)

            x_index = ruta_imagen.index('_x')
            y_index = ruta_imagen.index('_y')
            a_index = ruta_imagen.index('_a')

            coord_x=ruta_imagen[x_index+2:y_index]
            coord_y=ruta_imagen[y_index+2:a_index]

            coord_x=float(coord_x)
            coord_y=float(coord_y)

            centro_x+=coord_x
            centro_y+=coord_y
        centro_x/=len(lista_imagenes)
        centro_y/=len(lista_imagenes)


        dist_min=1000000
        indice_centro=0
        indice_img=0


        for img in lista_imagenes:
            ruta_imagen = os.path.join(ruta_habitacion, img)

            x_index = ruta_imagen.index('_x')
            y_index = ruta_imagen.index('_y')
            a_index = ruta_imagen.index('_a')

            coord_x=ruta_imagen[x_index+2:y_index]
            coord_y=ruta_imagen[y_index+2:a_index]

            coord_x=float(coord_x)
            coord_y=float(coord_y)

            dist= ((coord_x-centro_x)**2+(coord_y-centro_y)**2)**0.5


            if dist<dist_min:
                dist_min=dist
                indice_centro=indice_img
                ruta_centro=ruta_imagen
                img_x=coord_x
                img_y=coord_y

            indice_img+=1


        writer.writerow([habitacion, indice_hab, ruta_centro, indice_centro, img_x, img_y])

        print("CENTRO HABITACIÓN '", habitacion, "' : ")
        print("Indice: ", indice_centro)
        print("x= ", img_x)
        print("y= ", img_y)
        print("Centro geometrico=(", centro_x, ",", centro_y, ")")
        print("distancia= ",dist_min)

        indice_hab+=1






