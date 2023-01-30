import os
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import PIL.ImageOps
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import seaborn as sn
import time
import csv
from config import PARAMETERS
import pickle

#indica que si el computador dispone de cuda, trabajaremos con cuda, si no, se hará en la cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

def imshow(img, text=None, should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

#esta función genera una matriz de confusión entre las habitaciones del edificio
def display_confusion_matrix(cm, illumination,plt_name, net, gd):

    # os.makedirs(EXPCONFIG.base_dir + 'Run0_' + str(video_number) + '/smsd_results', exist_ok=True)
    plt.figure(figsize=(9, 5), dpi=120)
    df_cm = pd.DataFrame(cm, index=['1P0-A', '2P01-A', '2P02-A', 'CR-A', 'KT-A', 'LO-A', 'PA-A', 'ST-A', 'TL-A'], columns=['1P0-A', '2P01-A', '2P02-A', 'CR-A', 'KT-A', 'LO-A', 'PA-A', 'ST-A', 'TL-A'])

    sn.set(font_scale=1)  # for label size
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 12}, cmap='Blues', fmt='d', cbar=False)  # font size

    # name_title = 'Net: ' + net + ' ' + illumination + ' ' + gd
    # plt.figure(figsize=(9, 3), dpi=80)
    # plt.title(name_title)
    # Las filas son la True class, y las columnas las predicciones
    plt.ylabel('True class')
    plt.xlabel('Class predicted')
    plt.savefig(plt_name, dpi=400)

    plt.show()
    # plt_name = EXPCONFIG.base_dir + 'Run0_' + str(video_number) + '/smsd_results/cm.png'

    # plt.clf()

#Obtenemos la ruta del directorio en el que estamos trabajando
base_dir= os.getcwd()


#en esta clase definimos parámetros que utilizaremos durante el programa
# class Config():
#     # ground_dir = os.path.join(base_dir,'GroundTruthDataset')
#     test_cloudy_dir = PARAMETERS.test_cloudy_dir
#     test_sunny_dir = PARAMETERS.test_sunny_dir
#     test_night_dir = PARAMETERS.test_night_dir
#     test_batch_size = PARAMETERS.test_batch_size
#     # train_number_epochs = 5


#en esta clase obtenemos las combinaciones de imágenes de test en condiciones nubladas con las imágenes representativas de cada habitación
#guardadas en el archivo 'TestNublado.csv' generado en el programa generacionimagenes_test.py
#y las transformamos a formato RGB
class GeneracionDatasetCloudy(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        # self.imageFolderGroundTruth = imageFolderGroundTruth
        self.transform = transform
        self.should_invert = should_invert

        fichero_test = pd.read_csv(base_dir+PARAMETERS.cloudy_csv_dir)

        self.list_img_test = fichero_test['ImgTest']
        self.list_img_comp = fichero_test['ImgComp']
        self.indice = fichero_test['Indice']

    def __getitem__(self, index):



        img0 = self.list_img_test[index]
        img1 = self.list_img_comp[index]
        indice = self.indice[index]

        img_test = Image.open(img0)
        img_test = img_test.convert("RGB")

        img_comp = Image.open(img1)
        img_comp = img_comp.convert("RGB")

        if self.should_invert:
            img_test = PIL.ImageOps.invert(img_test)
            img_comp = PIL.ImageOps.invert(img_comp)

        if self.transform is not None:
            img_test = self.transform(img_test)
            img_comp = self.transform(img_comp)


        return img_test, img_comp, indice

    def __len__(self):
        return len(self.list_img_test)


#en esta clase obtenemos las combinaciones de imágenes de test en condiciones soleadas con las imágenes representativas de cada habitación
#guardadas en el archivo 'TestSoleado.csv' generado en el programa generacionimagenes_test.py
#y las transformamos a formato RGB
class GeneracionDatasetSunny(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        # self.imageFolderGroundTruth = imageFolderGroundTruth
        self.transform = transform
        self.should_invert = should_invert

        fichero_test = pd.read_csv(base_dir+PARAMETERS.sunny_csv_dir)

        self.list_img_test = fichero_test['ImgTest']
        self.list_img_comp = fichero_test['ImgComp']
        self.indice = fichero_test['Indice']

    def __getitem__(self, index):



        img0 = self.list_img_test[index]
        img1= self.list_img_comp[index]
        indice = self.indice[index]

        img_test = Image.open(img0)
        img_test = img_test.convert("RGB")

        img_comp = Image.open(img1)
        img_comp= img_comp.convert("RGB")

        if self.should_invert:
            img_test = PIL.ImageOps.invert(img_test)
            img_comp = PIL.ImageOps.invert(img_comp)

        if self.transform is not None:
            img_test = self.transform(img_test)
            img_comp = self.transform(img_comp)

        return img_test, img_comp, indice

    def __len__(self):
        return len(self.list_img_test)


#en esta clase obtenemos las combinaciones de imágenes de test en condiciones de noche con las imágenes representativas de cada habitación
#guardadas en el archivo 'TestNoche.csv' generado en el programa generacionimagenes_test.py
#y las transformamos a formato RGB
class GeneracionDatasetNight(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

        fichero_test = pd.read_csv(base_dir+PARAMETERS.night_csv_dir)

        self.list_img_test = fichero_test['ImgTest']
        self.list_img_comp = fichero_test['ImgComp']
        self.indice = fichero_test['Indice']

    def __getitem__(self, index):



        img0 = self.list_img_test[index]
        img1 = self.list_img_comp[index]
        indice = self.indice[index]

        img_test = Image.open(img0)
        img_test = img_test.convert("RGB")

        img_comp = Image.open(img1)
        img_comp = img_comp.convert("RGB")

        if self.should_invert:
            img_test = self.transform(img_test)
            img_comp = self.transform(img_comp)

        if self.transform is not None:
            img_test = self.transform(img_test)
            img_comp = self.transform(img_comp)

        return img_test, img_comp, indice

    def __len__(self):
        return len(self.list_img_test)


#Se carga la red 'vgg16' en la cpu
vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)


#En esta clase se define la estructura de la red neuronal, cómo están conectadas las capas entre sí,
#cómo se transmite la información de los inputs a los outputs,etc.
class TripletNetwork(nn.Module):

    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.cnn1 = vgg16.features
        if PARAMETERS.do_dataparallel == True:
            self.cnn1 = nn.DataParallel(self.cnn1)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 16))
        if PARAMETERS.do_dataparallel==True:
            self.avgpool = nn.DataParallel(self.avgpool)
        self.fc1 = nn.Sequential(
          nn.Linear(4*16*512, 500),
          nn.ReLU(inplace=True),

          nn.Linear(500, 500),
          nn.ReLU(inplace=True),

          nn.Linear(500, 5))
        if PARAMETERS.do_dataparallel == True:
            self.fc1 = nn.DataParallel(self.fc1)

    def forward_once(self, x):
        verbose = False

        if verbose:
            print("Input: ", x.size())

        output = self.cnn1(x)

        if verbose:
            print("Output matricial: ", output.size())

        output = self.avgpool(output)
        if verbose:
            print("Output avgpool: ", output.size())
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        return output

    #NOTA: esta función de la clase es muy importante porque indica cómo hemos planteado el problema:
    #En lugar de crear tres redes neuronales en la definición de la red, se creará una única red simple
    #y se llamará a la red varias veces, según el proceso que se desee realizar
    #Para el entrenamiento, se llamará a la red tres veces, y para la validación se hará en dos ocasiones
    def forward(self, input1):
        output1 = self.forward_once(input1)
        return output1

#Llamamos a la red
net = torch.load(PARAMETERS.test_net).to(device)  # Carga el modelo

print(net)


folder_dataset_test_cloudy = dset.ImageFolder(root=base_dir+PARAMETERS.test_cloudy_dir)
folder_dataset_test_sunny = dset.ImageFolder(root=base_dir+PARAMETERS.test_sunny_dir)
folder_dataset_test_night = dset.ImageFolder(root=base_dir+PARAMETERS.test_night_dir)
# folder_GroundTruth = dset.ImageFolder(root=Config.ground_dir)

print("La longitud de la carpeta test para cloudy: ", folder_dataset_test_cloudy)
print("La longitud de la carpeta test para sunny: ", folder_dataset_test_sunny)
print("La longitud de la carpeta test para night: ", folder_dataset_test_night)
# print("La longitud de la carpeta ground truth: ", folder_GroundTruth)


with open( base_dir + PARAMETERS.test_csv_dir, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Cloudy","Night","Sunny" ])

    print('TESTEO PARA NUBLADO')

    #Llamamos a la clase GeneracionDatasetCloudy para generar un dataset con las imágenes destinadas al testeo de la red
    #en condiciones de nublado
    test_dataset_cloudy = GeneracionDatasetCloudy(imageFolderDataset=folder_dataset_test_cloudy,
                                             transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                           transforms.ToTensor()
                                                                           ])
                                             , should_invert=False)

    #Cargamos los lotes de imágenes a la cpu
    test_dataloader_cloudy = DataLoader(test_dataset_cloudy, num_workers=PARAMETERS.workers_test, batch_size=PARAMETERS.test_batch_size, shuffle=PARAMETERS.shuffle_test)

    cont=0
    cont_correct=0
    hab_real =[]
    hab_pred =[]

    time_idx = 0
    sum_time = 0

    for i, data_cloudy in enumerate(test_dataloader_cloudy, 0):

         time_idx += 1
         start = time.time()

          # Asignamos las imágenes cargadas en cada lote a variables con las que trabajaremos
         img_test, img_comp, indice = data_cloudy
         img_test, img_comp, indice = img_test.to(device), img_comp.to(device), indice.to(device)



          # Llamamos a la red dos veces,una para cada imagen (imagen de test, imagen representativa)
         output1 = net(img_test)
         output2 = net(img_comp)

          #Calculamos la distancia euclídea de los 9 pares de imágenes y la distancia menor nos da la habitación predicha
         euclidean_distance = F.pairwise_distance(output1, output2)
         room_predicted = torch.argmin(euclidean_distance)

         end = time.time()
         time_execution = (end - start)

         # confidence = min(euclidean_distance)
         hab_real.append(indice[0].cpu().numpy())
         hab_pred.append(room_predicted.cpu().numpy())

         # Si la habitación real coincide con la habitación predicha, se considera un acierto de la red
         if room_predicted == indice[0]:
             cont_correct = cont_correct + 1
        #else:
              #print('Habitación predicha: ' + str(room_predicted) + ', habitación real: ' + str(room_test[0]))
              #print(euclidean_distance)
         cont = cont + 1

         if time_idx > 1:
             sum_time = time_execution + sum_time


    cm = confusion_matrix(hab_real, hab_pred)
    with open("matriz_confusion_cloudy_exp6c", "wb") as fp:
        pickle.dump(cm, fp)
    file = open('matriz_confusion_cloudy_exp6c', 'rb')
    cm = pickle.load(file)
    print("MATRIZ DE CONFUSIÓN TEST NUBLADO")
    print(cm)
    display_confusion_matrix(cm=cm, illumination='cloudy',plt_name='matriz_confusion_cloudy_exp6c.png', net=PARAMETERS.test_net, gd='gd_silhouette_net0')
    accuracy_cloudy=(cont_correct/cont)*100
    print('La precisión para nublado es: ',accuracy_cloudy)
    min_time=sum_time//60
    sec_time=sum_time%60
    print("Tiempo de ejecución para test de nublado: ", min_time,"min ", sec_time, "s")


    # Testeo para night

    print('TESTEO PARA NOCHE')
    #Llamamos a la clase GeneracionDatasetNight para generar un dataset con las imágenes destinadas al testeo de la red
    #en condiciones de noche
    test_dataset_night = GeneracionDatasetNight(imageFolderDataset=folder_dataset_test_night,
                                            transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False)

    #Cargamos los lotes de imágenes a la cpu
    test_dataloader_night = DataLoader(test_dataset_night, num_workers=PARAMETERS.workers_test, batch_size=PARAMETERS.test_batch_size, shuffle=PARAMETERS.shuffle_test)

    cont = 0
    cont_correct = 0
    hab_real = []
    hab_pred = []

    time_idx = 0
    sum_time = 0

    for i, data_night in enumerate(test_dataloader_night, 0):
        time_idx += 1
        start = time.time()
        # Asignamos las imágenes cargadas en cada lote a variables con las que trabajaremos
        img_test, img_comp, indice = data_night
        img_test, img_comp, indice = img_test.to(device), img_comp.to(device), indice.to(device)



        # Llamamos a la red dos veces,una para cada imagen (imagen de test, imagen representativa)
        output1 = net(img_test)
        output2 = net(img_comp)

        # Calculamos la distancia euclídea de los 9 pares de imágenes y la distancia menor nos da la habitación predicha
        euclidean_distance = F.pairwise_distance(output1, output2)
        room_predicted = torch.argmin(euclidean_distance)

        end = time.time()
        time_execution = (end - start)


        hab_real.append(indice[0].cpu().numpy())
        hab_pred.append(room_predicted.cpu().numpy())

        # Si la habitación real coincide con la habitación predicha, se considera un acierto de la red
        if room_predicted == indice[0]:
            cont_correct = cont_correct + 1
        # else:
        #     print('Habitación predicha: ' + str(room_predicted) + ', habitación real: ' + str(room_test[0]) )
        #     print(euclidean_distance)

        # if confidence < 0.5:
        #     if room_predicted == room_test[0]:
        #         cont_correct = cont_correct + 1
        #     # else:
        #     #     print('Habitación predicha: ' + str(room_predicted) + ', habitación real: ' + str(room_test[0]) )
        #     #     print(euclidean_distance)
        # else:
        #     no_me_localizo = no_me_localizo + 1

        cont = cont + 1

        if time_idx > 1:
            sum_time = time_execution + sum_time


    cm = confusion_matrix(hab_real, hab_pred)
    with open("matriz_confusion_night_exp6c", "wb") as fp:
        pickle.dump(cm, fp)
    file = open('matriz_confusion_night_exp6c', 'rb')
    matriz_night = pickle.load(file)
    print("MATRIZ DE CONFUSIÓN TEST NOCHE")
    print(matriz_night)
    display_confusion_matrix(cm=cm, illumination='night',plt_name='matriz_confusion_night_exp6c.png', net=PARAMETERS.test_net, gd='gd_silhouette_net0')

    accuracy_night=(cont_correct/cont)*100
    print('La precisión en condiciones de noche es: ',accuracy_night)
    min_time=sum_time//60
    sec_time=sum_time%60
    print("Tiempo de ejecución para test de noche: ", min_time,"min ", sec_time, "s")



    print('TESTEO PARA SOLEADO')

    #Llamamos a la clase GeneracionDatasetSunny para generar un dataset con las imágenes destinadas al testeo de la red
    #en condiciones soleadas
    test_dataset_sunny = GeneracionDatasetSunny(imageFolderDataset=folder_dataset_test_sunny,
                                            transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                          transforms.ToTensor()
                                                                          ])
                                            , should_invert=False)

    #Cargamos los lotes de imágenes a la cpu
    test_dataloader_sunny = DataLoader(test_dataset_sunny, num_workers=PARAMETERS.workers_test, batch_size=PARAMETERS.test_batch_size, shuffle=PARAMETERS.shuffle_test)

    cont = 0
    cont_correct = 0
    hab_real = []
    hab_pred = []

    time_idx = 0
    sum_time = 0

    for i, data_sunny in enumerate(test_dataloader_sunny, 0):

        time_idx += 1
        start = time.time()

        # Asignamos las imágenes cargadas en cada lote a variables con las que trabajaremos
        img_test, img_comp, indice = data_sunny
        img_test, img_comp, indice = img_test.to(device), img_comp.to(device), indice.to(device)



        # Llamamos a la red dos veces,una para cada imagen (imagen de test, imagen representativa)
        output1 = net(img_test)
        output2 = net(img_comp)

        # Calculamos la distancia euclídea de los 9 pares de imágenes y la distancia menor nos da la habitación predicha
        euclidean_distance = F.pairwise_distance(output1, output2)
        room_predicted = torch.argmin(euclidean_distance)

        end = time.time()
        time_execution = (end - start)

        hab_real.append(indice[0].cpu().numpy())
        hab_pred.append(room_predicted.cpu().numpy())

        #Si la habitación real coincide con la habitación predicha, se considera un acierto de la red
        if room_predicted == indice[0]:
            cont_correct = cont_correct + 1
        # else:
        #     print('Habitación predicha: ' + str(room_predicted) + ', habitación real: ' + str(room_test[0]))
        #     print(euclidean_distance)
        # if confidence < 0.5:
        #     if room_predicted == room_test[0]:
        #         cont_correct = cont_correct + 1
        #     # else:
        #     #     print('Habitación predicha: ' + str(room_predicted) + ', habitación real: ' + str(room_test[0]) )
        #     #     print(euclidean_distance)
        # else:
        #     no_me_localizo = no_me_localizo + 1

        cont = cont + 1

        if time_idx > 0:
            sum_time = time_execution + sum_time


    cm = confusion_matrix(hab_real, hab_pred)
    with open("matriz_confusion_sunny_exp6c", "wb") as fp:
        pickle.dump(cm, fp)
    file = open('matriz_confusion_sunny_exp6c', 'rb')
    matriz_sunny = pickle.load(file)
    print("MATRIZ DE CONFUSIÓN TEST SOLEADO")
    print(matriz_sunny)
    display_confusion_matrix(cm=cm, illumination='sunny',plt_name='matriz_confusion_sunny_exp6c.png', net=PARAMETERS.test_net, gd='gd_silhouette_net0')
    accuracy_sunny=(cont_correct/cont)*100
    print('La precisión para soleado es: ', accuracy_sunny)
    min_time=sum_time//60
    sec_time=sum_time%60
    print("Tiempo de ejecución para test de soleado: ", min_time,"min ", sec_time, "s")
    writer.writerow([accuracy_cloudy,accuracy_night,accuracy_sunny])