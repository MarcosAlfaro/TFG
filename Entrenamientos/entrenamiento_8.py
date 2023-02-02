# Commented out IPython magic to ensure Python compatibility.

# importamos los módulos y librerías necesarios
# torch: realizar cálculos numéricos haciendo uso de la programación de tensores
# numpy: operaciones numéricas
# matplotlib: representar gráficas
# torchvision: datasets, transformación de imágenes en visión por computador
# PIL: procesamiento de imágenes
# os:sirve para trabajar con archivos,carpetas y rutas

import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import PIL.ImageOps
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pandas as pd
import os
import time
import csv

# importamos los parámetros del programa definidos en el archivo .yaml
from config import PARAMETERS


# indica que si el computador dispone de cuda, trabajaremos con cuda, si no, se hará en la cpu
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

# definimos esta función que nos permite mostrar la gráfica de la función de pérdida


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    # plt.scatter(iteration_hard,loss_hard,color='red')
    plt.show()

# definimos estas dos funciones que nos permiten mostrar por pantalla el tiempo dedicado por el programa al entrenamiento y a la validación de la red


def mostrar_tiempo_train(t):
    day_time = int(t // 86400)
    sec_time = int(t % 86400)
    hour_time = sec_time // 3600
    sec_time = sec_time % 3600
    min_time = sec_time // 60
    sec_time = sec_time % 60

    if day_time > 0:
        print("Tiempo de entrenamiento: ", day_time, "d ", hour_time, "h ", min_time, "min ", sec_time, "s")
    elif hour_time > 0:
        print("Tiempo de entrenamiento: ", hour_time, "h ", min_time, "min ", sec_time, "s")
    elif min_time > 0:
        print("Tiempo de entrenamiento: ", min_time, "min ", sec_time, "s")
    else:
        print("Tiempo de entrenamiento: ", sec_time, "s")
    return


def mostrar_tiempo_val(t):
    day_time = int(t // 86400)
    sec_time = int(t % 86400)
    hour_time = sec_time // 3600
    sec_time = sec_time % 3600
    min_time = sec_time // 60
    sec_time = sec_time % 60

    if day_time > 0:
        print("Tiempo de validación: ", day_time, "d ", hour_time, "h ", min_time, "min ", sec_time, "s")
    elif hour_time > 0:
        print("Tiempo de validación: ", hour_time, "h ", min_time, "min ", sec_time, "s")
    elif min_time > 0:
        print("Tiempo de validación: ", min_time, "min ", sec_time, "s")
    else:
        print("Tiempo de validación: ", sec_time, "s")
    return


# Obtenemos la ruta del directorio en el que estamos trabajando
base_dir = os.getcwd()


# en esta clase obtenemos las combinaciones de tripletas de imágenes (anchor, positive, negative)
# guardadas en el archivo 'ListaEntrenamiento.csv' generado en el programa generacionimagenes.py
# y las transformamos a formato RGB
class GeneracionDatasetEntrenamiento(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        fichero_train = pd.read_csv(PARAMETERS.base_dir + PARAMETERS.train_csv_dir)

        self.list_anc = fichero_train['Anchor']
        self.list_pos = fichero_train['Positive']
        self.list_neg = fichero_train['Negative']

        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):

        img_anc = self.list_anc[index]
        img_pos = self.list_pos[index]
        img_neg = self.list_neg[index]

        anc = Image.open(img_anc)
        anc = anc.convert("RGB")

        pos = Image.open(img_pos)
        pos = pos.convert("RGB")


        neg = Image.open(img_neg)
        neg = neg.convert("RGB")

        if self.should_invert:
            anc = PIL.ImageOps.invert(anc)
            pos = PIL.ImageOps.invert(pos)
            neg = PIL.ImageOps.invert(neg)

        if self.transform is not None:
            anc = self.transform(anc)
            pos = self.transform(pos)
            neg = self.transform(neg)

        return anc, pos, neg

    def __len__(self):
        return len(self.list_anc)


# en esta clase obtenemos las combinaciones de imgs de validación con las imágenes representativas de cada habitación
# guardadas en el archivo 'ListaValidacion.csv' generado en el programa generacionimagenes_val.py
# y las transformamos a formato RGB
class GeneracionDatasetValidacion(Dataset):

    def __init__(self, imageFolderDataset, transform=None, should_invert=True):

        fichero_val = pd.read_csv(PARAMETERS.base_dir + PARAMETERS.val_csv_dir)

        self.list_img0 = fichero_val['Img0']
        self.list_img1 = fichero_val['Img1']
        self.indice_hab = fichero_val['Indice']

        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):

        img0 = self.list_img0[index]
        img1 = self.list_img1[index]
        indice = self.indice_hab[index]

        img0_rgb = Image.open(img0)
        img0_rgb = img0_rgb.convert("RGB")

        img1_rgb = Image.open(img1)
        img1_rgb = img1_rgb.convert("RGB")

        if self.should_invert:
            img0_rgb = PIL.ImageOps.invert(img0_rgb)
            img1_rgb = PIL.ImageOps.invert(img1_rgb)

        if self.transform is not None:
            img0_rgb = self.transform(img0_rgb)
            img1_rgb = self.transform(img1_rgb)

        return img0_rgb, img1_rgb, indice

    def __len__(self):
        return len(self.list_img0)


# Llamamos a las dos clases anteriores para generar un dataset con las imágenes destinadas al entrenamiento de la red
# y otro para las imágenes destinadas al proceso de validación
training_dataset = GeneracionDatasetEntrenamiento(imageFolderDataset=PARAMETERS.base_dir + PARAMETERS.training_dir,
                                                  transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                                transforms.ToTensor()
                                                                                ]),
                                                  should_invert=False)

validation_dataset = GeneracionDatasetValidacion(imageFolderDataset=PARAMETERS.base_dir + PARAMETERS.validation_dir,
                                                 transform=transforms.Compose([transforms.Resize((128, 512)),
                                                                               transforms.ToTensor()
                                                                               ]),
                                                 should_invert=False)


def run():
    torch.multiprocessing.freeze_support()
    print('loop')


if __name__ == '__main__':
    run()


# Se carga la red 'vgg16' en la cpu
vgg16 = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)


# En esta clase se define la estructura de la red neuronal, cómo están conectadas las capas entre sí,
# cómo se transmite la información de los inputs a los outputs,etc.
class TripletNetwork(nn.Module):

    def __init__(self):
        super(TripletNetwork, self).__init__()
        self.cnn1 = vgg16.features
        if PARAMETERS.do_dataparallel:
            self.cnn1 = nn.DataParallel(self.cnn1)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 16))
        if PARAMETERS.do_dataparallel:
            self.avgpool = nn.DataParallel(self.avgpool)
        self.fc1 = nn.Sequential(
          nn.Linear(4*16*512, 500),
          nn.ReLU(inplace=True),

          nn.Linear(500, 500),
          nn.ReLU(inplace=True),

          nn.Linear(500, 5))
        if PARAMETERS.do_dataparallel:
            self.fc1 = nn.DataParallel(self.fc1)

    def forward(self, x):
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

    # NOTA: esta función de la clase es muy importante porque indica cómo hemos planteado el problema:
    # En lugar de crear tres redes neuronales en la definición de la red, se creará una única red simple
    # y se llamará a la red varias veces, según el proceso que se desee realizar
    # Para el entrenamiento, se llamará a la red tres veces, y para la validación se hará en dos ocasiones
    # def forward(self, input1):
    #     output1 = self.forward_once(input1)
    #     return output1


# Cargamos los lotes de imágenes de entrenamiento y validación a la cpu
train_dataloader = DataLoader(training_dataset,
                              shuffle=PARAMETERS.shuffle_train,       # si shuffle==True, carga los lotes de forma aleatoria
                              num_workers=PARAMETERS.workers_train,      # en el PC habitual da error si vale 16 -> hay que poner 0
                              batch_size=PARAMETERS.train_batch_size)

validation_dataloader = DataLoader(validation_dataset,
                                   shuffle=PARAMETERS.shuffle_val,       # si shuffle=False, carga los lotes en orden
                                   num_workers=PARAMETERS.workers_val,
                                   batch_size=PARAMETERS.val_batch_size)

print('Training batch number: {}'.format(len(train_dataloader)))

# def best_pos_distance(query, pos_vecs):
#     with torch.name_scope('best_pos_distance') as scope:
#         #batch = query.get_shape()[0]
#         num_pos = pos_vecs.get_shape()[1]
#         query_copies = torch.tile(query, [1,int(num_pos),1]) #shape num_pos x output_dim
#         best_pos=torch.reduce_min(torch.reduce_sum(torch.squared_difference(pos_vecs,query_copies),2),1)
#         #best_pos=tf.reduce_max(tf.reduce_sum(tf.squared_difference(pos_vecs,query_copies),2),1)
#         return best_pos
#
# def lazy_triplet_loss(q_vec, pos_vecs, neg_vecs, margin):
#     best_pos=best_pos_distance(q_vec, pos_vecs)
#     num_neg = neg_vecs.get_shape()[1]
#     batch = q_vec.get_shape()[0]
#     query_copies = torch.tile(q_vec, [1, int(num_neg),1])
#     best_pos=torch.tile(torch.reshape(best_pos,(-1,1)),[1, int(num_neg)])
#     m=torch.fill([int(batch), int(num_neg)],margin)
#     triplet_loss=torch.reduce_mean(torch.reduce_max(torch.maximum(torch.add(m,torch.subtract(best_pos,torch.reduce_sum(torch.squared_difference(neg_vecs,query_copies),2))), torch.zeros([int(batch), int(num_neg)])),1))
#     return triplet_loss


def best_pos_distance(query, pos_vecs):
    # num_pos = pos_vecs.shape[0]
    # query_copies = query.repeat(int(num_pos), 1)
    diff = ((pos_vecs - query) ** 2).sum(1)

    min_pos, _ = diff.min(0)
    max_pos, _ = diff.max(0)
    return min_pos, max_pos


def triplet_loss(q_vec, pos_vecs, neg_vecs, margin, use_min=False, lazy=True, ignore_zero_loss=False):

    min_pos, max_pos = best_pos_distance(q_vec, pos_vecs)

    if use_min:
        positive = min_pos
    else:
        positive = max_pos
    num_neg = neg_vecs.shape[0]
    num_pos = pos_vecs.shape[0]
    query_copies = q_vec
    positive = positive.view(-1, 1)
    positive = positive.repeat(int(num_neg), 1)

    negative = ((neg_vecs - query_copies) ** 2).sum(1).unsqueeze(1)

    loss = margin + positive - ((neg_vecs - query_copies) ** 2).sum(1).unsqueeze(1)

    loss = loss.clamp(min=0.0)

    if lazy:
        triplet_loss = loss.max(1)[0]
    else:
        triplet_loss = loss.sum(0)
    if ignore_zero_loss:
        hard_triplets = torch.gt(triplet_loss, 1e-16).float()
        num_hard_triplets = torch.sum(hard_triplets)
        triplet_loss = triplet_loss.sum() / (num_hard_triplets + 1e-16)
    else:
        triplet_loss = triplet_loss.mean()
    return triplet_loss


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = F.pairwise_distance(anchor, positive, keepdim=True)
        distance_negative = F.pairwise_distance(anchor, negative, keepdim=True)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


class LazyTripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(LazyTripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        distance = (x1 - x2).pow(2).sum(1)
        return distance

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:

        distance_positive = F.pairwise_distance(anchor, positive, keepdim=True)
        distance_negative = F.pairwise_distance(anchor, negative, keepdim=True)
        loss = torch.relu(distance_positive - distance_negative + self.margin)

        return loss.max()


# Llamamos a la red
net = TripletNetwork().to(device)

# Aplicamos la función de pérdida TripletMarginLoss
# criterion = nn.TripletMarginLoss(margin=1.0, p=2)


# Definimos algunos de los hiperparámetros de la red
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print(net)
params = list(net.parameters())
print('El numero de parametros de la red es: ', len(params))


# Realizamos el entrenamiento de la red
"""ENTRENAMIENTO DE LA RED"""

counter = []
loss_history = []
iteration_hard = []
loss_hard = []
max_accuracy = 0
iteration_number = 0
nombre = PARAMETERS.train_net

time_idx = 0
sum_time_train = 0
sum_time_val = 0

with open(PARAMETERS.base_dir + PARAMETERS.data_csv_dir, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(
        ["Epoca", "Num lotes cargados", "Num combinaciones cargadas", "Acierto funcion de perdida","Max accuracy", "Tiempo ejecucion entrenamiento","Tiempo ejecucion validacion" ])

    for epoch in range(0, PARAMETERS.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):

            time_idx += 1
            start_train = time.time()
            # Asignamos las imágenes cargadas en cada lote a variables con las que trabajaremos
            anc, pos, neg = data
            # anc, pos, neg, label_pos, label_neg = anc.to(device), pos.to(device), neg.to(device), label_pos, label_neg

            # Reseteamos el optimizador entre iteración e iteración
            optimizer.zero_grad()

            # Llamamos a la red tres veces,una para cada imagen(anchor, positive,negative)
            output1 = net(anc)
            output2 = net(pos)
            output3 = net(neg)

            accuracy_loss = 100
            # Llamamos a la función de pérdida para que actualice los pesos internos de la red según el error que haya cometido
            # loss = triplet_loss(output1, output2, output3, margin=1.0)
            # if i%100==0:
            #     criterion = LazyTripletLoss()
            #     loss, indice_hard = criterion(output1, output2, output3)
            #     iteration_hard.append(iteration_number)
            #     loss_hard.append(loss)
            #     accuracy_loss = 0
            #     if indice_hard != 0:
            #         accuracy_loss += 1
            #
            # else:
            criterion = TripletLoss()
            loss = criterion(output1, output2, output3)

            loss.backward()   # triplet_loss = criterion(output1,output2,output3)
            # triplet_loss.backward()

            # Actualizamos los pesos internos de la red
            optimizer.step()

            mean_accuracy = 0
            counter.append(iteration_number)
            loss_history.append(loss.item())

            # Cada cierto número de lotes de imágenes definido por el usuario, guardamos el valor actual de la función de pérdida y lo mostramos por pantalla
            if i % PARAMETERS.show_loss == 0:
                print("Iteracion ", i)
                print("Epoch number {}\n Current loss {}".format(epoch,loss.item()))
                iteration_number += PARAMETERS.show_loss

            end_train = time.time()
            time_execution_train = (end_train - start_train)

            if time_idx > 0:
                sum_time_train = time_execution_train + sum_time_train
                # print("Tiempo de ejecución", sum_time_train)

            # Cada 200 lotes de imágenes, realizamos una validación para saber el porcentaje de acierto actual de la red
            if i % PARAMETERS.do_validation == 0:
                correctas = 0
                start_val = time.time()
                for j, validation_data in enumerate(validation_dataloader, 0):

                    img_val, img_comp, indice_hab = validation_data
                    output1 = net(img_val)
                    output2 = net(img_comp)
                    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
                    room_predicted = torch.argmin(euclidean_distance)

                    # print("Room predicted: ", room_predicted)
                    # print("Room groundtruth: ", indice_hab[0])
                    if room_predicted == indice_hab[0]:
                        # print("La habitacion es correcta")
                        correctas += 1
                    # else:
                    #     print("La habitacion es erronea")
                end_val = time.time()
                time_execution_val = (end_val - start_val)
                if time_idx > 0:
                    sum_time_val = time_execution_val + sum_time_val

                accuracy = (correctas/(j+1))*100

                print("Precision= ", accuracy, " %")

                if accuracy >= max_accuracy and accuracy > PARAMETERS.min_val_accuracy:
                    max_accuracy = accuracy
                    mean_accuracy += accuracy
                    # Si la precisión de la validación es mayor que la máxima hasta el momento,guardamos la red y mostramos por pantalla la gráfica de la función de pérdida

                    net_epochs = nombre + "_epoch" + str(epoch) + "_iteration" + str(i/200)+"_accuracy"+str(accuracy)
                    torch.save(net, net_epochs)
                    print("RED GUARDADA")
                    print("Época: ", epoch)
                    print("Nº de lotes cargados: ", i+1)
                    print("Nº de combinaciones de imágenes cargadas: ", ((i+1)*PARAMETERS.train_batch_size)+epoch*len(anc))
                    print("Acierto de validación máximo: ", max_accuracy, "%")
                    # print("Acierto función de pérdida: ",accuracy_loss*100/PARAMETERS.num_iteraciones,"%")
                    mostrar_tiempo_train(sum_time_train)
                    mostrar_tiempo_val(sum_time_val)
                    writer.writerow([epoch, i+1, ((i+1)*PARAMETERS.train_batch_size)+epoch*len(anc), accuracy_loss*100/PARAMETERS.num_iteraciones, max_accuracy, sum_time_train, sum_time_val])

                elif accuracy >= max_accuracy:
                    max_accuracy = accuracy
                    # show_plot(counter, loss_history,iteration_hard,loss_hard)
                    # show_plot(counter, loss_history)

                if accuracy >= PARAMETERS.accuracy_end_train:
                    break
                    break
                    break
                    # show_plot(counter, loss_history,iteration_hard,loss_hard)
                    show_plot(counter, loss_history)

        print("Epoca ", epoch, " finalizada")
    print("Max accuracy: ", max_accuracy, "%")