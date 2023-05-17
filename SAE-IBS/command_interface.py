#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import sys

from SAE-IBS.DB_PASS import DB_PASS
sys.path.append('../')

from mid2matrix.matrix2mid import matrix2mid
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import math
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

import pyaudio
import time
import mido
import pandas as pd
import io
import threading

import torch
import scipy.io as sio
import psycopg2


dir_name = ''
model_name = 'SAE'
latent_space_model_number = 7
sub_dir_name = f'results/{model_name}/{model_name}_3_Layer_{latent_space_model_number}_Latent_9216_2560_512/{model_name}_3_Layer_{latent_space_model_number}_Latent_9216_2560_512.tar'
latent_path = f'results/{model_name}/{model_name}_3_Layer_{latent_space_model_number}_Latent_9216_2560_512/latent/latent_project.mat'
complete_latent_path = f'results/{model_name}/{model_name}_3_Layer_{latent_space_model_number}_Latent_9216_2560_512/latent/all_latent_project.mat'

sample_rate = 44100
buffer_size = 64
buffer_sec = (buffer_size / sample_rate)
note_dt = 2000
note_duration = 20000

use_pca = True
is_ae = True


note_w = 4
note_h = 128




project_mat = sio.loadmat(latent_path)
all_project_mat = sio.loadmat(complete_latent_path)

principal_components = project_mat['latent_project']
all_principal_components = all_project_mat['latent_project']
all_principal_components = np.transpose(all_principal_components, (1,0))

X_samples = project_mat['test_data']

num_params = principal_components.shape[1] # DIMENSIONS
slider_num = min(10, num_params)

pcs_distributions = []
interpolators = []

print(principal_components.shape)

for x in range(principal_components.shape[1]):
    pcs_distributions.append((np.min(principal_components[:, x]), np.max(principal_components[:,x]), np.mean(principal_components[:,x]), np.std(principal_components[:,x])))
    f = interp1d([-5, 0, 5], [np.min(principal_components[:,x]), np.mean(principal_components[:,x]), np.max(principal_components[:,x])])  #when mapping from slider to component
    f_inv = interp1d([np.min(principal_components[:,x]), np.mean(principal_components[:,x]), np.max(principal_components[:,x])], [-5, 0, 5], assume_sorted=True, kind='linear', bounds_error=False) #when mapping from component to slider
    interpolators.append((f,f_inv))

all_principal_components_statistics = []
z_scores_components = []
for x in all_principal_components:
    z_scores = (x-np.mean(x))/np.std(x)
    z_scores_components.append(z_scores)
    all_principal_components_statistics.append({'x_min' : np.min(x), 'x_max': np.max(x), 'x_mean': np.mean(x), 'x_std': np.std(x), 'z_min' : np.min(z_scores), 'z_max' : np.max(z_scores), 'z_mean' : np.mean(z_scores), 'z_std': np.std(z_scores)})
   
print(all_principal_components_statistics)


note_time = 0
note_time_dt = 0
audio_reset = False
audio_pause = False


needs_update = False
current_params = np.zeros((1,num_params), dtype=np.float32)
current_params_statics = np.copy(current_params)
current_notes = np.zeros((note_h, note_w), dtype=np.uint8)
current_midi_events = matrix2mid(current_notes.astype(int))
current_midi_events = [msg for msg in current_midi_events]
current_midi_size = len(current_midi_events) 
current_file_index = 0

next_song = False

port = mido.open_output('IAC Driver Bus 1')
play_thread=None
_play=True

def play_midi():    
    while _play:
        start_time = time.time()
        input_time = 0.0 
        idx=0
        # print(" hola")

        while idx<len(current_midi_events) and _play:
            msg= current_midi_events[idx]
            idx+=1
            input_time +=msg.time
            playback_time = time.time() - start_time
            duration_to_next_event = input_time - playback_time            
            if duration_to_next_event > 0.0:
                time.sleep(duration_to_next_event)
            if not (msg.type == 'note_on' or msg.type == 'note_off'):
                continue
            else:
                port.send(msg)
            

def map_range(value, from_range, to_range):
    # Obtener los valores mínimos y máximos de los rangos de entrada y salida
    from_min, from_max = from_range
    to_min, to_max = to_range

    # Convertir el valor en una fracción del rango de entrada y luego escalar esa fracción al rango de salida
    return ((value - from_min) * (to_max - to_min) / (from_max - from_min)) + to_min

def map_range_inverse(value, from_range, to_range):
    from_min, from_max = from_range
    to_min, to_max = to_range
    return ((value - to_min) * (from_max - from_min) / (to_max - to_min)) + from_min

def z_transform(value, mean, std):
    # Perform z-score normalization manually
    return (value - mean) / std

def inverse_z_transform(value,mean,std):
    # Perform inverse z-score normalization
    return (value * std) + mean

def test_function():
    while True:
        print("Función ejecutándose...")
        time.sleep(5)


## database conection

dbname = 'aeros'
user = 'm5242108'
password = DB_PASS
host = '163.143.165.136'
port = '5432'

# Establecer la conexión
conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)

# Crear un cursor para ejecutar consultas
cur = conn.cursor()

# Ejecutar una consulta
cur.execute('''SELECT * FROM data WHERE time >= '2023-01-01' AND time <= '2023-01-31' limit 20;''')

# Obtener los resultados
results = cur.fetchall()

# Imprimir los resultados
for row in results:
    print(row)

# Cerrar el cursor y la conexión
cur.close()
conn.close()


def play():
    global mouse_pressed
    global current_notes
    global audio_pause
    global needs_update
    global current_params
    global prev_mouse_pos
    global audio_reset
    global instrument
    global songs_loaded
    global current_midi_events
    global current_midi_size
    global current_file_index
    global next_song
    global current_params_statics
    global play_thread
    global _play
    # global steps


    print("Loading model...")
    checkpoint = torch.load(sub_dir_name, map_location=torch.device('cpu'))  
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])

    random_sample = current_notes.reshape(-1)

    play_thread=threading.Thread(target=play_midi)
    play_thread.start()

    test_t = threading.Thread(target=test_function)
    test_t.start()

    running = True
    random_song_ix = 0
    max_random_songs = float('inf')
    # apply_controls()

    while running:
        # process events
        # for event in pygame.event.get()


        #     elif event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_o:
        #             if not songs_loaded:
        #                 print("Loading songs...")
        #                 try:
        #                     songs_loaded = True
        #                     max_random_songs = principal_components.shape[0]
        #                 except Exception as e:
        #                     print("This functionality is to check if the model training went well by reproducing an original song. "
        #                           "The composer could not load samples and lengths from model training. "
        #                           "If you have the midi files, the model was trained with, process them by using"
        #                           " the preprocess_songs.py to find the requested files in data/interim "
        #                           "(Load exception: {0}".format(e))


        #             if songs_loaded:
        #                 print("Random Song Index: " + str(random_song_ix))
        #                 current_params = [principal_components[int(random_song_ix)]]
        #                 random_song_ix = (random_song_ix + 1) % max_random_songs

        #                 needs_update = True
        #                 audio_reset = True
        #         if event.key == pygame.K_r:
        #             needs_update = True
        #             current_params = np.copy(current_params_statics)



        # if next_song: 
        #     print("Random Song Index: " + str(random_song_ix))
        #     current_params = principal_components[int(random_song_ix)]

        #     random_song_ix = (random_song_ix + 1) % max_random_songs
        #     needs_update = True
        #     audio_reset = True
        #     next_song = False

        # if needs_update:
        #     current_params_statics = np.copy(current_params)

        #     latent_current = torch.from_numpy(np.array(current_params))

        #     if 'SAE' in type(model).__name__:
        #         X_recon = model.decoder_svd(latent_current)
        #     else:
        #         X_recon = model.decoder(latent_current)

        #     X_recon = X_recon.detach().numpy() 
        #     current_notes = np.argmax(X_recon.reshape((-1, X_recon.shape[-1])), axis=-1).reshape((X_recon.shape[1],X_recon.shape[2]))

        #     try:
        #         temp_midi_events = matrix2mid(current_notes.astype(int))
        #     except:
        #         running = False
        #         _play=False

        #     current_midi_events = [msg for msg in temp_midi_events]

        #     current_midi_size = len(current_midi_events)
        #     # print(current_params)
        #     if audio_reset:
        #         current_file_index = 0
        #         audio_reset = False
        #     # current_file_index = 0
        #     needs_update = False
        print('In construction')
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CHANGE NAME')
    # parser.add_argument('--model_path', type=str, required=True)
    # parser.add_argument('--latent_path', type=str, required=True)
    args = parser.parse_args()
    # sub_dir_name = args.model_path
    # latent_path = args.latent_path

    play()
