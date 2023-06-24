#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
import sys
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
import pygame
import mido
import pandas as pd
import io
import threading

import torch
import scipy.io as sio


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
note_decay = 5.0 / sample_rate
num_measures = 16
num_sigmas = 5.0
note_threshold = 32
use_pca = True
is_ae = True

background_color = (255, 255, 255)
edge_color = (60, 60, 60)
slider_colors = [(90, 20, 20), (90, 90, 20), (20, 90, 20), (20, 90, 90), (20, 20, 90), (90, 20, 90)]

note_w = 4
note_h = 128

control_num = 2
control_inits = [0.75, 0.5, 0.5]
control_colors = [(255, 128, 0), (0, 0, 255)]

window_width = 800
window_height = 600
margin = 20
sliders_width = int(window_width * (2.0/4.0))
sliders_height = int(window_height * (2.5/3.0))
slider_width = int((sliders_width-margin*2) / 4.0)
slider_height = sliders_height-margin*2

controls_width = int(window_width * (2.0/4.0))
controls_height = int(window_height * (1.0/3.0))
control_width = controls_width - margin*2
control_height = int((controls_height-margin*2) / 2.0)
cur_control_iy = 0
detected_keys = []
prev_measure_ix = 0

filename = '../assets/knob.png'
knob = pygame.image.load(filename)
filename = '../assets/control_knob.png'
c_knob = pygame.image.load(filename)
filename = '../assets/button.png'
button_png = pygame.image.load(filename)

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

prev_mouse_pos = None
mouse_pressed = 0
cur_slider_ix = 0
cur_control_ix = 0
cur_control_iy = 0

volume = 3000
instrument = 0
needs_update = False
current_params = np.zeros((1,num_params), dtype=np.float32)
current_params_statics = np.copy(current_params)
current_notes = np.zeros((note_h, note_w), dtype=np.uint8)
current_midi_events = matrix2mid(current_notes.astype(int), tempo=750000)
current_midi_events = [msg for msg in current_midi_events]
current_midi_size = len(current_midi_events) 
current_file_index = 0

cur_controls = np.array(control_inits, dtype=np.float32)
songs_loaded = False

FUND_PITCH = 48
sound_bank = []
dev_cnt = 0
flag = 0
flag_midi_reset = 0
sr = 44100

audio = pyaudio.PyAudio()
audio_notes = []
audio_time = 0
note_time = 0
note_time_dt = 0
audio_reset = False
audio_pause = False

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

def update_mouse_click(mouse_pos):
    global cur_slider_ix
    global cur_control_ix
    global mouse_pressed

    global cur_control_iy
    global audio_pause
    global slider_num

    if margin <= mouse_pos[0] < margin+slider_width*slider_num and margin <= mouse_pos[1] < margin+slider_height:
        cur_slider_ix = int((mouse_pos[0]-margin*2) / slider_width)
        mouse_pressed = 1

    if margin*2 <= mouse_pos[0] < margin*2+control_width and ((sliders_height + margin*2 <= mouse_pos[1] < sliders_height + margin*2 + (control_height/2)) or (sliders_height + margin*2 + control_height <= mouse_pos[1] < sliders_height + margin*2 + control_height + (control_height/2))):
        cur_control_iy = int((mouse_pos[1] - (sliders_height + margin*2)) / (control_height)) 
        mouse_pressed = 2


def update_mouse_move(mouse_pos):
    global needs_update
    global audio_reset

    if mouse_pressed == 1:
        if margin <= mouse_pos[1] <= margin+slider_height:
            val = (float(mouse_pos[1]-margin) / slider_height - 0.5) * (num_sigmas * 2)
            # mapped_range = interpolators[cur_slider_ix][0](val)
            mapped_range = -1 * val * all_principal_components_statistics[cur_slider_ix]['x_std'] + all_principal_components_statistics[cur_slider_ix]['x_mean'] 
            current_params[0][int(cur_slider_ix)] = mapped_range
            needs_update = True
            audio_reset = True
    
def draw_sliders(screen):
    global knob
    knob = pygame.transform.scale(knob, (30, 50))

    for i in range(slider_num):
        slider_color = (100, 100, 100)
        slider_color_layer = (195, 195, 195)
        x = margin + i * slider_width + 30
        y = margin*2

        cx = x + slider_width / 2
        cy_start = y
        cy_end = y + slider_height
        pygame.draw.line(screen, slider_color_layer, (cx, cy_start), (cx, cy_end), 16)
        pygame.draw.circle(screen, slider_color_layer, (cx+1, cy_start), 8)
        pygame.draw.circle(screen, slider_color_layer, (cx+1, cy_end), 8)
        pygame.draw.line(screen, slider_color, (cx, cy_start), (cx, cy_end), 4)

        cx_1 = x + int(slider_width* (3.0/4.0))
        cx_2 = x + slider_width-int(slider_width * (0.75/4.0))
        for j in range(int(num_sigmas * 2 + 1)):
            ly = y + slider_height / 2.0 + (j - num_sigmas) * slider_height / (num_sigmas * 2.0)
            ly = int(ly)
            col = (0, 0, 0) if j - num_sigmas == 0 else slider_color
            pygame.draw.line(screen, col, (cx_1, ly), (cx_2, ly), 1)


        # val = interpolators[i][1](current_params[0][i])
        val = -1 * (current_params[0][i] - all_principal_components_statistics[i]['x_mean']) / all_principal_components_statistics[i]['x_std']

        py = y + int((val / (num_sigmas * 2) + 0.5) * slider_height) - 25
        screen.blit(knob, (int(cx-15), int(py)))


def draw_text(screen):

    global detected_keys

    pygame.font.init()
    font = pygame.font.SysFont(None, 50)
    description_font = pygame.font.SysFont(None, 25)
    label_font = pygame.font.SysFont(None, 15)

    text_sliders = label_font.render(f'LATENT VALUES (TOP {slider_num})', True, (0, 0, 0))
    screen.blit(text_sliders, (margin*2.5, margin-5))



    x = margin - 5
    y = margin*2
    cx_2 = x + slider_width - 30

    y1 = y + slider_height / 2.0 + (10 - num_sigmas) * slider_height / (num_sigmas * 2.0)
    y1 = int(y1)
    for index, value in enumerate(["x_curr", "z_curr", "x_min", "x_max", "x_std", "z_min", "z_max"]):
            temp_text = label_font.render(value, True, (0, 0, 0))
            text_height = (temp_text.get_rect().height) / 2.0
            screen.blit(temp_text, (cx_2-52, y1 + (20 + index * 10)))

    for i in range(slider_num):
        x = margin + i * slider_width + 30
        y = margin*2
        cx_2 = x + slider_width - 8

        y1 = y + slider_height / 2.0 + (10 - num_sigmas) * slider_height / (num_sigmas * 2.0)
        y1 = int(y1)
        text_slider_value_5 = label_font.render(f'{all_principal_components_statistics[i]["x_std"] * -5:.3f}', True, (0, 0, 0))
        text_height = (text_slider_value_5.get_rect().height) / 2.0
        screen.blit(text_slider_value_5, (cx_2, y1-text_height))

        ## Current parameter
        current_param = label_font.render(f'{current_params[0][i]:.7f}', True, (0, 0, 0))
        text_height = (current_param.get_rect().height) / 2.0
        screen.blit(current_param, (cx_2-52, y1 + 20))

        ## Current parameter in z-score
        val =  (current_params[0][i] - all_principal_components_statistics[i]['x_mean']) / all_principal_components_statistics[i]['x_std']
        current_param = label_font.render(f'{val:.7f}', True, (0, 0, 0))
        text_height = (current_param.get_rect().height) / 2.0
        screen.blit(current_param, (cx_2-52, y1 + 30))

        ## Extra parameters
        for index, value in enumerate(["x_min", "x_max", "x_std", "z_min", "z_max"]):
            temp_text = label_font.render(f'{all_principal_components_statistics[i][value]:.7f}', True, (0, 0, 0))
            text_height = (temp_text.get_rect().height) / 2.0
            screen.blit(temp_text, (cx_2-52, y1 + (40 + index * 10)))

        y1 = y + slider_height / 2.0 + (5 - num_sigmas) * slider_height / (num_sigmas * 2.0)
        y1 = int(y1)
        text_slider_value_0 = label_font.render('0', True, (0, 0, 0))
        text_height = (text_slider_value_0.get_rect().height) / 2.0
        screen.blit(text_slider_value_0, (cx_2, y1-text_height))

        y2 = y + slider_height / 2.0 + (0 - num_sigmas) * slider_height / (num_sigmas * 2.0)
        y2 = int(y2)
        text_slider_value_m5 = label_font.render(f'{all_principal_components_statistics[i]["x_std"] * 5:.3f}', True, (0, 0, 0))
        text_height = (text_slider_value_m5.get_rect().height) / 2.0
        screen.blit(text_slider_value_m5, (cx_2, y2-text_height))


def detect_keys(keys):
    result_keys = []
    semitone_dic = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
                    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}

    for key in keys:
        semitone = key%12
        octave = str(key//12 - 2)
        if semitone in semitone_dic:
            semitone = semitone_dic[semitone]
        result_keys.append(semitone+octave)
    return result_keys

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


    pygame.init()
    pygame.mixer.init()
    pygame.font.init()
    screen = pygame.display.set_mode((int(window_width), int(window_height)))
    pygame.display.set_caption('')

    play_thread=threading.Thread(target=play_midi)
    play_thread.start()
    running = True
    random_song_ix = 0
    max_random_songs = float('inf')
    # apply_controls()

    while running:
        # process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                _play=False
                break

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if pygame.mouse.get_pressed()[0]:
                    prev_mouse_pos = pygame.mouse.get_pos()
                    update_mouse_click(prev_mouse_pos)
                    update_mouse_move(prev_mouse_pos)
                elif pygame.mouse.get_pressed()[2]:
                    current_params = np.zeros((1,num_params), dtype=np.float32)
                    needs_update = True

            elif event.type == pygame.MOUSEBUTTONUP:
                mouse_pressed = 0
                prev_mouse_pos = None

            elif event.type == pygame.MOUSEMOTION and mouse_pressed > 0:
                update_mouse_move(pygame.mouse.get_pos())

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    if not songs_loaded:
                        print("Loading songs...")
                        try:
                            songs_loaded = True
                            max_random_songs = principal_components.shape[0]
                        except Exception as e:
                            print("This functionality is to check if the model training went well by reproducing an original song. "
                                  "The composer could not load samples and lengths from model training. "
                                  "If you have the midi files, the model was trained with, process them by using"
                                  " the preprocess_songs.py to find the requested files in data/interim "
                                  "(Load exception: {0}".format(e))


                    if songs_loaded:
                        print("Random Song Index: " + str(random_song_ix))
                        current_params = [principal_components[int(random_song_ix)]]
                        random_song_ix = (random_song_ix + 1) % max_random_songs

                        needs_update = True
                        audio_reset = True
                if event.key == pygame.K_r:
                    needs_update = True
                    current_params = np.copy(current_params_statics)



        if next_song: 
            print("Random Song Index: " + str(random_song_ix))
            current_params = principal_components[int(random_song_ix)]

            random_song_ix = (random_song_ix + 1) % max_random_songs
            needs_update = True
            audio_reset = True
            next_song = False

        if needs_update:
            current_params_statics = np.copy(current_params)

            latent_current = torch.from_numpy(np.array(current_params))

            if 'SAE' in type(model).__name__:
                X_recon = model.decoder_svd(latent_current)
            else:
                X_recon = model.decoder(latent_current)

            X_recon = X_recon.detach().numpy() 
            current_notes = np.argmax(X_recon.reshape((-1, X_recon.shape[-1])), axis=-1).reshape((X_recon.shape[1],X_recon.shape[2]))

            try:
                temp_midi_events = matrix2mid(current_notes.astype(int), tempo=750000)
            except:
                running = False
                _play=False

            current_midi_events = [msg for msg in temp_midi_events]

            current_midi_size = len(current_midi_events)
            # print(current_params)
            if audio_reset:
                current_file_index = 0
                audio_reset = False
            # current_file_index = 0
            needs_update = False

        screen.fill(background_color)
        draw_sliders(screen)
        # draw_controls(screen)
        # text_background(screen)
        draw_text(screen)

        pygame.display.flip()
        pygame.time.wait(10)

    # audio_stream.stop_stream()
    # audio_stream.close()
    audio.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CHANGE NAME')
    # parser.add_argument('--model_path', type=str, required=True)
    # parser.add_argument('--latent_path', type=str, required=True)
    args = parser.parse_args()
    # sub_dir_name = args.model_path
    # latent_path = args.latent_path

    play()
