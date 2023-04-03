#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings

from mid2matrix.matrix2mid import matrix2mid
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import math
import numpy as np
from sklearn.decomposition import PCA
import pyaudio
import time
import pygame
import mido
import pandas as pd
import io


# df = pd.read_csv('csv/meteorological.csv')
# list = df.to_numpy()
# normalized_list = np.array([])
# steps = 0

# def min_max(x, axis=None):
#     min = x.min(axis=axis, keepdims=True)
#     max = x.max(axis=axis, keepdims=True)
#     result = (x-min)/(max-min)
#     return result

# for i in range(list.shape[1]-2):

#     result = min_max(list[:,i+2])
#     result = result - 0.5

#     if i>0:
#         normalized_list = np.vstack((normalized_list, result))
#     else:
#         normalized_list = np.append(normalized_list, result)

# normalized_list = normalized_list.T

dir_name = ''
sub_dir_name = ''
sample_rate = 44100
buffer_size = 64
buffer_sec = (buffer_size / sample_rate)
note_dt = 2000
note_duration = 20000
note_decay = 5.0 / sample_rate
num_params = 200 # DIMENSIONS
num_measures = 16
num_sigmas = 5.0
note_threshold = 32
use_pca = True
is_ae = True

background_color = (210, 210, 210)
edge_color = (60, 60, 60)
slider_colors = [(90, 20, 20), (90, 90, 20), (20, 90, 20), (20, 90, 90), (20, 20, 90), (90, 20, 90)]

note_w = 4
note_h = 128

slider_num = min(10, num_params)

control_num = 2
control_inits = [0.75, 0.5, 0.5]
control_colors = [(255, 128, 0), (0, 0, 255)]

window_width = 800
window_height = 600
margin = 20
sliders_width = int(window_width * (2.0/4.0))
sliders_height = int(window_height * (2.0/3.0))
slider_width = int((sliders_width-margin*2) / 5.0)
slider_height = sliders_height-margin*2

controls_width = int(window_width * (2.0/4.0))
controls_height = int(window_height * (1.0/3.0))
control_width = controls_width - margin*2
control_height = int((controls_height-margin*2) / 2.0)
cur_control_iy = 0
detected_keys = []
prev_measure_ix = 0

filename = 'assets/knob.png'
knob = pygame.image.load(filename)
filename = 'assets/control_knob.png'
c_knob = pygame.image.load(filename)
filename = 'assets/button.png'
button_png = pygame.image.load(filename)

X_samples = np.load('PCA/samples/samples.npy')
principal_components = np.load('PCA/samples/principal_components.npy')
pcs_distributions = []
for x in range(principal_components.shape[0]):
    pcs_distributions.append((np.min(principal_components[x]), np.max(principal_components[x]), np.mean(principal_components[x])))

print(pcs_distributions)

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
current_midi_events = matrix2mid(current_notes.astype(int))
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
sonification_mode = False

audio = pyaudio.PyAudio()
audio_notes = []
audio_time = 0
note_time = 0
note_time_dt = 0
audio_reset = False
audio_pause = False

next_song = False

# midi_output = mido.open_output('IAC Driver Bus 1')

port = mido.open_output('IAC Driver Bus 1')

def audio_callback(in_data, frame_count, time_info, status):
    # global current_midi_events
    global audio_reset
    global needs_update
    global next_song
    global current_file_index

    data = np.zeros((frame_count,), dtype=np.float32)

    if audio_pause and status is not None:
        data = np.zeros((frame_count,), dtype=np.float32)
        return data.tobytes(), pyaudio.paContinue
    
    if current_file_index >= current_midi_size:
        current_file_index = 0
        # next_song = True
    # print(current_file_index)   
    current_streaming_notes = []
    dt = 0
    for msg in current_midi_events[current_file_index:]:
        if msg.type == 'note_on' or msg.type == 'note_off':
            current_streaming_notes.append(msg)
            dt += (msg.time)
            if dt > buffer_sec:
                current_file_index += 1
                break
        current_file_index += 1
    
    start_time = time.time()
    input_time = 0.0
    for msg in current_streaming_notes:
        input_time += msg.time
        playback_time = time.time() - start_time
        duration_to_next_event = input_time - playback_time
        port.send(msg)
        if duration_to_next_event > 0.0:
            time.sleep(duration_to_next_event)

    return data.tobytes(), pyaudio.paContinue

def map_range(value, from_range, to_range):
    # Obtener los valores mínimos y máximos de los rangos de entrada y salida
    from_min, from_max = from_range
    to_min, to_max = to_range

    # Convertir el valor en una fracción del rango de entrada y luego escalar esa fracción al rango de salida
    return ((value - from_min) * (to_max - to_min) / (from_max - from_min)) + to_min

def map_range_inverse(value, from_range, to_range):
    # Obtener los valores mínimos y máximos de los rangos de entrada y salida
    from_min, from_max = from_range
    to_min, to_max = to_range

    # Convertir el valor mapeado en una fracción del rango de salida y luego escalar esa fracción al rango de entrada
    return ((value - to_min) * (from_max - from_min) / (to_max - to_min)) + from_min

def update_mouse_click(mouse_pos):
    global cur_slider_ix
    global cur_control_ix
    global mouse_pressed

    global cur_control_iy
    global audio_pause
    global sonification_mode
    global slider_num

    if margin <= mouse_pos[0] < margin+slider_width*slider_num and margin <= mouse_pos[1] < margin+slider_height:
        cur_slider_ix = int((mouse_pos[0]-margin*2) / slider_width)
        mouse_pressed = 1

    if margin*2 <= mouse_pos[0] < margin*2+control_width and ((sliders_height + margin*2 <= mouse_pos[1] < sliders_height + margin*2 + (control_height/2)) or (sliders_height + margin*2 + control_height <= mouse_pos[1] < sliders_height + margin*2 + control_height + (control_height/2))):
        cur_control_iy = int((mouse_pos[1] - (sliders_height + margin*2)) / (control_height))
        mouse_pressed = 2

    # x = window_width*(2.0/4.0)+margin
    # y = margin*2
    # if x <= mouse_pos[0] < x+window_width*(2.0/4.0)-margin*3 and y <= mouse_pos[1] < y + window_height*(1/3.0)-margin*2:
    #     audio_pause = not audio_pause

    # x = window_width*(2.0/4.0)+margin
    # y = window_height*(1 / 3.0)+margin*2
    # if x <= mouse_pos[0] < x+window_width*(2.0/4.0)-margin*3 and y <= mouse_pos[1] < y + window_height*(1/3.0)-margin*2:
    #     sonification_mode = not sonification_mode

def apply_controls():
    global note_threshold
    global note_dt
    global volume

    note_threshold = (1.0 - cur_controls[0]) * 200 + 10
    note_dt = (1.0 - cur_controls[1]) * 1800 + 200

def update_mouse_move(mouse_pos):
    global needs_update
    global audio_reset

    if mouse_pressed == 1:
        if margin <= mouse_pos[1] <= margin+slider_height:
            val = (float(mouse_pos[1]-margin) / slider_height - 0.5) * (num_sigmas * 2)
            mapped_range = map_range_inverse(val, (pcs_distributions[cur_slider_ix][0], pcs_distributions[cur_slider_ix][1]), (-5,5))
            current_params[0][int(cur_slider_ix)] = mapped_range
            needs_update = True
            audio_reset = True
    # elif mouse_pressed == 2:
    #     if margin <= mouse_pos[0] <= margin+control_width:
    #         val = float(mouse_pos[0] - margin) / control_width
    #         cur_controls[int(cur_control_iy)] = val
    #         apply_controls()

# def update_with_sonification():
#     global needs_update
#     global steps

#     for i in range(5):
#         current_params[i] = float(normalized_list[steps][i]) * 10.0 * (2.5 / 5.0)
#         needs_update = True

#     steps = steps + 1
#     if steps == list.shape[0]:
#         steps = 0

def draw_controls(screen):
    global c_knob
    c_knob = pygame.transform.scale(c_knob, (30, 40))

    slider_color = (100, 100, 100)
    slider_color_layer = (195, 195, 195)

    for i in range(control_num):
        x = margin + slider_width / 2 + 5
        y = sliders_height + margin*2 + i * control_height
        w = control_width - margin*3 - 5
        h = int(control_height / 2.0)
        col = control_colors[i]

        pygame.draw.line(screen, slider_color_layer,
                         (x-15, y+(h/2.0)), (x+w+15, y+(h/2.0)), 40)
        pygame.draw.line(screen, (75, 75, 75),
                         (x, y+(h/2.0)), (x+w, y+(h/2.0)), 4)

        pygame.draw.line(screen, col,
                         (x, y+(h/2.0)), (x+int(w * cur_controls[i])-15, y+(h/2.0)), 4)
        screen.blit(c_knob, (x+int(w * cur_controls[i])-15, y))
        

def draw_sliders(screen):
    global knob
    knob = pygame.transform.scale(knob, (30, 50))

    for i in range(slider_num):
        slider_color = (100, 100, 100)
        slider_color_layer = (195, 195, 195)
        x = margin + i * slider_width
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

        py = y + int((map_range(current_params[0][i], (pcs_distributions[i][0], pcs_distributions[i][1]), (-5,5)) / (num_sigmas * 2) + 0.5) * slider_height) - 25
        screen.blit(knob, (int(cx-15), int(py)))

# def draw_button(screen):

#     global button_png
#     button_play = pygame.transform.scale(
#         button_png, (int(window_width*(2.0/4.0)-margin*2), int(window_height*(1 / 3.0)-margin*2)))
#     button_change_mode = pygame.transform.scale(
#         button_png, (int(window_width*(2.0/4.0)-margin*2), int(window_height*(1 / 3.0)-margin*2)))

#     screen.blit(button_play, (window_width*(2.0/4.0)+margin, margin*2))
#     screen.blit(button_change_mode, (window_width*(2.0/4.0) +
#                 margin, window_height*(1 / 3.0)+margin*2))

def text_background(screen):
    text_background_color = (195, 195, 195)
    x = window_width*(2.0/4.0)+margin
    y = sliders_height + margin*2
    w = int(window_width*(2.0/4.0)-margin*2)
    h = int(control_height*2-margin*2)
    background_rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(screen, text_background_color, background_rect)

def draw_text(screen):

    global detected_keys

    pygame.font.init()
    font = pygame.font.SysFont(None, 50)
    description_font = pygame.font.SysFont(None, 25)
    label_font = pygame.font.SysFont(None, 15)

    text_sliders = label_font.render('LATENT VALUES (TOP 5)', True, (0, 0, 0))
    screen.blit(text_sliders, (margin*2.5, margin-5))

    text_threshold = label_font.render('THRESHOLD', True, (0, 0, 0))
    screen.blit(text_threshold, (margin*2.5, sliders_height + margin+5))

    text_speed = label_font.render('SPEED', True, (0, 0, 0))
    screen.blit(text_speed, (margin*2.5, sliders_height + margin+5 + control_height))


    for i in range(slider_num):
        x = margin + i * slider_width
        y = margin*2
        cx_2 = x + slider_width

        y1 = y + slider_height / 2.0 + (10 - num_sigmas) * slider_height / (num_sigmas * 2.0)
        y1 = int(y1)
        text_slider_value_5 = label_font.render('-5', True, (0, 0, 0))
        text_height = (text_slider_value_5.get_rect().height) / 2.0
        screen.blit(text_slider_value_5, (cx_2, y1-text_height))

        y1 = y + slider_height / 2.0 + (5 - num_sigmas) * slider_height / (num_sigmas * 2.0)
        y1 = int(y1)
        text_slider_value_0 = label_font.render('0', True, (0, 0, 0))
        text_height = (text_slider_value_0.get_rect().height) / 2.0
        screen.blit(text_slider_value_0, (cx_2, y1-text_height))

        y2 = y + slider_height / 2.0 + (0 - num_sigmas) * slider_height / (num_sigmas * 2.0)
        y2 = int(y2)
        text_slider_value_m5 = label_font.render('5', True, (0, 0, 0))
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
    global sonification_mode
    global current_midi_events
    global current_midi_size
    global current_file_index
    global next_song
    global current_params_statics
     
    # global steps


    print("Loading gaussian/pca statistics...")
    latent_means = np.load(dir_name + sub_dir_name + '/latent_means.npy')
    latent_stds = np.load(dir_name + sub_dir_name + '/latent_stds.npy')
    latent_pca_values = np.load(dir_name + sub_dir_name + '/latent_pca_values.npy')
    latent_pca_vectors = np.load(dir_name + sub_dir_name + '/latent_pca_vectors.npy')

    pca = PCA(n_components=len(latent_pca_values))
    pca.components_ = latent_pca_vectors
    pca.explained_variance_ = latent_pca_values
    pca.mean_ = latent_means

    random_sample = current_notes.reshape(-1)


    pygame.init()
    pygame.mixer.init()
    pygame.font.init()
    screen = pygame.display.set_mode((int(window_width), int(window_height)))
    pygame.display.set_caption('')

    audio_stream = audio.open(
        format=audio.get_format_from_width(2),
        channels=1,
        rate=sample_rate,
        output=True,
        frames_per_buffer=buffer_size,
        stream_callback=audio_callback)
    
    audio_stream.start_stream()

    running = True
    random_song_ix = 0
    max_random_songs = float('inf')
    # apply_controls()

    while running:
        # process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
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
                            max_random_songs = X_samples.shape[0]
                        except Exception as e:
                            print("This functionality is to check if the model training went well by reproducing an original song. "
                                  "The composer could not load samples and lengths from model training. "
                                  "If you have the midi files, the model was trained with, process them by using"
                                  " the preprocess_songs.py to find the requested files in data/interim "
                                  "(Load exception: {0}".format(e))


                    if songs_loaded:
                        print("Random Song Index: " + str(random_song_ix))
                        random_sample = X_samples[int(random_song_ix)]
                        current_params = pca.transform([random_sample])
                        random_song_ix = (random_song_ix + 1) % max_random_songs

                        # current_params = np.dot(latent_x - latent_means, latent_pca_vectors.T) / latent_pca_values  # REVISAR ESTO
                        needs_update = True
                        audio_reset = True

        if next_song: 
            print("Random Song Index: " + str(random_song_ix))
            random_sample = X_samples[int(random_song_ix)]
            current_params = pca.transform([random_sample])

            random_song_ix = (random_song_ix + 1) % max_random_songs
            needs_update = True
            audio_reset = True
            next_song = False

        if needs_update:
            # latent_x = latent_means + np.dot(current_params * latent_pca_values, latent_pca_vectors) # REVISAR ESTO
            current_params_statics = np.copy(current_params)
            reconstructed_x = pca.inverse_transform(current_params)
            reconstructed_x = np.reshape(reconstructed_x*255, (128, 4))
            current_notes = np.rint(reconstructed_x)
            current_midi_events = matrix2mid(current_notes.astype(int))
            current_midi_events = [msg for msg in current_midi_events]
            current_midi_size = len(current_midi_events)
            print(current_params)
            if audio_reset:
                current_file_index = 0
                audio_reset = False
            current_file_index = 0
            needs_update = False

        screen.fill(background_color)
        draw_sliders(screen)
        # draw_controls(screen)
        # draw_button(screen)
        # text_background(screen)
        # draw_text(screen)

        pygame.display.flip()
        pygame.time.wait(10)

    audio_stream.stop_stream()
    audio_stream.close()
    audio.terminate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CHANGE NAME')
    parser.add_argument('--model_path', type=str, help='The folder the PCA weights are stored.', required=True)

    args = parser.parse_args()
    sub_dir_name = args.model_path
    play()
