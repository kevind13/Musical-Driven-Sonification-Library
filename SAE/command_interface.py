#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import warnings

from DB_PASS import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER

sys.path.append('../')

from mid2matrix.matrix2mid import matrix2mid

warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import io
import math
import threading
import time
from datetime import datetime, timedelta

import mido
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import DictCursor
import pyaudio
import scipy.io as sio
import torch
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

note_w = 4
note_h = 128

current_params = np.zeros((1,7), dtype=np.float32)
current_notes = np.zeros((note_h, note_w), dtype=np.uint8)
current_midi_events = matrix2mid(current_notes.astype(int), tempo=750000)
current_midi_events = [msg for msg in current_midi_events]
current_midi_size = len(current_midi_events) 

audio_reset = False
current_file_index = 0

port = mido.open_output('IAC Driver Bus 1')
# play_thread=None
_play=True
running = True


def play_midi():    
    global audio_reset
    global current_file_index

    while _play:
        start_time = time.time()
        input_time = 0.0 
        current_file_index = 0

        while current_file_index<len(current_midi_events) and _play:
            msg= current_midi_events[current_file_index]
            current_file_index+=1
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

def query_thread(delta_time, start_date, db_connection, column_names, model, current_params, all_principal_components_statistics, order_list, db_statistics):
    global running
    global _play
    global current_midi_events
    global audio_reset
    global current_file_index

    temp_current_params = current_params.copy()

    tmp_start_data = start_date
    while running:
        # print("Función ejecutándose...")

        # Ejecutar una consulta
        cursor = db_connection.cursor(cursor_factory=DictCursor)

        # query = "SELECT "
        # query += ", ".join([f'AVG({x})' for x in column_names]) 
        # query += " FROM data WHERE time BETWEEN %s AND %s"
        # query += "".join([f' AND {x} BETWEEN 0 and 1' for x in column_names])

        query = """
            SELECT sname, AVG(pm25) AS average_pm25 FROM data WHERE sname = ANY(%s) AND pm25 >= 0 AND pm25 < 100
            AND time BETWEEN %s AND %s GROUP BY sname;
        """

        initial_time = tmp_start_data
        final_time = tmp_start_data + timedelta(minutes=delta_time)
        print(column_names, initial_time, final_time)
        cursor.execute(query, (column_names, initial_time, final_time))
        results = cursor.fetchall()
        cursor.close()

        result = [row['average_pm25'] for row in results]
        print(result)

        tmp_start_data = tmp_start_data + timedelta(minutes=delta_time)


        ### CALCULAR EL TIEMPO ANTERIOR PARA SABER CUANTO TIEMPO SE PONE A DORMIR EXCTO

        latent_current = np.array(temp_current_params)
        # print(latent_current)

        for index, x in enumerate(order_list):
            z_latent = (latent_current[0][x] - all_principal_components_statistics[x]['x_mean']) / all_principal_components_statistics[x]['x_std']

            temp_column_avg = db_statistics[column_names_list[index]]['avg']
            temp_column_std = db_statistics[column_names_list[index]]['std']

            z_query = (result[index] - temp_column_std) / temp_column_avg

            # new_z_latent = z_latent + z_query/100
            new_z_latent = z_query
            
            latent_current[0][x] = new_z_latent * all_principal_components_statistics[x]['x_std'] + all_principal_components_statistics[x]['x_mean']

        print('real latent space:')
        print(current_params)
        print('modified latent space:')
        print(latent_current) 
        print('')

        latent_current = torch.from_numpy(latent_current)

        
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

        if current_midi_events[current_file_index].type == 'note_on' or current_midi_events[current_file_index].type == 'note_off':
            port.send(mido.Message('note_off', note=current_midi_events[current_file_index].note, velocity=64, time=0))
        current_midi_events = [msg for msg in temp_midi_events]
        # print(current_midi_events[current_file_index].note)
        current_file_index = 0

        ## REVISAR EL TEMPO Y CALCULAR LA DURACIÓN
        time.sleep(22)


if __name__ == "__main__":

    ### PARSER 

    parser = argparse.ArgumentParser(description="CHANGE NAME")
    parser.add_argument("--model_path", type=str, help="File path containing the model", default='results/SAE/SAE_3_Layer_7_Latent_9216_2560_512/SAE_3_Layer_7_Latent_9216_2560_512.tar')
    parser.add_argument("--latent_path", type=str, help="File path containing the latent use to sonify", default='results/SAE/SAE_3_Layer_7_Latent_9216_2560_512/latent/latent_project.mat')
    parser.add_argument("--all_latent_path", type=str, help="File path containig all the latent to get statistics", default='results/SAE/SAE_3_Layer_7_Latent_9216_2560_512/latent/all_latent_project.mat')

    parser.add_argument("--orders", nargs="+", type=int, help="List of components to be mapped (integers between 1 and 7)", default=[1,2,3])
    parser.add_argument("--column_names", nargs="+", type=str, help="List of station names", default=['no', 'no2', 'nox'])
    parser.add_argument("--start_date", type=str, help="Starting date (YYYY-MM-DD)")
    parser.add_argument("--delta_time", type=int, help="Delta time in minutes")

    parser.add_argument("--db_host", type=str, help="Database host", default=DB_HOST)
    parser.add_argument("--db_name", type=str, help="Database name", default=DB_NAME)
    parser.add_argument("--db_user", type=str, help="Database username", default=DB_USER)
    parser.add_argument("--db_password", type=str, help="Database password", default=DB_PASS)
    parser.add_argument("--db_port", type=str, help="Database port", default=DB_PORT)

    args = parser.parse_args()

    directory_path = args.model_path
    latent_path = args.latent_path
    all_latent_path = args.all_latent_path

    order_list = args.orders
    column_names_list = args.column_names
    start_date_value = datetime.strptime(args.start_date, "%Y-%m-%d")
    delta_time_value = args.delta_time

    db_host = args.db_host
    db_name = args.db_name
    db_user = args.db_user
    db_password = args.db_password
    db_port = args.db_port

    ### END PARSER    

    
    '''
    #this part of the code is to detect the stationsid, the patterns are gotten in Points but this gets the stationsid in the format of the table: data

    cur = db_connection.cursor(cursor_factory=DictCursor)
    
    points = [
        'Point(130.0889357 33.3001304)',
        'Point(130.4334872 33.5533525)',
        'Point(130.4349005 33.0344876)',
        'Point(130.0946999 33.0968652)',
        'Point(130.5006302 33.510286)',
        'Point(129.9041873 33.2769087)',
        'Point(130.3755655 33.3045177)',
        'Point(130.1504313 33.1849969)'
    ]

    query = """
        SELECT stationid
        FROM stationinfoupdated
        WHERE ST_DWithin(
            location::geography,
            ST_GeomFromText(%s)::geography,
            0.01
        );
    """

    stations_ids = []
    for point in points:
        cur.execute(query, (point,))
        results = cur.fetchall()

        # Process the results
        for row in results:
            stations_ids.append(row['stationid'])

    cur.close()

    ## Stations [41204010, 40134510, 41207010, 40221010, 41205080, 41425010]
    ['41204010', '40134510', '40202060', '41207010', '40221010', '41205080', '41210010', '41425010']
    '''


    # cursor = db_connection.cursor()
    # query = "SELECT "
    # query += ", ".join([f'AVG({x}), STDDEV({x})' for x in column_names_list])
    # query += " FROM data WHERE "
    # query += "AND ".join([f'{x} BETWEEN 0 and 1' for x in column_names_list])

    # print(query)

    # cursor.execute(query)
    # result = cursor.fetchone()
    # db_statistics = {}
    # for i in range(int(len(result) / 2)):
    #     db_statistics[column_names_list[i]] = {'avg': result[i*2], 'std': result[(i*2) + 1]}
    # cursor.close()
    # print(db_statistics)

    ## Testing stations

    db_connection = psycopg2.connect(dbname=db_name, user=db_user, password=db_password, host=db_host, port=db_port)
    cur = db_connection.cursor()

    query = """
        SELECT sname, AVG(pm25), STDDEV(pm25)
        FROM data
        WHERE sname IN %s and pm25 >= 0 and pm25 < 100
        GROUP BY sname;
    """
    cur.execute(query, (tuple(column_names_list),))
    results = cur.fetchall()

    db_statistics = {}

    for row in results:
        station = row[0]
        avg_value = row[1]
        std_value = row[2]
        db_statistics[station] = {'avg': avg_value, 'std': std_value}

    # Close the cursor and connection
    cur.close()

    print(db_statistics)
    ### END DB CONECTION AND STATISTICS

    ### MODEL LOADING
    print("Loading model...")
    checkpoint = torch.load(directory_path, map_location=torch.device('cpu'))  
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    ###

    ### LATENT LOADING
    project_mat = sio.loadmat(latent_path)
    all_project_mat = sio.loadmat(all_latent_path)

    principal_components = project_mat['latent_project']
    all_principal_components = all_project_mat['latent_project']
    all_principal_components = np.transpose(all_principal_components, (1,0))

    # print(all_principal_components.shape)
    
    all_principal_components_statistics = []
    z_scores_components = []
    for x in all_principal_components:
        z_scores = (x-np.mean(x))/np.std(x)
        z_scores_components.append(z_scores)
        all_principal_components_statistics.append({'x_min' : np.min(x), 'x_max': np.max(x), 'x_mean': np.mean(x), 'x_std': np.std(x), 'z_min' : np.min(z_scores), 'z_max' : np.max(z_scores), 'z_mean' : np.mean(z_scores), 'z_std': np.std(z_scores)})
    ###

    ### SELECTION OF SONG
    random_song_ix = 0 ## agregar randomness luego
    current_params = [principal_components[int(random_song_ix)]]

    message_thread = threading.Thread(target=query_thread, args=(delta_time_value, start_date_value, db_connection, column_names_list, model, current_params, all_principal_components_statistics, order_list, db_statistics))
    message_thread.start()

    play_thread=threading.Thread(target=play_midi)
    play_thread.start()

