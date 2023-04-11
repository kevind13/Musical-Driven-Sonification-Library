import string
from typing import Optional
import numpy as np
import mido
import os
from utils.list_files import list_of_files, list_of_files_no_depth
from scipy import sparse
from utils.matrix import bin_to_int_array
from utils.midi import gcd_and_min_delta, get_midi_key, get_same_key_midis


def msg2dict(msg):
    '''
        Extracts important information (note, velocity, time, on or off) from each message.
    '''
    result = dict()
    on_ = None
    if 'note_on' in msg:
        on_ = True
    elif 'note_off' in msg:
        on_ = False
    properties_to_find = ['time']

    if on_ is not None:
        properties_to_find.extend(['note', 'velocity'])

    for k in properties_to_find:
        value = msg[msg.rfind(k):].split(' ')[0].split('=')[1]
        value = value.translate(str.maketrans('', '', string.punctuation))
        result[k] = int(value)

    return [result, on_]


def track2seq(track, block_size: Optional[int] = None):
    result_seq = []
    last_note, last_status = None, None
    for msg in track:
        if type(msg) == mido.midifiles.meta.MetaMessage:
            continue
        result, on_ = msg2dict(str(msg))
        note = result.get('note', None)
        time = result.get('time', None)
        if note is not None:
            if last_note is not None and last_note == note and on_ == last_status:
                raise Exception('cant have two consecutive notes with same note type')
            if on_ and time != 0 and last_note != None:
                result_seq.extend([0]*int((time/block_size)))
            if not on_:
                if time is None: 
                    raise Exception('off values should have time because we are not using midis with notes at the same time')
                result_seq.extend([note]*int((time/block_size)))
            last_note, last_status = note, on_
    return result_seq


def mid2matrix(mid,
             block_size: Optional[int] = 128,
             truncate_length: Optional[bool] = 128):
    '''
        Convert MIDI file to numpy matrix
    '''

    # convert each track to nested list
    all_arys = []
    for tr in mid.tracks:
        ary = track2seq(tr, block_size=block_size)
        all_arys.append(ary)

    n_shorts = 0
    for ary in range(len(all_arys)):
        if len(all_arys[ary]) < truncate_length:
            all_arys[ary].extend([0] * truncate_length - len(ary))
            n_shorts+=1
        elif len(all_arys[ary]) > truncate_length:
            all_arys[ary] = all_arys[ary][:truncate_length]
    if n_shorts == 4:
        raise Exception('midi to short, it wont be used')
    all_arys = np.array(all_arys, np.uint8)
    return np.transpose(all_arys, (1,0))


def create_timeseries_dataset(path, key: Optional[str] =  None, transpose: Optional[bool] = True):
    from scipy.io import savemat

    if transpose:
        list_of_midi_files = list_of_files(path)
    else: 
        list_of_midi_files = list_of_files_no_depth(path)

    if key is not None:
        list_of_midi_files = get_same_key_midis(path, key)

    number_of_midi_files = len(list_of_midi_files)

    np_path = 'midi_np_dataset'
    midi_arrays = []
    if not os.path.exists(np_path):
        os.makedirs(np_path)

    for index, midi_file in enumerate(list_of_midi_files):
        if '.DS_Store' in midi_file:
            continue
        tmp_midi = mido.MidiFile(midi_file)
        block_size, _ = gcd_and_min_delta(tmp_midi, path=False)
        try:
            midi_array = mid2matrix(tmp_midi, block_size=block_size)
        except Exception as e:
            print(e)
            continue
        # midi_array = bin_to_int_array(midi_array)
        if midi_array.shape != (128,4):
            print(f'Different dimension in {midi_file}')
            continue

        midi_arrays.append(midi_array)

        
        print(f'{index}/{number_of_midi_files} midi:{midi_file}')
    midi_arrays = np.array(midi_arrays)
    file_name = f'{np_path}/timeseries_midi_dataset_with_same_key.mat'    
    savemat(file_name, {'train_data': midi_arrays})