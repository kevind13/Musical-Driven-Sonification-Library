import string
from typing import Optional
import numpy as np
import mido
import os
from utils.list_files import list_of_files, list_of_files_no_depth
from scipy import sparse


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


def switch_note(last_state, note, velocity, on_=True, truncate_range: Optional[tuple[int, int]] = None):
    '''
        Changes the last_state (the state of the 88 note at the previous time step) based on new value of note, velocity, note on or note off. 
        The state of each time step contains 88 values.
        
        Piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
        If truncate_range is not None then the range will be different, i.e, 38 to 80.
    '''
    notes_range = 88
    bottom_value = 21
    top_value = 108
    if truncate_range is not None:
        notes_range = truncate_range[1] - truncate_range[0]
        bottom_value = truncate_range[0]
        top_value = truncate_range[1]

    result = [0] * notes_range if last_state is None else last_state.copy()
    if bottom_value <= note <= top_value:
        result[note - bottom_value] = velocity if on_ else 0
    return result


def get_new_state(new_msg, last_state, truncate_range: Optional[tuple[int, int]] = None):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = last_state
    if on_ is not None:
        new_state = switch_note(last_state,
                                note=new_msg['note'],
                                velocity=new_msg['velocity'],
                                on_=on_,
                                truncate_range=truncate_range)
    return [new_state, new_msg['time']]


def track2seq(track, block_size: Optional[int] = None, truncate_range: Optional[tuple[int, int]] = None):
    '''
        Converts each message in a track to a list of 88 values, and stores each list in the result list in order.
        
        Piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    '''
    result = []
    notes_range = 88 if truncate_range is None else truncate_range[1] - truncate_range[0]

    last_state, last_time = get_new_state(str(track[0]), [0] * notes_range)
    for msg in track[1:]:
        new_state, new_time = get_new_state(msg, last_state, truncate_range=truncate_range)
        if new_time > 0:
            replicate_time = new_time // block_size if block_size else new_time
            result += [last_state] * replicate_time
        last_state, last_time = new_state, new_time
    return result


def mid2arry(mid,
             min_msg_pct=0.1,
             block_size: Optional[int] = None,
             truncate_range: Optional[tuple[int, int]] = None,
             fixed_len: Optional[int] = None,
             truncate_length: Optional[bool] = None):
    '''
        Convert MIDI file to numpy array
    '''

    tracks_len = [len(tr) for tr in mid.tracks]
    min_n_msg = max(tracks_len) * min_msg_pct

    # convert each track to nested list
    all_arys = []
    for tr in mid.tracks:
        if len(tr) > min_n_msg:
            ary = track2seq(tr, block_size=block_size, truncate_range=truncate_range)
            all_arys.append(ary)

    max_len = fixed_len if fixed_len is not None else max([len(ary) for ary in all_arys])
    notes_range = 88 if truncate_range is None else truncate_range[1] - truncate_range[0]

    # make all nested list the same length
    for ary in all_arys:
        if len(ary) < max_len:
            ary += [[0] * notes_range for _ in range(max_len - len(ary))]
        elif len(ary) > max_len:
            assert not (truncate_length == False and fixed_len is not None), 'This file is greater than max_len and can not be truncated.'
            ary = ary[:max_len]

    all_arys = np.array(all_arys, np.uint8)
    all_arrays = all_arys.max(axis=0)
    return all_arrays, all_arys


def create_all_arrays(path):
    list_of_midi_files = list_of_files(path)
    number_of_midi_files = len(list_of_midi_files)
    np_path = 'midi_np_dataset'
    np_4d_path = 'midi_4d_np_dataset'
    if not os.path.exists(np_path):
        os.makedirs(np_path)
    if not os.path.exists(np_4d_path):
        os.makedirs(np_4d_path)
    for index, midi_file in enumerate(list_of_midi_files):
        dir_name = midi_file.split('/')[-1].split('.')[0]
        file_name_np = f'{np_path}/{dir_name}.npz'
        file_name_np_4d = f'{np_4d_path}/{dir_name}.npz'

        tmp_midi = mido.MidiFile(midi_file)
        midi_array, _ = mid2arry(tmp_midi)

        sparse_midi_array = sparse.csr_matrix(midi_array)
        sparse.save_npz(file_name_np, sparse_midi_array)
        # sparse.save_npz(file_name_np_4d, sparse_midi_array)
        print(f'Successfully saved {midi_file}, progression: {index+1}/{number_of_midi_files}')
        break
        # TODO Continue with all process, but i need to know if this is necesary


def create_exploratory_row_data(path):
    from utils.constants import MIDI_ARRAY_MAX_LEN, MIDI_BOTTOM_NOTE, MIDI_GCD_TIME, MIDI_NOTE_RANGE, MIDI_TOP_NOTE
    from scipy.io import savemat

    list_of_midi_files = list_of_files_no_depth(path)
    number_of_midi_files = len(list_of_midi_files)
    np_path = 'midi_np_dataset'
    midi_arrays = []
    if not os.path.exists(np_path):
        os.makedirs(np_path)
    for index, midi_file in enumerate(list_of_midi_files):
        tmp_midi = mido.MidiFile(midi_file)
        _, midi_array = mid2arry(tmp_midi, block_size=MIDI_GCD_TIME, truncate_range=(MIDI_BOTTOM_NOTE,MIDI_TOP_NOTE), fixed_len=MIDI_ARRAY_MAX_LEN)
         
        midi_1d_array = midi_array.reshape(-1,)
        midi_arrays.append(midi_1d_array)
        print(midi_array.shape, midi_1d_array.shape)
        if index == 40:
            break
    midi_arrays = np.array(midi_arrays)
    savemat(f'{np_path}/exploratory_data.mat', {'train_data': midi_arrays})