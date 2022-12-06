import string
import numpy as np

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
        result[k] = int(msg[msg.rfind(k):].split(' ')[0].split('=')[1].translate(
            str.maketrans({a: None for a in string.punctuation})))

    return [result, on_]


def switch_note(last_state, note, velocity, on_=True):
    '''
        Changes the last_state (the state of the 88 note at the previous time step) based on new value of note, velocity, note on or note off. 
        The state of each time step contains 88 values.
        
        Piano has 88 notes, corresponding to note id 21 to 108, any note out of this range will be ignored
    '''
    result = [0 for _ in range(88)] if last_state is None else last_state.copy()
    if 21 <= note <= 108:
        result[note - 21] = velocity if on_ else 0
    return result


def get_new_state(new_msg, last_state):
    new_msg, on_ = msg2dict(str(new_msg))
    new_state = switch_note(last_state, note=new_msg['note'], velocity=new_msg['velocity'],
                            on_=on_) if on_ is not None else last_state
    return [new_state, new_msg['time']]


def track2seq(track):
    '''
        Converts each message in a track to a list of 88 values, and stores each list in the result list in order.
        
        Piano has 88 notes, corresponding to note id 21 to 108, any note out of the id range will be ignored
    '''
    result = []
    last_state, last_time = get_new_state(str(track[0]), [0] * 88)
    for i in range(1, len(track)):
        new_state, new_time = get_new_state(track[i], last_state)
        if new_time > 0:
            result += [last_state] * new_time
        last_state, last_time = new_state, new_time
    return result


def mid2arry(mid, min_msg_pct=0.1):
    '''
        Convert MIDI file to numpy array
    '''

    tracks_len = [len(tr) for tr in mid.tracks]
    print('Track len is:', tracks_len)
    min_n_msg = max(tracks_len) * min_msg_pct
    # convert each track to nested list
    all_arys = []
    for i in range(len(mid.tracks)):
        if len(mid.tracks[i]) > min_n_msg:
            ary_i = track2seq(mid.tracks[i])
            print(f'Sum of notes in track {i}:', np.sum(ary_i))
            all_arys.append(ary_i)
    # make all nested list the same length
    max_len = max([len(ary) for ary in all_arys])
    for i in range(len(all_arys)):
        if len(all_arys[i]) < max_len:
            all_arys[i] += [[0 for _ in range(88)] for _ in range(max_len - len(all_arys[i]))]
    all_arys = np.array(all_arys)
    all_arrays = all_arys.max(axis=0)
    # trim: remove consecutive 0s in the beginning and at the end
    sums = all_arrays.sum(axis=1)
    ends = np.where(sums > 0)[0]
    return all_arrays[min(ends):max(ends)], all_arys
