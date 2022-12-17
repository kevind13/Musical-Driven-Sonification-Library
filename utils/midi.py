import mido
import math

from utils.list_files import list_of_files

def get_tempo(mid):
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return msg.tempo
    else:
        return None


def cosecutive_on_off(midi_path):
    # Open the MIDI file
    mid = mido.MidiFile(midi_path)

    for index, track in enumerate(mid.tracks):
        total_len = len(track)
        for i in range(1, total_len):
            if 'note_on' in str(track[i]) and i < total_len and 'note_off' not in str(track[i+1]):
                print(midi_path, index)
                print(str(track[i-10: i+5]))
                print(str(track[i]) == str(track[i+1]))
                return False
    return True

def gcd_and_min_delta(midi_path):
    import math
    mid = mido.MidiFile(midi_path)

    gcd_and_min_delta_values = {}

    for track in mid.tracks:
        gcd_val = float('inf')
        min_delta = float('inf')
        
        for i, event in enumerate(track):
            if i == 0: continue
            delta = event.time
            if gcd_val == float('inf'): gcd_val = delta
            if delta == 0: continue
            gcd_val = math.gcd(gcd_val, delta)
            min_delta = min(min_delta, delta)
        
        gcd_and_min_delta_values[track.name] = [gcd_val, min_delta]
    
    list_of_gcd, list_of_min_deltas = zip(*gcd_and_min_delta_values.values())
    return math.gcd(*list_of_gcd), min(list_of_min_deltas)

def gcd_from_list_of_midi(midi_directory):
    list_of_paths = list_of_files(midi_directory)
    list_of_gcd = []
    for midi_file in list_of_paths:
        try: 
            temp_gcd, _ = gcd_and_min_delta(midi_file)
            list_of_gcd.append(temp_gcd)
        except:
            print(midi_file)
    return math.gcd(*list_of_gcd)

def max_len_of_midi(midi_directory):
    list_of_paths = list_of_files(midi_directory)
    max_len = float('-inf')
    for midi_path in list_of_paths:
        try: 
            mid = mido.MidiFile(midi_path)
            tmp_max_len = float('-inf')
            for track in mid.tracks:
                tmp_len = 0
                for i in range(1, len(track)):
                    if 'note_on' in str(track[i]) or 'note_off' in str(track[i]):
                        tmp_len += track[i].time
                tmp_max_len=max(tmp_max_len, tmp_len)
                
            max_len = max(max_len, tmp_max_len)
        except:
            print(midi_path)
    return(max_len)