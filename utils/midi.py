import mido
import math

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
            if i == 0: continue
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
