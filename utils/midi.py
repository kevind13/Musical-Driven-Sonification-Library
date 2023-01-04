import mido
import math

from utils.list_files import list_of_files, list_of_files_no_depth
from utils.statistics import draw_histogram, find_max_non_outlier


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
            if 'note_on' in str(track[i]) and i < total_len and 'note_off' not in str(track[i + 1]):
                print(midi_path, index)
                print(str(track[i - 10:i + 5]))
                print(str(track[i]) == str(track[i + 1]))
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
            if i == 0:
                continue
            delta = event.time
            if gcd_val == float('inf'):
                gcd_val = delta
            if delta == 0:
                continue
            gcd_val = math.gcd(gcd_val, delta)
            min_delta = min(min_delta, delta)

        gcd_and_min_delta_values[track.name] = [gcd_val, min_delta]

    list_of_gcd, list_of_min_deltas = zip(*gcd_and_min_delta_values.values())
    return math.gcd(*list_of_gcd), min(list_of_min_deltas)

def notes_range_midi(midi_path):
    mid = mido.MidiFile(midi_path)

    max_note, min_note = float('-inf'), float('inf')

    for track in mid.tracks:
        for i, event in enumerate(track):
            if 'note=' in str(event):
                tmp_note = event.note
                max_note = max(max_note, tmp_note)
                min_note = min(min_note, tmp_note)

    return max_note, min_note


def gcd_from_list_of_midi(midi_directory):
    list_of_paths = list_of_files_no_depth(midi_directory)
    list_of_gcd = []
    for midi_file in list_of_paths:
        try:
            temp_gcd, _ = gcd_and_min_delta(midi_file)
            list_of_gcd.append(temp_gcd)
        except:
            print(midi_file)
    return math.gcd(*list_of_gcd)


def notes_range_from_list_of_midi(midi_directory):
    list_of_paths = list_of_files_no_depth(midi_directory)
    max_note, min_note = float('-inf'), float('inf')
    for midi_file in list_of_paths:
        try:
            tmp_max, tmp_min = notes_range_midi(midi_file)
            max_note, min_note = max(max_note, tmp_max),  min(min_note, tmp_min)
        except:
            print(midi_file)
    return max_note, min_note, max_note - min_note


def len_histogram_of_midi(midi_directory):
    list_of_paths = list_of_files_no_depth(midi_directory)
    max_len = float('-inf')
    list_of_lengths = []
    max_file = ''
    for midi_path in list_of_paths:
        try:
            mid = mido.MidiFile(midi_path)
            tmp_max_len = float('-inf')
            for track in mid.tracks:
                tmp_len = 0
                for i in range(1, len(track)):
                    if 'note_on' in str(track[i]) or 'note_off' in str(track[i]):
                        tmp_len += track[i].time
                tmp_max_len = max(tmp_max_len, tmp_len)

            list_of_lengths.append((tmp_max_len, midi_path))
        except Exception as e:
            print(midi_path, e)

    draw_histogram([x[0] for x in list_of_lengths], title='Histogram of Length of MIDI files', x_label='Length of MIDI (ticks)', y_label='Number of MIDI files')
    max_non_outlier = find_max_non_outlier([x[0] for x in list_of_lengths])
    max_len, max_file = max(list_of_lengths)
    min_len, min_file = min(list_of_lengths)
    return ({
        'max_len': max_len,
        'max_file': max_file,
        'max_non_outlier': max_non_outlier,
        'min_len': min_len,
        'min_file': min_file
    })
