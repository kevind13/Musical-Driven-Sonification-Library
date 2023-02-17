from typing import Optional
import numpy as np
import mido


def array2mid(ary, tempo=None, truncate_range: Optional[tuple[int, int]] = None, block_size: Optional[int] = None, velocity: Optional[int] = 0):
    '''
        Convert numpy array to MIDI file.


        Iniciando una serie 
    '''
    # get the difference

    notes_range = 88
    bottom_value = 21
    block_size = block_size or 1
    if truncate_range is not None:
        notes_range = truncate_range[1] - truncate_range[0]
        bottom_value = truncate_range[0]

    mid_new = mido.MidiFile()

    for track_number in range(4):
        new_ary = np.array(ary[track_number])

        track = mido.MidiTrack()
        
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

        if tempo:
            track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
        
        time_tuple=(None,0)

        for ch in new_ary:
            
            # print('where: ', np.where(ch > 0))
            on_notes = np.where(ch > 0)[0]
            # print('on_notes: ',on_notes)
            on_notes_vol = ch[on_notes]

            if len(on_notes) > 1:
                middle = len(on_notes) // 2
                on_notes = [on_notes[middle]]
                on_notes_vol = [on_notes_vol[middle]]

            if len(on_notes) == 0 and time_tuple[0] is not None:
                track.append(mido.Message('note_off', note=time_tuple[0] + bottom_value, velocity=0, time=time_tuple[1]))
                # print(f'Agregando: note_off, note={time_tuple[0] + bottom_value}, time={time_tuple[1]}, time tuple queda: ({None}, {block_size})')
                time_tuple = (None, block_size)
            elif len(on_notes) == 0 and time_tuple[0] is None:
                # print(f'No se agrega nota pues no hay nota y noz había nota antes, time tuple queda: ({None}, {time_tuple[1] + block_size})')
                time_tuple = (None, time_tuple[1] + block_size)
                
            elif on_notes[0] != time_tuple[0] and time_tuple[0] is not None:
                track.append(mido.Message('note_off', note=time_tuple[0] + bottom_value, velocity=0, time=time_tuple[1]))
                track.append(mido.Message('note_on', note=on_notes[0] + bottom_value, velocity=(velocity or on_notes_vol[0]), time=0))
                # print(f'Agregando: note_off, note={time_tuple[0] + bottom_value}, time={time_tuple[1]}, note_on, note={on_notes[0] + bottom_value}, time={0} y time tuple queda: ({on_notes[0]}, {block_size})')
                time_tuple = (on_notes[0], block_size)
                
            elif on_notes[0] != time_tuple[0] and time_tuple[0] is None:
                track.append(mido.Message('note_on', note=on_notes[0] + bottom_value, velocity=(velocity or on_notes_vol[0]), time=time_tuple[1]))
                # print(f'Agregando: note_on, note={on_notes[0] + bottom_value}, time={time_tuple[1]}, time tuple queda: ({on_notes[0]}, {block_size})')
                time_tuple = (on_notes[0], block_size)
            elif on_notes[0] == time_tuple[0]:
                # print(f'No se agrega nota pues no cambia la nota, time tuple queda: ({time_tuple[0]}, {time_tuple[1] + block_size})')
                time_tuple = (time_tuple[0], time_tuple[1] + block_size)

        if time_tuple[0] is not None:
            track.append(mido.Message('note_off', note=time_tuple[0] + bottom_value, velocity=0, time=time_tuple[1]))
        
        mid_new.tracks.append(track)

    return mid_new