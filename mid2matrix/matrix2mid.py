from typing import Optional
import numpy as np
import mido


def matrix2mid(ary, velocity: Optional[int] = 90, block_size: Optional[int] = 128):
    '''
        Convert numpy array to MIDI file.
    '''
    mid_new = mido.MidiFile()

    for track_number in range(ary.shape[1]):
        new_ary = np.array(ary[:, track_number])

        track = mido.MidiTrack()
        
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

        channels = [1,2,3,4]
        count = 0
        zeros = 0
        previous_val = None
        for i in range(len(new_ary)):
            if new_ary[i] == 0:
                ## Si tengo ceros entonces los cuento y ya para ponerlos luego.
                zeros += 1
            elif new_ary[i] == previous_val:
                ## Si el valor es igual al anterior lo voy acumulando
                count += 1
            else:
                ## Entra cuando cambia de valor en algún momento y tengo dos opciones 
                if previous_val is not None:
                    track.append(mido.Message('note_on', note=previous_val, velocity=velocity, time=zeros*block_size, channel=channels[track_number]))
                    track.append(mido.Message('note_off', note=previous_val, velocity=0, time=count*block_size, channel=channels[track_number]))
                previous_val = new_ary[i]
                count = 1
                zeros = 0

        if previous_val is not None:
            track.append(mido.Message('note_on', note=previous_val, velocity=velocity, time=zeros*block_size, channel=channels[track_number]))
            track.append(mido.Message('note_off', note=previous_val, velocity=0, time=count*block_size, channel=channels[track_number]))

        mid_new.tracks.append(track)

    return mid_new