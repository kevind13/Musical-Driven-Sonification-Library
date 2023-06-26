from typing import Optional
import numpy as np
import mido


def matrix2mid(ary, velocity: Optional[int] = 90, block_size: Optional[int] = 128, tempo: Optional[int] = None):
    '''
        Convert numpy array to MIDI file.
    '''
    mid_new = mido.MidiFile()

    for track_number in range(ary.shape[1]):
        new_ary = np.array(ary[:, track_number])

        track = mido.MidiTrack()
        
        track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))

        if tempo:
            track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))

        channels = [1,2,3,4]
        count = 0
        zeros = 0
        previous_val = None
        for i in range(len(new_ary)):
            if new_ary[i] == 0:
                ## If I have zeros then I count them to put them later.
                zeros += 1
            elif new_ary[i] == previous_val:
                ## If the value is equal to the previous one, I accumulate it
                count += 1
            else:
                ## when it changes value at some point and i have two options
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