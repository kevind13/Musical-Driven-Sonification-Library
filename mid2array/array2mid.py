import numpy as np
import mido


def array2mid(ary, tempo=None):
    '''
        Convert numpy array to MIDI file.
    '''
    # get the difference
    new_ary = np.concatenate([np.array([[0] * 88]), np.array(ary)], axis=0)
    changes = new_ary[1:] - new_ary[:-1]
    # create a midi file with an empty track
    mid_new = mido.MidiFile()
    track = mido.MidiTrack()
    mid_new.tracks.append(track)
    if tempo:
        track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    # track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    # add difference in the empty track
    last_time = 0
    for ch in changes:
        if set(ch) == {0}:  # no change
            last_time += 1
            continue

        on_notes = np.where(ch > 0)[0]
        on_notes_vol = ch[on_notes]
        off_notes = np.where(ch < 0)[0]
        first_ = True
        for n, v in zip(on_notes, on_notes_vol):
            new_time = last_time if first_ else 0
            track.append(mido.Message('note_on', note=n + 21, velocity=v, time=new_time))
            first_ = False
        for n in off_notes:
            new_time = last_time if first_ else 0
            track.append(mido.Message('note_off', note=n + 21, velocity=0, time=new_time))
            first_ = False
        last_time = 0
    return mid_new