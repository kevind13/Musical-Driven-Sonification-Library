from typing import Optional
import mido


def transpose_midi_file(midi_file, notes, save=Optional[bool]):
    input_midi = mido.MidiFile(midi_file)
    output_midi = mido.MidiFile()
    output_midi.ticks_per_beat = input_midi.ticks_per_beat
    for original_track in input_midi.tracks:
        new_track = mido.MidiTrack()
        for msg in original_track:
            if msg.type in ['note_on', 'note_off']:
                origin_note = msg.note
                new_track.append(msg.copy(note=origin_note + notes))
            else:
                new_track.append(msg)
        output_midi.tracks.append(new_track)
        if save:
            output_midi.save('./output_files/exported_midi.mid')
    return output_midi
