from typing import Optional
import mido
import os

def transpose_midi_file(input_midi, notes, save=Optional[bool], path=Optional[str]):
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
    if save and not path:
        output_midi.save('./output_files/exported_midi.mid')
    if save and path: 
        print(path)
        output_midi.save(path)
    return output_midi


def transpose_chorales():
    def files(path):
        for file in os.listdir(path):
            if os.path.isfile(os.path.join(path, file)):
                yield file

    notes = [1,2,3,4,5,6,-1,-2,-3,-4,-5]
    path = 'chorales_compositions'
    if not os.path.exists(path):
        raise Exception("No chorales to transpose")

    midi_failed = []

    for chorale in files(path):
        dir_name = chorale.split('.')[0]

        try:
            midi_to_transpose = mido.MidiFile(f'{path}/{chorale}')

            path_name = f'{path}/{dir_name}'
            if not os.path.exists(path_name):
                os.makedirs(path_name)

            for note in notes:
                transpose_midi_file(midi_to_transpose, note, save=True, path=f'./{path_name}/{dir_name}_{note}.mid')
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")
            midi_failed.append(chorale)
    print(midi_failed)