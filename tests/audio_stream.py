import mido
import pyaudio
import numpy as np
import time

# Configurar el puerto MIDI para enviar los mensajes
outport = mido.open_output('IAC Driver Bus 1')

note_w = 4
note_h = 128
buffer_size = 64
sr = 44100
buffer_sec = (buffer_size / sr)

# Cargar los archivos MIDI que se van a reproducir
midi_files = [
    mido.MidiFile('./chorales_compositions/midi_200.mid'),
    mido.MidiFile('./chorales_compositions/midi_201.mid'),
    mido.MidiFile('./chorales_compositions/midi_202.mid')
]

midi_file = [msg for msg in midi_files[0]]

midi_size = len(midi_file) 
current_file_index = 0

current_time = 0

audio_notes = []
audio_time = 0
note_time = 0
note_time_dt = 0
audio_reset = False
audio_pause = False

change_audio = False 
index_to_change = 0

def audio_callback(in_data, frame_count, time_info, status):
    global midi_file
    global current_file_index
    global change_audio

    data = np.zeros((frame_count,), dtype=np.float32)
    # print(frame_count)
    dt = 0

    if current_file_index >= midi_size:
        current_file_index = 0
        change_audio = True

    current_notes = []
    for msg in midi_file[current_file_index:]:
        if msg.type == 'note_on' or msg.type == 'note_off':
            current_notes.append(msg)
            dt += (msg.time)
            print(dt, buffer_sec)
            if dt > buffer_sec:
                current_file_index += 1
                break
        current_file_index += 1
    
    start_time = time.time()
    input_time = 0.0
    for msg in current_notes:
        input_time += msg.time
        playback_time = time.time() - start_time
        duration_to_next_event = input_time - playback_time
        outport.send(msg)
        if duration_to_next_event > 0.0:
            time.sleep(duration_to_next_event)

    return data.tobytes(), pyaudio.paContinue

# Configurar PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(2),
                channels=2,
                rate=sr,
                output=True,
                frames_per_buffer=buffer_size,
                stream_callback=audio_callback)

# Iniciar el stream de audio
stream.start_stream()

# Esperar a que la transmisión termine
while stream.is_active():
    if change_audio:
        change_audio = False 
        index_to_change = (index_to_change + 1)%2
        midi_file = [msg for msg in midi_files[index_to_change]]
        midi_size = len(midi_file)
        current_file_index = 0

# Detener el stream de audio y cerrar PyAudio
stream.stop_stream()
stream.close()
p.terminate()
