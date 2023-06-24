from music21 import corpus, midi
import os

'''
Extraction of chorales adapted from
https://github.com/feynmanliang/bachbot
'''

def standardize_key(score):
    """Converts into the key of C major or A minor.

    Adapted from https://gist.github.com/aldous-rey/68c6c43450517aa47474
    """
    # major conversions
    majors = dict([("A-", 4), ("A", 3), ("B-", 2), ("B", 1), ("C", 0), ("C#", -1), ("D-", -1), ("D", -2), ("E-", -3),
                   ("E", -4), ("F", -5), ("F#", 6), ("G-", 6), ("G", 5)])
    minors = dict([("A-", 1), ("A", 0), ("B-", -1), ("B", -2), ("C", -3), ("C#", -4), ("D-", -4), ("D", -5), ("E-", 6),
                   ("E", 5), ("F", 4), ("F#", 3), ("G-", 3), ("G", 2)])

    # transpose score
    key = score.analyze('key')
    if key.mode == "major":
        halfSteps = majors[key.tonic.name]
    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]
    tScore = score.transpose(halfSteps)

    # transpose key signature
    for ks in tScore.flat.getKeySignatures():
        ks.transpose(halfSteps, inPlace=True)
    return tScore


def extract_SATB(score):
    """
    Extracts the Soprano, Alto, Tenor, and Bass parts from a piece. The returned score is guaranteed
    to have parts with names 'Soprano', 'Alto', 'Tenor', and 'Bass'.

    This method mutates its arguments.
    """
    ids = dict()
    ids['Soprano'] = {
        'Soprano',
        'S.',
        'Soprano 1',  # NOTE: soprano1 or soprano2?
        'Soprano\rOboe 1\rViolin1'
    }
    ids['Alto'] = {'Alto', 'A.'}
    ids['Tenor'] = {'Tenor', 'T.'}
    ids['Bass'] = {'Bass', 'B.'}
    id_to_name = {id: name for name in ids for id in ids[name]}
    for part in score.parts:
        if part.id in id_to_name:
            part.id = id_to_name[part.id]
        else:
            score.remove(part)
    return score


def iter_standardized_chorales():
    "Iterator over 4/4 Bach chorales standardized to Cmaj/Amin with SATB parts extracted."
    for score in corpus.chorales.Iterator(numberingSystem='bwv', returnType='stream'):
        if score.getTimeSignatures()[0].ratioString == '4/4':  # only consider 4/4
            yield extract_SATB(standardize_key(score))


def generate_chorales():
    it = iter_standardized_chorales()

    path = 'chorales_compositions'
    if not os.path.exists(path):
        os.makedirs(path)

    for index, value in enumerate(it):
        print(f'Index {index}, value: ', value)
        midi_test = midi.translate.streamToMidiFile(value)
        midi_test.open(f'{path}/midi_{index}.mid', 'wb')
        midi_test.write()
        midi_test.close()