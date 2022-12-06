# SonificationThesis
Implementation of master's thesis


# Questions
1. Use arrays of (n_rows, 88 columns), in this case we join all the tracks in a single array, if two sound track on the same note at the same time, it takes the larger velocity. When decoding we cannot identify the different tracks, so the resulting MIDI will be a composition with only one track.

2. Use arrays of (n*tracks, n_rows, 88 columns). In this case it is possible to keep the different tracks and thus be able to decode a musical composition with its different tracks.

3. To train the models we must have normalized all the input so that all the arrays must have the same size, there are three ideas for this case:

    a. Cut the arrays to a number of rows that depends on the shortest MIDI.

    b. Average the length of all files and fill with zeros at the beginning and at the end of the files that do not reach the set size.

    c. Add as many zeros as are missing to get arrays of the same size as the longest MIDI.


    Note: In arrays where zeros are added, we will remove the zeros when decoding. The problem is that the model will learn from these zeros that do not give information. In case there are a large number of consecutive zeros and then the model adds new notes, the composition should end there.