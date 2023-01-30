import numpy as np

def binary_list_to_int(bin_list, n):
    assert len(bin_list) == n, f'bin_list is not {n} bits list'
    return int(''.join([str(value) for value in bin_list]), 2)

def split_list_in_n_groups(L, n):
    assert len(L) % n == 0, f'L can not be group in groups of {n} exactly'
    temp_list = np.where(L>0, 1, 0)
    return zip(*(iter(temp_list),) * n)

def bin_to_byte_array(bin_array):
    new_array = []
    for channel in bin_array:
        new_channel = []
        for row in range(channel.shape[1]):
            grouped_row = split_list_in_n_groups(channel[:, row], 8)
            byte_row = [binary_list_to_int(x, 8) for x in grouped_row]
            if sum(byte_row) > 0:
                print(byte_row)
            new_channel.append(byte_row)
        new_array.append(new_channel)

    new_array = np.array(new_array)

    return new_array
