import numpy as np

def binary_list_to_int(bin_list, n):
    assert len(bin_list) == n, f'bin_list is not {n} bits list'
    return int(''.join([str(value) for value in bin_list]), 2)

def flat_list(a):
    out = []
    for sublist in a:
        out.extend(sublist)
    return out

def int_list_to_binary_list(bin_list):
    temp_bin_list = [[int(x) for x in list('{:08b}'.format(value))] for value in bin_list]
    return flat_list(temp_bin_list)

def split_list_in_n_groups(L, n):
    assert len(L) % n == 0, f'L can not be group in groups of {n} exactly'
    temp_list = np.where(L>0, 1, 0)
    return zip(*(iter(temp_list),) * n)

def bin_to_int_array(bin_array):
    new_array = []
    for channel in bin_array:
        new_channel = []
        for row in range(channel.shape[1]):
            grouped_row = split_list_in_n_groups(channel[:, row], 8)
            byte_row = [binary_list_to_int(x, 8) for x in grouped_row]
            new_channel.append(byte_row)
        new_array.append(new_channel)

    new_array = np.array(new_array, np.uint8)
    new_array = np.transpose(new_array, (0, 2, 1))
    return new_array

def int_to_bit_array(int_array):
    new_array = []
    for channel in int_array:
        new_channel = []
        for row in range(channel.shape[1]):
            bits_row = int_list_to_binary_list(channel[:, row])
            new_channel.append(bits_row)
        new_array.append(new_channel)

    new_array = np.array(new_array, np.uint8)
    new_array = np.transpose(new_array, (0, 2, 1))  
    return new_array