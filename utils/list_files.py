import os

def list_of_files(dir_name):
    '''
        For the given path, get the List of all files in the directory tree 
    '''
    list_of_file = os.listdir(dir_name)
    all_files = list()
    for entry in list_of_file:
        full_path = os.path.join(dir_name, entry)
        if os.path.isdir(full_path):
            all_files = all_files + list_of_files(full_path)
        else:
            all_files.append(full_path)
    return all_files

def list_of_files_no_depth(dir_name):
    list_of_file = os.listdir(dir_name)
    all_files = list()
    for entry in list_of_file:
        full_path = os.path.join(dir_name, entry)
        if not os.path.isdir(full_path):
            all_files.append(full_path)
    return all_files