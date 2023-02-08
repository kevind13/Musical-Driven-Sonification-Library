from typing import Optional
import matplotlib.pyplot as plt

def multiple_plots(info_dict_list,
                   title: Optional[str] = None,
                   x_label: Optional[str] = None,
                   y_label: Optional[str] = None,
                   log_plot: Optional[bool] = False,
                   from_one: Optional[bool] = False):
    '''
        info_dict_list: List with dictionaries with all the info to plot, should have at least the name of the data and a label key. Example: 
            [{'train_loss': {'data': [...], 'label': 'Training Loss' }}, {'train_content_loss': {'data': [...], 'label': 'train_content_loss' }}]
    '''

    base = 0 if not from_one else 1

    number_of_plots = len(info_dict_list)

    n_cols = min(4, number_of_plots)
    n_rows = number_of_plots // n_cols
    if number_of_plots % n_cols > 0:
        n_rows = (number_of_plots // n_cols) +1
    print(n_rows, n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 8))
    for index, ax in enumerate(axs.flat):
        for key in info_dict_list[index]:
            ax.plot(info_dict_list[index][key]['data'][base:], label=info_dict_list[index][key]['label'])
        ax.legend()

    if log_plot:
        plt.yscale('log')
    
    plt.tight_layout()
    plt.show()