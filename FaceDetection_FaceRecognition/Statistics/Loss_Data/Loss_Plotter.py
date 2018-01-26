import pandas as pd

DATA = 'loss_adadelta_56_ID.csv'


def plot_line_chart(data, label_x='Epochs', label_y='Loss', title='Training Model'):
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.ylabel(label_y)
    plt.xlabel(label_x)
    plt.title(title)
    plt.show()




if __name__ == '__main__':
    data = pd.read_csv(DATA, quotechar='"', encoding='utf-8')
    plot_line_chart(data, title='Training Model ' + DATA[14:16])

