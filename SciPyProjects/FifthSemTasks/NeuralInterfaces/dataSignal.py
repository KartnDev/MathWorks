import numpy as np
from matplotlib import pyplot as plt


def read_signals(path: str):
    first_signal = []
    second_signal = []
    third_signal = []
    with open(path) as file:
        file_lines = file.read().split('\n')[4:]
        for line in file_lines:
            values = line.split(' ')
            if len(values) == 3:
                first_signal.append(int(values[0]))
                second_signal.append(int(values[1]))
                third_signal.append(int(values[2]))

    first_signal = np.array(first_signal)
    second_signal = np.array(second_signal)
    third_signal = np.array(third_signal)

    return np.absolute(np.fft.rfft(first_signal)) ** 2, \
           np.absolute(np.fft.rfft(second_signal)) ** 2, \
           np.absolute(np.fft.rfft(third_signal)) ** 2


if __name__ == '__main__':
    open_eyes = read_signals("Resource\\OpenEyes.asc")
    closed_eyes = read_signals("../DigitalSignalProcessing/Resources/ClosedEyes.asc")
    plt.plot(open_eyes[0][3:105], label='open')
    plt.plot(closed_eyes[0][3:105], label='closed')
    plt.legend()
    plt.show()

    plt.plot(open_eyes[1][3:105], label='open')
    plt.plot(closed_eyes[1][3:105], label='closed')
    plt.legend()
    plt.show()

    plt.plot(open_eyes[2][3:105], label='open')
    plt.plot(closed_eyes[2][3:105], label='closed')
    plt.legend()
    plt.show()
