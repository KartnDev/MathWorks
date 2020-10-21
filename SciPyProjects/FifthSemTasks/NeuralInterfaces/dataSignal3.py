import socket
import numpy as np
import matplotlib.pyplot as plt
from mne.time_frequency import morlet

from NeuralInterfaces.dataSignal import read_signals

HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 1337  # The port used by the server


def inf_list_draw(data, fig, sample):
    fig.canvas.draw()
    fig.clf()
    plt.ylim(bottom=0, top=1e8)
    plt.plot(np.abs(np.fft.fft(data)) ** 2, color='green')
    plt.plot(sample, color='red')


def inf_draw(data, fig):
    fig.canvas.draw()
    fig.clf()
    plt.plot(data, color='green')


def inf2(data, fig):
    fig.canvas.draw()
    fig.clf()
    sf = 36
    cf = 13  # Central spindles frequency in Hz
    nc = 12  # Number of oscillations in the spindles
    times = np.arange(data.size) / sf
    wlt = morlet(sf, [cf], n_cycles=nc)[0]
    analytic = np.convolve(data, wlt, mode='same')
    magnitude = np.abs(analytic)
    phase = np.angle(analytic)

    # Square and normalize the magnitude from 0 to 1 (using the min and max)
    power = np.square(magnitude)
    norm_power = (power - power.min()) / (power.max() - power.min())

    # Define the threshold
    thresh = 0.25

    # Find supra-threshold values
    supra_thresh = np.where(norm_power >= thresh)[0]

    # Create vector for plotting purposes
    val_spindles = np.nan * np.zeros(data.size)
    val_spindles[supra_thresh] = data[supra_thresh]

    plt.plot(times, norm_power)
    #plt.set_xlabel('Time [sec]')
    #plt.set_ylabel('Normalized wavelet power')
    #plt.axhline(thresh, ls='--', color='indianred', label='Threshold')
    plt.fill_between(times, norm_power, thresh, where=norm_power >= thresh,
                     color='indianred', alpha=.8)
    #plt.legend(loc='best')


def main():
    sigFirst = read_signals("Resource\\ClosedEyes.asc")[0][3:103]

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))

        fig = plt.gcf()
        fig.show()
        fig.canvas.draw()

        data = list(s.recv(1024)[1:100])
        # plt.plot(data)

        plt.ion()  # enable interactivity

        while True:
            new = list(s.recv(202)[1:100])
            for i in new:
                data.append(i)
                inf2(np.array(data[-100:]), fig)
                #inf_list_draw(data[-100:], fig, sigFirst)


if __name__ == '__main__':
    main()
