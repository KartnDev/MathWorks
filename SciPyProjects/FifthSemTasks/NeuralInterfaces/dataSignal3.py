import asyncio

import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

from NeuralInterfaces.dataSignal2 import read_signals


async def yield_signal(src_signal_open, src_signal_close):
    while True:
        new_sig_open = src_signal_open + 0.0
        new_sig_close = src_signal_close + 0.0

        batch_size = 50
        rand_index = np.random.randint(0, 105 - batch_size)

        open_batch = new_sig_open[rand_index: rand_index + batch_size]
        close_batch = new_sig_close[rand_index: rand_index + batch_size]

        normal_emit_open = 1
        normal_emit_close = 1

        open_batch += np.random.normal(0, normal_emit_open, open_batch.shape)
        close_batch += np.random.normal(0, normal_emit_close, close_batch.shape)
        for sig_open, sig_close in zip(open_batch, close_batch):
            yield sig_open, sig_close
            await asyncio.sleep(0.01)


async def main():
    o_signal_first, o_signal_second, o_signal_third = read_signals("Resource\\OpenEyes.asc")
    c_signal_first, c_signal_second, c_signal_third = read_signals("Resource\\ClosedEyes.asc")
    arr_open = []
    arr_close = []
    async for sig_open, sig_close in yield_signal(o_signal_second, c_signal_second):
        arr_open.append(sig_open)
        arr_close.append(sig_close)
        if len(arr_open) > 50:
            plt.plot(np.abs(np.fft.rfft(arr_open)) ** 2, label='open')
            plt.plot(np.abs(np.fft.rfft(arr_close)) ** 2, label='close')
            plt.legend()
            plt.show()
            arr_open = []
            arr_close = []


if __name__ == '__main__':
    asyncio.run(main())
