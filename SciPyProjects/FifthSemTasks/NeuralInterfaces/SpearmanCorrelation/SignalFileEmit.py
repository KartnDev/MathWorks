import numpy as np


def write_file_signal(signals, length, path):
    with open(path, "w") as o_file:
        for j in range(length):
            o_file.writelines(" ".join([str(item) if i % 2 == 0 else str(np.sin(j + i)) for i, item in enumerate(np.random.randn(signals))]))
            o_file.write("\n")

if __name__ == '__main__':
    path = "C:\\Users\\Dmitry\\Desktop\\Newfolder\\NewSig.txt"

    write_file_signal(8, 5000, path)
