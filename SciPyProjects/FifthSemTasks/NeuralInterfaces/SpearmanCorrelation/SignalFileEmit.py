import numpy as np


def write_file_signal(signals, length, path):
    with open(path, "w") as o_file:
        for j in range(length):
            o_file.writelines("\t".join([str(item) if i % 2 == 0 else str(np.sin(j)) for i, item in enumerate(np.random.randn(signals))]))
            o_file.write("\n")

if __name__ == '__main__':
    path = "C:\\Users\\Dmitry\\Desktop\\NewSig.txt"

    write_file_signal(30, 5000, path)
