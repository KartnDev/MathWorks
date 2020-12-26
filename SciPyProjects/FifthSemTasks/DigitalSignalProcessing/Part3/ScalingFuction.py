import math
from typing import Iterable, Sized


def read_file(value):
    try:
        elem_list = []
        with open(f'..\\Resources\\D{value}.txt') as file:
            for elem in file.read().split('\n'):
                elem_list.append(float(elem))
        return elem_list
    except FileNotFoundError:
        print('Wrong N')
        return False


class ScaleCheck(object):
    def __init__(self, n_upper_value: int):
        self.N = n_upper_value
        self.h_coefficient: Iterable and Sized = None

    def check_sqrt(self):
        h_sum = sum(self.h_coefficient)
        result = math.fabs(h_sum - math.sqrt(2))
        print(f'h_k - sqrt(2) = {result}')
        return result

    def check_delta(self):
        self.h_coefficient[len(self.h_coefficient):] = [0 for _ in range(2 * self.N - 2)]
        res = []
        for m in range(0, self.N):
            temp = 0
            for k in range(0, 2 * self.N):
                temp += (self.h_coefficient[k] * self.h_coefficient[k + 2 * m])
            res.append(temp)
        print(f'Check Delta = {res}')
        return res

    def check(self):
        self.h_coefficient = read_file(self.N)
        if self.h_coefficient:
            print(f'For N = {self.N}')
            self.check_sqrt()
            self.check_delta()
        return self


if __name__ == '__main__':
    ScaleCheck(2).check()
