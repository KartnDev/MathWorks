import numpy


def bubble_sort(arr: list):
    n = len(arr)

    for i in range(n - 1):

        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:

                arr[j], arr[j + 1] = arr[j + 1], arr[j]


rand_list = numpy.random.randint(-100, 100, 15)

print("List: ", rand_list)
bubble_sort(rand_list)
print("SortedList: ", rand_list)


