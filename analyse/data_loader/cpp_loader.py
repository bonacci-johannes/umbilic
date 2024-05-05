import glob
import numpy
import re


def read_data_file(file_path):
    # extract gamma, length, num from file_path
    file_name = file_path.split('/')[-1]
    file_sp = re.split(r"[_.]+", file_name)
    gamma = float(file_sp[2]) / 1000
    length = int(file_sp[4])
    num = int(file_sp[8])
    # t_max = int(file_sp[6])
    # seed = int(file_sp[0])

    # read data from file
    data = []
    with open(file_path, 'r') as file:
        # Skip the first 4 rows
        for _ in range(4):
            next(file)

        # Read each subsequent row
        for line in file:
            # Split the line into two integers
            values = line.strip().split()
            if len(values) == 2:
                try:
                    num1 = int(values[0])
                    num2 = int(values[1])
                    data.append((num1, num2))
                except ValueError:
                    print("Error: Skipping line with invalid data format:", line.strip())
            else:
                print("Error: Skipping line with invalid data format:", line.strip())

    # normalize and transform to eigen modes
    s_bar = numpy.array(data).T / (length * num * 2)
    s_eig = numpy.zeros_like(s_bar)
    s_eig[0, :] = (s_bar[0, :] + s_bar[1, :]) * (1 + numpy.sqrt(gamma))
    s_eig[1, :] = (s_bar[0, :] - s_bar[1, :]) * (1 + 1 / numpy.sqrt(gamma))

    return s_eig


def load_cpp_data(base_root, gamma, length, num, t_max):
    base_str = f'gam_{int(gamma * 1000)}_len_{length}_t_{t_max}_num_{num}'
    root_path = f'{base_root}/{base_str}'

    # find all files in root_path matching base_str pattern in filename
    record_files = sorted(glob.glob(f'{root_path}/*{base_str}.txt'))

    time = numpy.arange(1, t_max + 1)
    corr = numpy.zeros((len(record_files), 2, t_max))

    for n in range(len(record_files)):
        corr[n, :, :] = read_data_file(record_files[n])

    return corr, time
