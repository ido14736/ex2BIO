import random
import sys

import numpy as np


# TODO: delete all the lines "print(...)" from the code

# TODO: write again without duplicate code?
def parse_line(line, matrix_size, values_num):
    splited_line = line.split()
    if (values_num == 3):
        row = int(splited_line[0])
        col = int(splited_line[1])
        value = int(splited_line[2])

        # validates input
        if not (row >= 1 and row <= matrix_size) and (col >= 1 and col <= matrix_size) and (
                value >= 1 and value <= matrix_size):
            print("Invalid given number or given number position found.")
            return -1

        return row - 1, col - 1, value
    else:
        row_index1 = int(splited_line[0])
        col_index1 = int(splited_line[1])
        row_index2 = int(splited_line[2])
        col_index2 = int(splited_line[3])

        # validates input
        if not (row_index1 >= 1 and row_index1 <= matrix_size) and (col_index1 >= 1 and col_index1 <= matrix_size) and (
                row_index2 >= 1 and row_index2 <= matrix_size) and (col_index2 >= 1 and col_index2 <= matrix_size):
            print("Invalid given number or given number position found.")
            return -1

        return row_index1 - 1, col_index1 - 1, row_index2 - 1, col_index2 - 1


def parse_input(input_file_path):
    input_file = open(input_file_path, 'r')
    lines = input_file.readlines()

    # The size of the matrix
    matrix_size = int(lines[0])

    # validates input
    if not (matrix_size >= 4 and matrix_size <= 9):
        print("Please choose size between 4 and 9.")
        return -1

    # Number of given digits (0 if the matrix is given empty)
    given_digits_num = int(lines[1])

    # validates input
    if not (given_digits_num >= 0 and given_digits_num <= (matrix_size * matrix_size)):
        print("Invalid number of numbers given.")
        return -1

    given_digits = []
    for i in range(2, given_digits_num + 2):
        row, col, value = parse_line(lines[i], matrix_size, 3)
        given_digits.append([[row, col], value])

    # The number of “greater than” signs
    greater_than_num = int(lines[given_digits_num + 2])

    greater_than_signes_positions = []
    for j in range(given_digits_num + 3, given_digits_num + 3 + greater_than_num):
        row_index1, col_index1, row_index2, col_index2 = parse_line(lines[j], matrix_size, 4)
        greater_than_signes_positions.append([[row_index1, col_index1], [row_index2, col_index2]])

    return [matrix_size, given_digits, greater_than_signes_positions, greater_than_num]


# checks the state of the board
def validate_board_state(board):
    valid = True
    for i in range(len(board)):
        if len(board[i, :][board[i, :] != 0]) != len(set(board[i, :][board[i, :] != 0])) or len(
                board[:, i][board[:, i] != 0]) != len(set(board[:, i][board[:, i] != 0])):
            valid = False
    return valid


# by a position - returning a list of all the valid values for this position by the board state
def get_number_of_duplicates(board, size):
    num_of_duplicates = 0
    for i in range(len(board)):
        for current_num in range(size):
            current_count = 0
            for row_val in board[i, :]:
                #      print(row_val, " ", current_num+1)
                if row_val == current_num + 1:
                    current_count += 1
            if current_count > 0:
                current_count -= 1
            #  print(current_count)
            num_of_duplicates += current_count

            current_count = 0
            for col_val in board[:, i]:
                # print(col_val, " ", current_num + 1)
                if col_val == current_num + 1:
                    current_count += 1
            if current_count > 0:
                current_count -= 1
            # print(current_count)
            num_of_duplicates += current_count
    return num_of_duplicates
    # return np.setdiff1d(np.array(range(1, len(board)+1)), np.unique(np.concatenate((board[i, :], board[:, j]))))


# by a position - returning a list of all the valid values for this position by the board state
def get_valid_numbers_for_position(board, i, j):
    return np.setdiff1d(np.array(range(1, len(board) + 1)), np.unique(np.concatenate((board[i, :], board[:, j]))))


# initializing the Futoshiki board - adding random values to the empty positions
def initialize_board(size, given_numbers):
    # Create an empty 2D numpy array by the size
    board = np.full([size, size], 0)
    for num in given_numbers:
        board[num[0][0], num[0][1]] = num[1]
    if validate_board_state(board):
        for i in range(size):
            for j in range(size):
                if board[i, j] == 0:
                    # randomize from the valid values
                    # choice_options = get_valid_numbers_for_position(board, i, j)
                    # if np.array_equal(choice_options, []):
                    #   return np.array([-1])
                    board[i, j] = random.choice(range(1, len(board) + 1))

    else:
        print(
            "Invalid given numbers positions(there is more than one instance of the same number in the same row or column).")
        return np.array([])

    return board


def fitness_to_board(board, greater_than_signs_positions, greater_than_num, size):
    # count how many duplicate values there are in each col and row
    duplicates = get_number_of_duplicates(board, size)
    max_duplicates = 2 * (size * (size - 1))

    # iterate on 'greater_than_signs_positions' and count how many bad signs are there
    counter = 0
    for num in greater_than_signs_positions:
        # print(board[num[0][0], num[0][1]], " > ", board[num[1][0], num[1][1]])
        if not (board[num[0][0], num[0][1]] > board[num[1][0], num[1][1]]):
            counter += 1

    # calculate the grade of each evaluation parameter
    duplicates_grade = (duplicates / max_duplicates)
    greater_than_grade = (counter / greater_than_num)

    # TODO: change the % of each grade in order to get better results
    # calculates the final grade by giving each evaluation parameter different weight
    final_grade = (0.6 * duplicates_grade) + (0.4 * greater_than_grade)

    return final_grade


def fix_board(board, given_numbers):
    for num in given_numbers:
        board[num[0][0], num[0][1]] = num[1]


def combine_boards(size, board1, board2):
    new_board = np.full([size, size], 0)
    for i in range(size):
        for j in range(size):
            board_num = random.randint(1, 2)
            if (board_num == 1):
                new_board[i, j] = board1[i, j]
            else:
                new_board[i, j] = board2[i, j]

    return new_board


def create_mutation(size, board, given_numbers):
    new_board = np.full([size, size], 0)
    for i in range(size):
        for j in range(size):
            board_num = random.randint(1, 2)
            if (board_num == 1):
                new_board[i, j] = board[i, j]
            else:
                new_board[i, j] = random.randint(1, size)

    fix_board(new_board, given_numbers)

    return new_board


# remove 10% of the population
def create_generation(population):
    new_generation = []
    # sort the array according to the grades of each board
    population.sort(key=lambda x: x[1])
    population_size = int(0.57 * len(population))
    for i in range(population_size):
        new_generation.append(population[i])
        # print("Board Number:", i, "\nGrade:", new_generation[i][1])
        # print(new_generation[i][0], "\n")

    return new_generation


# main function
if __name__ == '__main__':

    # validates input
    if len(sys.argv) != 2:
        print("Not Enough Parameters!")
    else:
        input_file_path = sys.argv[1]

        # get the data from the input file
        data = parse_input(input_file_path)
        if data != -1:
            size = data[0]
            given_numbers = data[1]
            greater_than_signs_positions = data[2]
            greater_than_num = data[3]

        # TODO: check if need to change the number
        population_size = 100

        # create the population
        population = []
        for i in range(population_size):
            board = initialize_board(size, given_numbers)
            # initialize all boards with fitness grade
            board_grade = fitness_to_board(board, greater_than_signs_positions, greater_than_num, size)
            population.append([board, board_grade])
            # print("Board Number:", i, "\nGrade:", population[i][1])
            # print(population[i][0], "\n")

        # TODO: change the number of iterations
        iter_num = 100

        for iter in range(iter_num):
            if (len(population) == 1):
                print("END")
                break

            print("Iterations Number:", iter + 1)
            print("Population size:", len(population))

            # CROSSOVERS
            crossover_num = int(0.40 * len(population))
            for num in range(crossover_num):
                board1_num = random.randint(0, len(population) - 1)
                board2_num = random.randint(0, len(population) - 1)
                new_board = combine_boards(size, population[board1_num][0], population[board2_num][0])
                new_board_grade = fitness_to_board(new_board, greater_than_signs_positions, greater_than_num, size)
                population.append([new_board, new_board_grade])

            # Mutations
            mutations_num = int(0.25 * len(population))
            for num in range(mutations_num):
                board_num = random.randint(0, len(population) - 1)
                mutation_board = create_mutation(size, population[board_num][0], given_numbers)
                mutation_grade = fitness_to_board(new_board, greater_than_signs_positions, greater_than_num, size)
                population.append([mutation_board, mutation_grade])

            # New Generation
            population = create_generation(population)

            print("Best Grade", population[0][1])
            print("Best Board:\n", population[0][0], "\n")
