import random
import sys

import numpy as np


# parsing the lines from the input file that related to the values on the board
# values_num = 3 => a line of a given number, values_num = 4 => a line of a "greater than" sign
def parse_line(line, matrix_size, values_num):
    splited_line = line.split()
    # given number
    if (values_num == 3):
        row = int(splited_line[0])
        col = int(splited_line[1])
        value = int(splited_line[2])

        # validates input
        if not ((row >= 1 and row <= matrix_size) and (col >= 1 and col <= matrix_size) and (
                value >= 1 and value <= matrix_size)):
            print("Invalid given number or given number position found.")
            return -1

        return row - 1, col - 1, value

    # greater than sign
    else:
        row_index1 = int(splited_line[0])
        col_index1 = int(splited_line[1])
        row_index2 = int(splited_line[2])
        col_index2 = int(splited_line[3])

        # validates input
        if not ((row_index1 >= 1 and row_index1 <= matrix_size) and (col_index1 >= 1 and col_index1 <= matrix_size) and (
                row_index2 >= 1 and row_index2 <= matrix_size) and (col_index2 >= 1 and col_index2 <= matrix_size)):
            print("Invalid given number or given number position found.")
            return -1

        return row_index1 - 1, col_index1 - 1, row_index2 - 1, col_index2 - 1


# parsing the given input file from the user and returning all the data from the file
def parse_input(input_file_path):
    input_file = open(input_file_path, 'r')
    lines = input_file.readlines()

    # The size of the matrix
    matrix_size = int(lines[0])

    # validates matrix size
    if not (matrix_size >= 4 and matrix_size <= 9):
        print("Please choose size between 4 and 9.")
        return -1

    # Number of given digits (0 if the matrix is given initially empty)
    given_digits_num = int(lines[1])

    # validates number of given digits
    if not (given_digits_num >= 0 and given_digits_num <= (matrix_size * matrix_size)):
        print("Invalid number of numbers given.")
        return -1

    # getting the given digits positions with the value
    given_digits = []
    for i in range(2, given_digits_num + 2):
        row, col, value = parse_line(lines[i], matrix_size, 3)
        given_digits.append([[row, col], value])

    # The number of “greater than” signs
    greater_than_num = int(lines[given_digits_num + 2])

    # getting the positions of the "grater than" signs
    greater_than_signes_positions = []
    for j in range(given_digits_num + 3, given_digits_num + 3 + greater_than_num):
        row_index1, col_index1, row_index2, col_index2 = parse_line(lines[j], matrix_size, 4)
        greater_than_signes_positions.append([[row_index1, col_index1], [row_index2, col_index2]])

    return [matrix_size, given_digits, greater_than_signes_positions, greater_than_num]


# checks the state of the board  - if there are two similar values in the same line/column
# (not including 0 - an empty position)
def validate_board_state(board):
    valid = True
    for i in range(len(board)):
        if len(board[i, :][board[i, :] != 0]) != len(set(board[i, :][board[i, :] != 0])) or len(
                board[:, i][board[:, i] != 0]) != len(set(board[:, i][board[:, i] != 0])):
            valid = False
    return valid


# returning the total number of duplicates in row i and in column j
def get_number_of_duplicates_by_value_and_position(board, val, i, j):
    duplications_count = 0

    # checking for duplicates in row i
    current_count = 0
    for row_val in board[i, :]:
        if row_val == val:
            current_count += 1
    if current_count > 0:
        current_count -= 1

    # including in the total duplicates
    duplications_count += current_count

    # checking for duplicates in column j
    current_count = 0
    for col_val in board[:, j]:
        # print(col_val, " ", current_num + 1)
        if col_val == val:
            current_count += 1
    if current_count > 0:
        current_count -= 1

    # including in the total duplicates
    duplications_count += current_count

    return duplications_count


# returning the total number of duplicates in the whole board
def get_number_of_duplicates(board, size):
    num_of_duplicates = 0
    # checking duplicates for every possible value(1-size) in every row and column
    for i in range(len(board)):
        for current_num in range(size):
            current_count = get_number_of_duplicates_by_value_and_position(board, current_num+1, i, i)
            #including in total
            num_of_duplicates += current_count
    return num_of_duplicates


# initializing the Futoshiki board - setting the given numbers and adding random values to the empty positions
def initialize_board(size, given_numbers):
    # Create an empty 2D numpy array by the size
    board = np.full([size, size], 0)

    # setting the given numbers on the board
    for num in given_numbers:
        board[num[0][0], num[0][1]] = num[1]

    # if the given numbers are valid - filling the rest of the positions with random possible numbers(from 1 to size)
    if validate_board_state(board):
        for i in range(size):
            for j in range(size):
                if board[i, j] == 0:
                    board[i, j] = random.choice(range(1, len(board) + 1))

    else:
        print(
            "Invalid given numbers positions(there is more than one instance of the same number in the same row or column).")
        return np.array([])

    return board


# calculating the fitness value of a board
def fitness_to_board(board, greater_than_signs_positions, greater_than_num, size):
    # count how many duplicate values there are in each col and row
    duplicates = get_number_of_duplicates(board, size)
    max_duplicates = 2 * (size * (size - 1))

    # iterate on 'greater_than_signs_positions' and counts how many bad signs are there
    # (the condition of the sign is incorrect)
    counter = 0
    for num in greater_than_signs_positions:
        if not (board[num[0][0], num[0][1]] > board[num[1][0], num[1][1]]):
            counter += 1

    # calculate the grade of each evaluation parameter - the calculated value divided by the total
    duplicates_grade = (duplicates / max_duplicates)
    greater_than_grade = (counter / greater_than_num)

    # calculates the final grade by giving each evaluation parameter different weight
    final_grade = (0.6 * duplicates_grade) + (0.4 * greater_than_grade)

    return final_grade


# prints the given board with the given "grater than" signs
def print_board(board, greater_than_signs_positions, size):
    # creating the arrays for the lines "greater than" signs and for the column "greater than" signs
    signs_array_x = np.full([size, size-1], ' ')
    signs_array_y = np.full([size-1, size], ' ')
    for sign in greater_than_signs_positions:
        if sign[0][0] == sign[1][0] and sign[0][1] < sign[1][1]:
            signs_array_x[sign[0][0]][sign[0][1]] = '>'

        elif sign[0][0] == sign[1][0] and sign[0][1] > sign[1][1]:
            signs_array_x[sign[0][0]][sign[1][1]] = '<'

        elif sign[0][0] < sign[1][0] and sign[0][1] == sign[1][1]:
            signs_array_y[sign[0][0]][sign[0][1]] = 'v'

        elif sign[0][0] > sign[1][0] and sign[0][1] == sign[1][1]:
            signs_array_y[sign[1][0]][sign[0][1]] = '^'

    # printing the board
    print("----" * (size-1) + '---')
    for i in range(size):
        print('|', end='')
        for j in range(size):
            if j == size - 1:
                print(str(board[i][j]), end='')
            else:
                print(str(board[i][j]) + ' ', end='')
            if j < (size-1):
                print(signs_array_x[i][j] + ' ', end='')
            else:
                print('|')
        if i < (size-1):
            print('|', end='')
            for j in range(size):
                if j == size-1:
                    print(signs_array_y[i][j], end='')
                else:
                    print(signs_array_y[i][j] + '   ', end='')
            print('|')
    print("----" * (size-1) + '---')


# the crossover function
# the first half of lines will be of board1 and the second half of lines will be of board2
def crossover(board1, board2, size):
    combined_board = []
    for j in range(size):
        # the first half of lines
        if j < int(size / 2):
            combined_board.append(board1[j])
        # the second half of lines
        else:
            combined_board.append(board2[j])
    return np.array(combined_board)


# returning all the positions that there is a duplicate of the value in it's row/column
# and/or if the value makes v "greater than" sign invalid
def get_possible_bad_indexes(board, given_numbers_positions, greater_than_signs_positions, size):
    possible_bad_indexes = []
    # checking for duplicates for every position
    for i in range(size):
        for j in range(size):
            if ([i,j] not in given_numbers_positions) and (get_number_of_duplicates_by_value_and_position(board, board[i][j], i, j) > 0):
                possible_bad_indexes.append([i,j])

    # checking the positions for every "greater than" sign
    for positions_pair in greater_than_signs_positions:
        if not board[positions_pair[0][0]][[positions_pair[0][1]]] > board[positions_pair[1][0]][[positions_pair[1][1]]]:
            if [positions_pair[0][0], positions_pair[0][1]] not in given_numbers_positions and [positions_pair[0][0], positions_pair[0][1]] not in possible_bad_indexes:
                possible_bad_indexes.append([positions_pair[0][0], positions_pair[0][1]])
            if [positions_pair[1][0], positions_pair[1][1]] not in given_numbers_positions and [positions_pair[1][0], positions_pair[1][1]] not in possible_bad_indexes:
                possible_bad_indexes.append([positions_pair[1][0], positions_pair[1][1]])

    return possible_bad_indexes


# the optimization function
# the number of optimizations for every board will be as "size"
# every optimization - switching the values between two "bad positions"
def optimize(board, grade, given_numbers_positions, greater_than_signs_positions, greater_than_num, size):
    # if the board is finished - no optimization needed
    if grade > 0:
        opt_board = board.copy()

        # num of optimizations = size
        for i in range(size):
            possible_bad_indexes = get_possible_bad_indexes(board, given_numbers_positions, greater_than_signs_positions, size)
            # if there are less than two bad indexes - no optimization
            if len(possible_bad_indexes) < 2:
                return board.copy()
            # if there are two bad indexes - selecting them
            elif len(possible_bad_indexes) == 2:
                indexes_to_switch = possible_bad_indexes
            # if there are more than two bad indexes - selecting two randomly
            elif len(possible_bad_indexes) > 2:
                indexes_to_switch = random.sample(get_possible_bad_indexes(board, given_numbers_positions, greater_than_signs_positions, size), 2)

            # switching the values between the positions
            if len(possible_bad_indexes) >= 2:
                temp = opt_board[indexes_to_switch[0][0]][indexes_to_switch[0][1]]
                opt_board[indexes_to_switch[0][0]][indexes_to_switch[0][1]] = opt_board[indexes_to_switch[1][0]][indexes_to_switch[1][1]]
                opt_board[indexes_to_switch[1][0]][indexes_to_switch[1][1]] = temp

        # returning the updated board only if it's fitness is better than the original board
        if fitness_to_board(opt_board, greater_than_signs_positions, greater_than_num, size) < grade:
            return opt_board
        return board.copy()
    return board.copy()


# main function
if __name__ == '__main__':
    # validates input
    if len(sys.argv) != 2:
        print("Not Enough Parameters!")
    else:
        input_file_path = sys.argv[1]

        # get the data from the input file
        data = parse_input(input_file_path)

        # validating the input data
        if data != -1:
            size = data[0]
            given_numbers = data[1]
            greater_than_signs_positions = data[2]
            greater_than_num = data[3]

            # given_numbers_positions - the indexes on the board of the given numbers
            given_numbers_positions = []
            for given_num in given_numbers:
                given_numbers_positions.append(given_num[0])

            # creating the starting population for all the algorithms
            population_size = 200
            population = []
            grades = []
            # create the population - generating the boards and calculating their grades
            for i in range(population_size):
                board = initialize_board(size, given_numbers)
                board_grade = fitness_to_board(board, greater_than_signs_positions, greater_than_num, size)
                population.append([board, board_grade])
                grades.append(board_grade)

            # for darvin
            darvin_population = []
            darvin_grades = []

            # for lamark
            lemark_population = []
            lemark_grades = []

            # optimizing every initial board, calculating it's grade.
            # for darving - adding the original board with the optimized grade
            # for lemark - adding the optimized board with the optimized grade
            for board in population:
                optimized_board = optimize(board[0], board[1], given_numbers_positions, greater_than_signs_positions, greater_than_num, size)
                optimized_board_grade = fitness_to_board(optimized_board, greater_than_signs_positions, greater_than_num, size)
                darvin_population.append([board[0], optimized_board_grade])
                darvin_grades.append(optimized_board_grade)
                lemark_population.append([optimized_board, optimized_board_grade])
                lemark_grades.append(optimized_board_grade)

            mutations_num = 450
            # creating a list with the possible positions for mutations
            # includes all indexes instead of the given numbers indexes
            # and the indexes of the first two boards - will be the boards with the best grade
            possible_mutations_indexes = []
            for i in range(population_size - 2):
                for j in range(size):
                    for k in range(size):
                        if [j,k] not in given_numbers_positions:
                            possible_mutations_indexes.append([i+2, [j, k]])

            # calculating the weights for the boards pair selection for the crossovers
            # we will be selecting the 50 best boards and in every crossover we will select two
            # the selection from the 50 boards will be weighted by the calculated weights
            best_boards_num = 50
            weights = []
            r = list(range(51))
            r.remove(0)
            s = sum(r)
            for i in r:
                weights.insert(0, (i / s))

            iter_num = 200

            print("The original algorithm running")
            # the main loop for the original algorithm
            for iter in range(iter_num):
                grades_avg = sum(grades)/len(grades)
                current_best_board_grade = population[grades.index(min(grades))][1]
                print("iter number:", iter, "best grade:", current_best_board_grade, "avd:", grades_avg)

                best_boards_indexes = sorted(range(len(grades)), key=lambda k: grades[k])[:best_boards_num]

                next_generation_population = []

                for i in range(2):
                    next_generation_population.append(population[best_boards_indexes[i]])

                # CROSSOVERS
                # generate 98 more boards by the best 50 boards in the current population
                for i in range(population_size - 2):
                    chosen_indexes = np.random.choice(best_boards_indexes,size=2,replace=False, p=weights)
                    next_generation_population.append(crossover(population[chosen_indexes[0]][0], population[chosen_indexes[1]][0], size))

                # MUTATIONS
                mutations_positions = random.sample(possible_mutations_indexes, mutations_num)
                for mut in mutations_positions:
                    prev_value = next_generation_population[mut[0]][mut[1][0], mut[1][1]]
                    possible_new_values = list(range(size))
                    possible_new_values.remove(prev_value-1)
                    next_generation_population[mut[0]][mut[1][0], mut[1][1]] = random.choice(possible_new_values) + 1

                #the first two will be the same
                for i in range(2):
                    population[i] = next_generation_population[i]
                    grades[i] = next_generation_population[i][1]

                for i in range(population_size - 2):
                    current_board = next_generation_population[i+2]
                    current_board_grade = fitness_to_board(current_board, greater_than_signs_positions, greater_than_num, size)
                    population[i+2] = [current_board, current_board_grade]
                    grades[i+2] = current_board_grade

            print("The original algorithm done")
            print("the best board:")
            print_board(population[grades.index(min(grades))][0], greater_than_signs_positions, size)
            print("its grade:", population[grades.index(min(grades))][1])

            print("The darvin algorithm running")
            # the main loop for the darvin algorithm
            for iter in range(iter_num):
                darvin_avg = sum(darvin_grades) / len(darvin_grades)
                current_darvin_best_board_grade = darvin_population[darvin_grades.index(min(darvin_grades))][1]
                print("iter number:", iter, "best grade:", current_darvin_best_board_grade, "avd:", darvin_avg)

                best_darvin_boards_indexes = sorted(range(len(darvin_grades)), key=lambda k: darvin_grades[k])[
                                             :best_boards_num]

                next_generation_darvin_population = []

                for i in range(2):
                    next_generation_darvin_population.append(darvin_population[best_darvin_boards_indexes[i]])

                # CROSSOVERS
                # generate 98 more boards by the best 50 boards in the current population
                for i in range(population_size - 2):
                    chosen_darvin_indexes = np.random.choice(best_darvin_boards_indexes, size=2, replace=False,
                                                             p=weights)
                    next_generation_darvin_population.append(
                        crossover(darvin_population[chosen_darvin_indexes[0]][0],
                                  darvin_population[chosen_darvin_indexes[1]][0], size))

                # MUTATIONS
                mutations_positions = random.sample(possible_mutations_indexes, mutations_num)
                for mut in mutations_positions:
                    prev_darvin_value = next_generation_darvin_population[mut[0]][mut[1][0], mut[1][1]]
                    possible_new_darvin_values = list(range(size))
                    possible_new_darvin_values.remove(prev_darvin_value - 1)
                    next_generation_darvin_population[mut[0]][mut[1][0], mut[1][1]] = random.choice(
                        possible_new_darvin_values) + 1

                # the first two will be the same
                for i in range(2):
                    darvin_population[i] = next_generation_darvin_population[i]
                    darvin_grades[i] = next_generation_darvin_population[i][1]

                for i in range(population_size - 2):
                    current_darvin_board = next_generation_darvin_population[i + 2]
                    current_darvin_board_grade = fitness_to_board(current_darvin_board,
                                                                  greater_than_signs_positions,
                                                                  greater_than_num, size)
                    darvin_population[i + 2] = [current_darvin_board, current_darvin_board_grade]
                    darvin_grades[i + 2] = current_darvin_board_grade

                for i in range(len(population)):
                    source_board = darvin_population[i][0].copy()
                    darvin_optimized_board = optimize(darvin_population[i][0], darvin_population[i][1],
                                                      given_numbers_positions, greater_than_signs_positions,
                                                      greater_than_num, size)
                    darvin_optimized_board_grade = fitness_to_board(darvin_optimized_board,
                                                                    greater_than_signs_positions,
                                                                    greater_than_num, size)
                    darvin_population[i] = [source_board, darvin_optimized_board_grade]
                    darvin_grades[i] = darvin_optimized_board_grade

            print("The darvin algorithm done")
            print("the best board:")
            print_board(darvin_population[darvin_grades.index(min(darvin_grades))][0], greater_than_signs_positions, size)
            print("its grade:", darvin_population[darvin_grades.index(min(darvin_grades))][1])

            print("The lemark algorithm running")
            # the main loop for the lemark algorithm
            for iter in range(iter_num):
                lemark_avg = sum(lemark_grades) / len(lemark_grades)
                current_lamark_best_board_grade = lemark_population[lemark_grades.index(min(lemark_grades))][1]
                print("iter number:", iter, "best grade:", current_lamark_best_board_grade, "avg:", lemark_avg)

                best_lemark_boards_indexes = sorted(range(len(lemark_grades)), key=lambda k: lemark_grades[k])[
                                             :best_boards_num]

                next_generation_lemark_population = []

                for i in range(2):
                    next_generation_lemark_population.append(lemark_population[best_lemark_boards_indexes[i]])

                # CROSSOVERS
                # generate 98 more boards by the best 50 boards in the current population
                for i in range(population_size - 2):
                    chosen_lemark_indexes = np.random.choice(best_lemark_boards_indexes, size=2, replace=False,
                                                             p=weights)
                    next_generation_lemark_population.append(
                        crossover(lemark_population[chosen_lemark_indexes[0]][0],
                                  lemark_population[chosen_lemark_indexes[1]][0], size))

                # MUTATIONS
                mutations_positions = random.sample(possible_mutations_indexes, mutations_num)
                for mut in mutations_positions:
                    prev_lemark_value = next_generation_lemark_population[mut[0]][mut[1][0], mut[1][1]]
                    possible_new_lemark_values = list(range(size))
                    possible_new_lemark_values.remove(prev_lemark_value - 1)
                    next_generation_lemark_population[mut[0]][mut[1][0], mut[1][1]] = random.choice(
                        possible_new_lemark_values) + 1

                # the first two will be the same
                for i in range(2):
                    lemark_population[i] = next_generation_lemark_population[i]
                    lemark_grades[i] = next_generation_lemark_population[i][1]

                for i in range(population_size - 2):
                    current_lemark_board = next_generation_lemark_population[i + 2]
                    current_lemark_board_grade = fitness_to_board(current_lemark_board,
                                                                  greater_than_signs_positions,
                                                                  greater_than_num, size)
                    lemark_population[i + 2] = [current_lemark_board, current_lemark_board_grade]
                    lemark_grades[i + 2] = current_lemark_board_grade

                for i in range(len(population)):
                    lemark_optimized_board = optimize(lemark_population[i][0], lemark_population[i][1],
                                                      given_numbers_positions, greater_than_signs_positions,
                                                      greater_than_num, size)
                    lemark_optimized_board_grade = fitness_to_board(lemark_optimized_board,
                                                                    greater_than_signs_positions,
                                                                    greater_than_num, size)
                    lemark_population[i] = [lemark_optimized_board, lemark_optimized_board_grade]
                    lemark_grades[i] = lemark_optimized_board_grade

            print("The lemark algorithm done")
            print("the best board:")
            print_board(lemark_population[lemark_grades.index(min(lemark_grades))][0], greater_than_signs_positions, size)
            print("its grade:", lemark_population[lemark_grades.index(min(lemark_grades))][1])
