import sys
import numpy as np
import random

# returning the board size, the given numbers, the "greater that" signs from the input file
# and checking if the input values are valid
def handle_input_file(input_file_path):
    input_file = open(input_file_path, 'r')
    lines = input_file.readlines()

    # initializing the parameters by the input of the user
    size = int(lines[0])
    if not(size >= 4 and size <= 9):
        print("Please choose size between 4 and 9.")
        return -1
    given_numbers_num = int(lines[1])
    if not(given_numbers_num >= 0 and given_numbers_num <= (size*size)):
        print("Invalid number of numbers given.")
        return -1
    given_numbers = []
    for i in range(2, given_numbers_num + 2):
        splited_line = lines[i].split()
        if int(splited_line[0]) >= 1 and int(splited_line[1]) >= 1 and int(splited_line[2]) >= 1 and int(splited_line[0]) <= size and int(splited_line[1]) <= size and int(splited_line[2]) <= size:
            given_numbers.append([[int(splited_line[0]) - 1, int(splited_line[1]) - 1], int(splited_line[2])])
        else:
            print("Invalid given number or given number position found.")
            return -1

    greater_than_signs_num = int(lines[2 + given_numbers_num])
    if not (greater_than_signs_num >= 0 and greater_than_signs_num <= ((size-1) * (size-1))):
        print("Invalid number of 'greater than' signs given.")
        return -1
    greater_than_signes_positions = []
    for i in range(3 + given_numbers_num, 3 + given_numbers_num + greater_than_signs_num):
        splited_line = lines[i].split()
        if int(splited_line[0]) >= 1 and int(splited_line[1]) >= 1 and int(splited_line[2]) >= 1 and int(splited_line[3]) >= 1 and int(splited_line[0]) <= size and int(splited_line[1]) <= size and int(splited_line[2]) <= size and int(splited_line[3]) <= size:
            if (int(splited_line[0]) == int(splited_line[2]) and abs(int(splited_line[1]) - int(splited_line[3])) == 1) or (int(splited_line[1]) == int(splited_line[3]) and abs(int(splited_line[0]) - int(splited_line[2])) == 1):
                greater_than_signes_positions.append([[int(splited_line[0]) - 1, int(splited_line[1]) - 1], [int(splited_line[2]) - 1, int(splited_line[3]) - 1]])
            else:
                print("Invalid 'greater than' sign positions pair found. Please choose two positions that are near each other(but not diagonal).")
                return -1

        else:
            print("Invalid 'greater than' sign position found.")
            return -1

    return [size, given_numbers, greater_than_signes_positions]

# checks the state of the board
def validate_board_state(board):
    valid = True
    for i in range(len(board)):
        if len(board[i,:][board[i,:]!=0]) != len(set(board[i,:][board[i,:]!=0])) or len(board[:,i][board[:,i]!=0]) != len(set(board[:,i][board[:,i]!=0])):
            valid = False
    return valid

# by a position - returning a list of all the valid values for this position by the board state
def get_valid_numbers_for_position(board, i, j):
    return np.setdiff1d(np.array(range(1, len(board)+1)), np.unique(np.concatenate((board[i, :], board[:, j]))))

#initializing the Futoshiki board - adding random values to the empty positions
def initialize_board(size, given_numbers):
    # Create an empty 2D numpy array by the size
    board = np.full([size, size], 0)
    for num in given_numbers:
        board[num[0][0], num[0][1]] = num[1]
    if validate_board_state(board):
        for i in range(size):
            for j in range(size):
                if board[i,j] == 0:
                    # randomize from the valid values
                    choice_options = get_valid_numbers_for_position(board, i, j)
                    if np.array_equal(choice_options, []):
                        return np.array([-1])
                    board[i,j] = random.choice(choice_options)

    else:
        print("Invalid given numbers positions(there is more than one instance of the same number in the same row or column).")
        return np.array([])

    return board

#main function
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Not Enough Parameters!")
    else:
        input_file_path = sys.argv[1]
        result = handle_input_file(input_file_path)
        if result != -1:
            size = result[0]
            given_numbers = result[1]
            greater_than_signs_positions = result[2]

            # creating a board until a valid one is created
            board = initialize_board(size, given_numbers)
            while np.array_equal(board, [-1]):
                board = initialize_board(size, given_numbers)

            # if the given numbers are invalid
            if not np.array_equal(board, []):
                print(greater_than_signs_positions)
                print("final board:")
                for row in board:
                    print(row)
