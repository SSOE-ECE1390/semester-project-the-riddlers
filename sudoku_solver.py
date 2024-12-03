from pprint import pprint
from random import shuffle, randint

def is_valid(puzzle, guess, row, col):
    """
    Checks if a guess at puzzle[row][col] is a valid number according to Sudoku rules.

    Args:
        puzzle (list of list of int): The current state of the puzzle.
        guess (int): The number being guessed (1-9).
        row (int): The row index of the guess.
        col (int): The column index of the guess.

    Returns:
        bool: True if the guess is valid, False otherwise.
    """

    # Check row
    if guess in puzzle[row]:
        return False

    # Check column
    col_vals = [puzzle[i][col] for i in range(9)]
    if guess in col_vals:
        return False

    # Check 3x3 square
    row_start = (row // 3) * 3
    col_start = (col // 3) * 3
    for r in range(row_start, row_start + 3):
        for c in range(col_start, col_start + 3):
            if puzzle[r][c] == guess:
                return False

    return True

def solve_sudoku(puzzle):
    """
    Solves a Sudoku puzzle using backtracking algorithm.

    Args:
        puzzle (list of list of int): A 9x9 2D list representing the Sudoku board.

    Returns:
        bool: True if the puzzle is solved, False if unsolvable.
    """

    def find_next_empty(puzzle):
        """
        Finds the next row, col on the puzzle that's not filled yet (represented by -1).
        Returns a tuple (row, col) or (None, None) if there is no empty space.
        """
        for r in range(9):
            for c in range(9):
                if puzzle[r][c] == -1:
                    return r, c
        return None, None  # No empty space

    # Main logic for backtracking solver
    row, col = find_next_empty(puzzle)
    if row is None:  # Puzzle solved
        return True

    # Try guesses from 1 to 9
    for guess in range(1, 10):
        if is_valid(puzzle, guess, row, col):
            puzzle[row][col] = guess  # Place guess on the board
            if solve_sudoku(puzzle):  # Recurse to solve the rest of the puzzle
                return True

        # Backtrack if the guess was not correct
        puzzle[row][col] = -1

    # If no valid guesses found, the puzzle is unsolvable
    return False


def flatten_board(board):
    """
    Converts a 2D list (board) to a single 1D list.

    Args:
        board (list of list of int): A 9x9 2D list representing the Sudoku board.

    Returns:
        list of int: A single list with all the elements of the 2D list in row-major order.
    """
    return [cell for row in board for cell in row]


def generate_board():
    """
    Generates a random sudoku board with fewer initial numbers.

    Returns:
        list[list[int]]: A 9x9 sudoku board represented as a list of lists of integers.
    """
    board = [[0 for i in range(9)] for j in range(9)]

    # Fill the diagonal boxes
    for i in range(0, 9, 3):
        nums = list(range(1, 10))
        shuffle(nums)
        for row in range(3):
            for col in range(3):
                board[i + row][i + col] = nums.pop()

    # Fill the remaining cells with backtracking
    def fill_cells(board, row, col):
        """
        Fills the remaining cells of the sudoku board with backtracking.

        Args:
            board (list[list[int]]): A 9x9 sudoku board represented as a list of lists of integers.
            row (int): The current row index to fill.
            col (int): The current column index to fill.

        Returns:
            bool: True if the remaining cells are successfully filled, False otherwise.
        """

        if row == 9:
            return True
        if col == 9:
            return fill_cells(board, row + 1, 0)

        if board[row][col] != 0:
            return fill_cells(board, row, col + 1)

        for num in range(1, 10):
            if is_valid(board, num, row, col):
                board[row][col] = num

                if fill_cells(board, row, col + 1):
                    return True

        board[row][col] = 0
        return False

    fill_cells(board, 0, 0)

    # Remove a greater number of cells to create a puzzle with fewer initial numbers
    for _ in range(randint(55, 65)):
        row, col = randint(0, 8), randint(0, 8)
        board[row][col] = 0

    return board


def print_board(board):
    """
    Prints the Sudoku board in a human-readable format.

    Args:
        board (list of list of int): A 9x9 2D list representing the Sudoku board.
    """
    for row in board:
        print(" ".join(str(cell) if cell != 0 else '.' for cell in row))


if __name__ == "__main__":
    board = generate_board()
    print("Generated Sudoku Board:")
    print_board(board)
    if solve_sudoku(board):
        print("\nSolved Sudoku Board:")
        print_board(board)
    else:
        print("\nPuzzle could not be solved.")
