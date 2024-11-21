from pprint import pprint

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


if __name__ == '__main__':
    board = [
        [3, -1, 9, 5, -1, -1, -1, 1, 4],
        [2, -1, -1, -1, -1, 3, 7, 5, 9],
        [7, -1, 4, -1, 2, -1, -1, 8, 6],
        [-1, 9, -1, 3, -1, -1, -1, -1, -1],
        [4, -1, 8, -1, -1, -1, 5, -1, 1],
        [-1, -1, -1, -1, -1, 6, -1, 3, -1],
        [5, 6, -1, -1, 3, -1, 8, -1, 7],
        [9, -1, 2, 6, -1, -1, -1, -1, 3],
        [8, 4, -1, -1, -1, 9, 6, -1, 5]
    ]

    solved = solve_sudoku(board)
    print(f"Solved: {solved}")
    pprint(board)