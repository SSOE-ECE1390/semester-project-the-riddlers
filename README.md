[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/tdy6BFPL)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=15757940&assignment_repo_type=AssignmentRepo)
# SemesterProject
This is the template repository that will be used to hold the semester project


## Description

The real-time sudoku solver is a versatile system designed to interpret and solve sudoku puzzles directly from live video feed. Its capabilities include identifying a paper when it is brought into the frame, extracting an ROI of the paper, identifying the gridlines that make up the sudoku puzzle, extracting each individual box, reading individual numbers from each box, solving the Sudoku puzzles in real time, and displaying the solution to the sudoku puzzle back on the paper in the video frame. This dynamic tool brings quick and accurate puzzle-solving to a variety of visual formats.

## Code Specifications

* General Inputs: Video Rate will be 30 FPS+, all puzzles will be on a sheet of paper with a white background. They should be held mostly still by the person holding the puzzle on the paper.  
* Sudoku:  
  * Paper Detection Inputs: Video frame with a solvable sudoku puzzle on a piece of paper that fills up roughly 40% of the frame, while still being completely visible.   
  * Paper Detection Outputs: ROI of the piece of paper; boolean indicating whether a paper was detected
  * Grid Inputs: ROI image of paper
  * Grid Outputs: Individual box ROI images comprising the sudoku puzzle
  * Reading Numbers Inputs: Each box ROI image containing a number or no number, center location for each box
  * Reading Numbers Outputs: The numbers, or lackthereof, detected in the order of top to bottom, left to right, as an array
  * Sudoku Solver Inputs: The numbers array read in from the paper
  * Sudoku Solver Output: The numerical solutions to the puzzle in an array ordered top to bottom, left to right
  * Displaying Solution Inputs: Center location for each box on the puzzle, array with numerical solutions
  * Displaying Solution Outputs: Paper image in the video frame with the numberical solution reflected in each box 

## Planned approach

This project has multiple elements, first is recognizing and processing the image. We will implement this with the OpenCV library for python. In the OpenCV library we plan to use a variety of algorithms to help isolate features, so that we can more easily recognize the puzzle, recognize the key features for solving the puzzle, and finally recognize where the answers need to go on the video feed to solve the puzzle. This would include background subtractor, edge detection, thresholding, or Aruco markers for isolating the paper. ORB for extracting the letter features, as well as OCR for recognizing and processing the characters to be used in the puzzle. Once we have found a variety of algorithms for solving the sudoku puzzle (see below), on github, we will have to take the features that we have processed off of the puzzles and format them such that they can be utilized by the algorithms that we have found off of github. We may also have to modify the algorithms so that they output correctly onto the puzzle in the video feed, again using feature detection to find where these answers go on the puzzle. Finally, there may need to be some optimization to try and speed up the processing time.

## Time-line

1. Recognition of Paper
2. Extraction of individual boxes using the grid of the puzzle
3. Recognition of Letters and Numbers  
4. Sudoku Solving Algorithm  
6. Taking a picture when the paper is correctly in frame  
7. Solving any sudoku puzzle on a still image   
8. Solving any sudoku puzzle on a video
9. Tracking the puzzle and where each of the solutions will be displayed on the puzzle
10. Implement optimization methods to reduce time taken to solve the puzzle

## Metrics of Success

The final goal for this project is to be able to solve all selected sudoku puzzles and display the solutions in real time. As we want to focus on the image processing for this project, the actual puzzle algorithms will be less of a priority, and getting a solution, whether correct or not, to display back on the paper in the video frame.

## Pitfalls and alternative solutions

One pitfall we faced in implementing this project was detecting the paper. After implementing multiple different methods such as thresholding, edge detection, Hough lines, corner detection, and extracting different types of contours based on size, location, and color, we decided to implement the use of Aruco markers. Aruco markers allowed us to easily identify, orient, and extract the paper in the video feed.

Another pitfall we faced was the length of time and accuracy for reading in each of the numbers from the puzzle. We implemented a few different number recognition algorithms including easyOCR and pytesseract and used multiple iterations of reading in the numbers to compared confidences to get the most accurate reading of each number. We also tried writing out our own numbers for the sudoku puzzles to see if this would improve number recognition. We implemented parallel processing to try and reduce length of time, but there remains a lag in taking in the paper and displaying the results.

Finally, we were able to implement our code on a still image, but translating to live video posed some challenges. 

* [OpenCV](https://opencv.org/)  
* [pytesseract](https://github.com/h/pytesseract)  
* [https://github.com/dhhruv/Sudoku-Solver](https://github.com/dhhruv/Sudoku-Solver)  
* [https://github.com/seancfong/Word-Search-Solver](https://github.com/seancfong/Word-Search-Solver)  
* [https://github.com/SSOE-ECE1390](https://github.com/SSOE-ECE1390)  
* [https://medium.com/@vinod.batra0311/solve-the-spot-the-differences-puzzle-with-computer-vision-2cb258fd2fc7](https://medium.com/@vinod.batra0311/solve-the-spot-the-differences-puzzle-with-computer-vision-2cb258fd2fc7) 

