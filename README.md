# wealthyunderstanding
A lackluster attempt at getting into AI image recognition of Chess
[Bonus points for whoever finds out the origin of the name]

The program reads the /images/bpr.jpg file, passes it to a pre-trained pytorch model that detects corners of the chessboard.
After that is done, we do transformations to stretch the image based on its corners, getting a uniform picture akin to a top-down view.
Dividing the board to an 8x8 grid with 9 lines we get 64 sectors based on the squares of the board.

We pass the original stretched picture to a different Pytorch model, that is trained to identify the pieces.

Then we look for the intersections of the 64 sectors and the identified pieces, which helps us place them on a matrix.

Lastly, we do a quick check for orientation, and then transform that matrix to a Forsynth-Edwards-Notation.
We generate a Lichess link with that FEN so that the game is openable in a browser or Lichess app.
