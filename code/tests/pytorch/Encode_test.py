import unittest
import chess
import numpy as np
import torch
from mmEngine.database.load_database import convert
from mmEngine.value_funtions import encode_board
from mmEngine.database import encoding_channels


class EncodeBoardTest(unittest.TestCase):
    def test_encode_white_pawns(self):
        """
        Test if the encode_board function works correctly:
            Create a board with a white pawn on the second row and check if the
            output is correct.
        """
        input = torch.zeros((1, 8, 8))
        second_row = 1
        # set all the white pawn on the first row
        for i in range(8):
            input[0, second_row, i] = 1
            input[0, second_row, i] = 1

        output = encode_board(input)
        # check that the output has the correct shape
        self.assertEqual(output.shape, (1, 12, 8, 8))

        # check if the pawn positions are 1 in the correct channel
        pawn_channel = 0
        for i in range(8):
            for j in range(8):
                if i == second_row:
                    self.assertEqual(output[0, pawn_channel, i, j], 1)
                else:
                    self.assertEqual(output[0, pawn_channel, i, j], 0)

    def test_code_start_board(self):
        board = chess.Board()
        x = convert(board)
        x = torch.from_numpy(np.array(x))
        x = torch.reshape(x, (1, 8, 8))
        x = encode_board(x)

        white_pawn_channel = encoding_channels["white pawn"]
        white_knight_channel = encoding_channels["white knight"]
        white_bishop_channel = encoding_channels["white bishop"]
        white_rook_channel = encoding_channels["white rook"]
        white_queen_channel = encoding_channels["white queen"]
        white_king_channel = encoding_channels["white king"]
        for i in range(8):
            for j in range(8):
                # white pawns start on the second row
                if i == 1:
                    self.assertEqual(x[0, white_pawn_channel, i, j], 1)
                else:
                    self.assertEqual(x[0, white_pawn_channel, i, j], 0)
                # white knights start on the first row
                if i == 0 and (j == 1 or j == 6):
                    self.assertEqual(x[0, white_knight_channel, i, j], 1)
                else:
                    self.assertEqual(x[0, white_knight_channel, i, j], 0)
                # white bishops start on the first row
                if i == 0 and (j == 2 or j == 5):
                    self.assertEqual(x[0, white_bishop_channel, i, j], 1)
                else:
                    self.assertEqual(x[0, white_bishop_channel, i, j], 0)
                # white rooks start on the first row
                if i == 0 and (j == 0 or j == 7):
                    self.assertEqual(x[0, white_rook_channel, i, j], 1)
                else:
                    self.assertEqual(x[0, white_rook_channel, i, j], 0)
                # white queen starts on the first row
                if i == 0 and j == 3:
                    self.assertEqual(x[0, white_queen_channel, i, j], 1)
                else:
                    self.assertEqual(x[0, white_queen_channel, i, j], 0)
                # white king starts on the first row
                if i == 0 and j == 4:
                    self.assertEqual(x[0, white_king_channel, i, j], 1)
                else:
                    self.assertEqual(x[0, white_king_channel, i, j], 0)

                # black pawns start on the seventh row
                if i == 6:
                    self.assertEqual(x[0, white_pawn_channel + 6, i, j], 1)
                else:
                    self.assertEqual(x[0, white_pawn_channel + 6, i, j], 0)
                # black knights start on the eighth row
                if i == 7 and (j == 1 or j == 6):
                    self.assertEqual(x[0, white_knight_channel + 6, i, j], 1)
                else:
                    self.assertEqual(x[0, white_knight_channel + 6, i, j], 0)
                # black bishops start on the eighth row
                if i == 7 and (j == 2 or j == 5):
                    self.assertEqual(x[0, white_bishop_channel + 6, i, j], 1)
                else:
                    self.assertEqual(x[0, white_bishop_channel + 6, i, j], 0)
                # black rooks start on the eighth row
                if i == 7 and (j == 0 or j == 7):
                    self.assertEqual(x[0, white_rook_channel + 6, i, j], 1)
                else:
                    self.assertEqual(x[0, white_rook_channel + 6, i, j], 0)
                # black queen starts on the eighth row
                if i == 7 and j == 3:
                    self.assertEqual(x[0, white_queen_channel + 6, i, j], 1)
                else:
                    self.assertEqual(x[0, white_queen_channel + 6, i, j], 0)
                # black king starts on the eighth row
                if i == 7 and j == 4:
                    self.assertEqual(x[0, white_king_channel + 6, i, j], 1)
                else:
                    self.assertEqual(x[0, white_king_channel + 6, i, j], 0)

if __name__ == "__main__":
    unittest.main()
