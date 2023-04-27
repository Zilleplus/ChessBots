import unittest
import chess
import torch
import torch.nn as nn
import numpy as np

from mmEngine.models import SmallCNN
from mmEngine.value_funtions.nn_pytorch import encode_board, convert

class SmallCnnTest(unittest.TestCase):
    def test_smallCnn(self):
        board = chess.Board()
        x = convert(board)
        x = torch.from_numpy(np.array(x))
        x = torch.reshape(x, (1, 8, 8))
        x = encode_board(x).float()

        model = SmallCNN()
        model.eval()
        out = model(x)
        print(str(out))

        self.assertGreater(abs(torch.sum(out)), 1e-5)
        