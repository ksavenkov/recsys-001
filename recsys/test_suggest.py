import unittest
import numpy as np
from suggest import top_n, top_ns

class SuggestTest(unittest.TestCase):

    def test(self):
        n = 5
        input = np.matrix([10,1,9,2,8,3,7,4,6,5])
        output = np.array([(0,10),(2,9),(4,8),(6,7),(8,6)])
        self.assertTrue(np.array_equal(top_n(input, n), output))
        self.assertTrue(np.array_equal(top_ns([input,]*3, n), [output,]*3))

if __name__ == '__main__':
    unittest.main()
