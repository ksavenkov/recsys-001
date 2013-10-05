import unittest
from printer import coursera_pa1_printer, coursera_pa2_printer

class PrinterTest(unittest.TestCase):
    
    def test_pa1(self):
        expected = '0,0,9.00,1,8.00,2,7.00,3,6.00,4,5.00,5,4.00,6,3.00,7,2.00,8,1.00,9,0.00\n50,0,9.00,1,8.00,2,7.00,3,6.00,4,5.00,5,4.00,6,3.00,7,2.00,8,1.00,9,0.00\n99,0,9.00,1,8.00,2,7.00,3,6.00,4,5.00,5,4.00,6,3.00,7,2.00,8,1.00,9,0.00'
        recs = [zip(range(10), range(10)[::-1]),]*3
        idxs = [0,50,99]
        self.assertTrue(coursera_pa1_printer(idxs, recs) == expected)

    def test_pa2(self):
        expected = 'recommendations for user 0:\n  0: 9.0000\n  1: 8.0000\n  2: 7.0000\n  3: 6.0000\n  4: 5.0000\n  5: 4.0000\n  6: 3.0000\n  7: 2.0000\n  8: 1.0000\n  9: 0.0000\nrecommendations for user 50:\n  0: 9.0000\n  1: 8.0000\n  2: 7.0000\n  3: 6.0000\n  4: 5.0000\n  5: 4.0000\n  6: 3.0000\n  7: 2.0000\n  8: 1.0000\n  9: 0.0000\nrecommendations for user 99:\n  0: 9.0000\n  1: 8.0000\n  2: 7.0000\n  3: 6.0000\n  4: 5.0000\n  5: 4.0000\n  6: 3.0000\n  7: 2.0000\n  8: 1.0000\n  9: 0.0000'
        recs = [zip(range(10), range(10)[::-1]),]*3
        idxs = [0,50,99]
        self.assertTrue(coursera_pa2_printer(idxs, recs) == expected)


if __name__ == '__main__':
    unittest.main()
