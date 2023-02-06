import unittest

import map_tags


class TestMapTagsMethods(unittest.TestCase):
    def test_main(self):
        self.assertTrue(map_tags.main())


if __name__ == '__main__':
    unittest.main()
