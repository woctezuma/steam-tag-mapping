import unittest

import download_json
import map_tags


class TestDownloadJsonMethods(unittest.TestCase):

    def test_main(self):
        self.assertTrue(download_json.main())


class TestMapTagsMethods(unittest.TestCase):

    def test_main(self):
        self.assertTrue(map_tags.main())


if __name__ == '__main__':
    unittest.main()
