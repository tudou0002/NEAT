import unittest
from extractors.entity import Entity

class TestEntity(unittest.TestCase):
    def setUp(self):
        self.entity_1 = Entity('Sam', 4)
        self.entity_2 = Entity('Sam', 4)
        self.entity_3 = Entity('Alex', 4)

    def test_equality(self):
        self.assertEqual(self.entity_1, self.entity_2)

    def test_inequality(self):
        self.assertNotEqual(self.entity_2, self.entity_3)

    def test_set_certainty(self):
        self.entity_1.score = 0.5

        self.assertEqual(self.entity_1.score, 0.5)


if __name__ == '__main__':
    unittest.main()