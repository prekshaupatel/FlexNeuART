# from utils import *
# import unittest
# from transforms import *
from random_word_transformations import RandomDeletion, RandomInsertion

class TestRandomWordTransformations():
    def test_random_insertions(self):
        # answer = True
        # self.assertEqual(answer, True)
        sentence = "This is a test sentence"
        random_inserted_sentence = RandomInsertion(alpha_ri = 0.2)
        print(random_inserted_sentence.augment(sentence))
    def test_random_deletions(self):
        sentence = "This is a test sentence"
        random_deletion = RandomDeletion(p = 0.1)
        print(random_deletion.augment(sentence))
        pass

    def test_random_deletions(self):
        text = "Testing random deletions in this sentence. Do you think words are correctly deleted"
        random_deletion = RandomDeletion(p = 0.5)
        text_words_deleted = random_deletion.augment(text)
        assert(len(text_words_deleted) < len(text))

if __name__ == "__main__":
    tester = TestRandomWordTransformations()
    tester.test_random_deletions()
