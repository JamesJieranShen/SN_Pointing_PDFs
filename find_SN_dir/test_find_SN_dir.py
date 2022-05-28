from unittest import TestCase
from find_SN_dir import *
class Tester(TestCase):
    def test_get_expected_counts_simple(self):
        a = get_expected_counts([10, 20], np.diag([1, 1]))
        assert (a == np.diag([10, 20])).all()

    def test_get_expected_counts_inefficient(self):
        a = get_expected_counts([2, 2], np.diag([0.5, 0.5]))
        assert (a == np.diag([1, 1])).all()

    def test_get_expected_counts_select(self):
        a = get_expected_counts([10, 20], np.diag([1, 1]), [0])
        assert (a == np.array([10, 0])).all()

    def test_get_expected_counts_mixture_1(self):
        confusion_matrix = np.array([[0.5, 0.5], [0.5, 0.5]])
        a = get_expected_counts([10, 20], confusion_matrix)
        assert (a == np.array([[5, 10], [5, 10]])).all()

    def test_get_expected_counts_mixture_2(self):
        confusion_matrix = np.array([[0.5, 1], [1, 0.5]])
        a = get_expected_counts([10, 20], confusion_matrix)
        assert (a == np.array([[5, 20], [10, 10]])).all()
