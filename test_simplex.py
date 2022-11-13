import numpy as np
import pytest

from simplex import simplex

# import simplex


def test_simplex_simplex():
    t = np.array([
        [0,6,4,3,0,0,0,0],
        [3,4,5,3,1,0,0,12],
        [4,3,4,2,0,1,0,10],
        [5,4,2,1,0,0,1,8]
    ], dtype=np.float128)

    basic, cost, c = simplex(t)
    z = list(zip(basic, cost))

    assert len(z) == 3
    assert (1,1.5) in z
    assert (3,2) in z
    assert (5,1.5) in z
    assert c == -15

def test_blands_rule():
    t = np.array([
        [0, 10, -57, -9, -24, 0, 0, 0, 0],
        [4, 0.5, -5.5, -2.5, 9, 1, 0, 0, 0],
        [5, 0.5, -1.5, -0.5, 1, 0, 1, 0, 0],
        [6, 1, 0, 0, 0, 0, 0, 1, 1,]
    ], dtype=np.float128)

    basic, cost, c = simplex(t, blandsRule=True)
    
    z = list(zip(basic, cost))
    assert len(basic) == 3
    assert c == -1
    assert (1,1) in z
    assert (3,1) in z
    assert (5,2) in z

def test_two_phase_first_step():
    t = np.array([
        [0, 1, 1, 0, 0, 0, -1,  0, 1],
        [2, 3, 4, 1, 0, 0,  0,  0, 12],
        [3, 3, 3, 0, 1, 0,  0,  0, 10],
        [4, 4, 2, 0, 0, 1,  0,  0, 8],
        [6, 1 ,1, 0, 0, 0, -1,  1, 1]], dtype=np.float128)
    
    basic, cost, c = simplex(t, blandsRule=True)
    
    assert c == 0
    z = list(zip(basic, cost))
    assert (3, 9) in z
    assert (4, 7) in z
    assert (5, 4) in z
    assert (1, 1) in z


if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)