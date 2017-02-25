from Interpolation import *
import unittest

class Test_lagrange_orthogonality(unittest.TestCase):
    def test(self):
        grid = grid_chebychev(-2,2,5)
        for i,x_i in enumerate(grid):
            phi = basis_lagrange([x_i],grid)
            self.assertAlmostEquals(phi[i],1)
            for j in range(len(grid)):
                if i != j:
                    self.assertAlmostEquals(phi[j],0)

class Test_basis_matrix_lagrange(unittest.TestCase):
    def test(self):
        grid = grid_chebychev(-2,2,5)
        coeff, A = find_intepolation_coeff('lagrange', grid,f)
        for i in range(len(grid)):
            for j in range(len(grid)):
                if i == j:
                    self.assertAlmostEqual(A[i,j],1)
                else:
                    self.assertAlmostEqual(A[i,j],0)



if __name__ == '__main__':
    unittest.main()