from root_finding import *
import random


class Test_fixed_pt(unittest.TestCase):
    def test(self):
        self.assertAlmostEqual(fixed_point(phi,0.3,50)[0], 0.450183609841)


class Test_recursive_bisection(unittest.TestCase):
    def test(self):
        self.assertAlmostEqual(recursive_bisection(f_poly,-2,2)[0],-1.324717957526)

    def test1(self):
        self.assertAlmostEqual(f_poly(recursive_bisection(f_poly,-2,2)[0]), 0)


class Test_newton(unittest.TestCase):
    def test_0(self):
        self.assertAlmostEqual(newton_method(f_poly, dfdx_poly,-1,20)[0], recursive_bisection(f_poly,-2,2)[0])

class Test_newton_2d(unittest.TestCase):
    def test(self):
        x_0 = np.array((random.random(), random.random()))
        self.assertAlmostEqual(f_2d(newton_multidim(f_2d,dfdx_2d,x_0,40)[0])[0], 0)
        self.assertAlmostEqual(f_2d(newton_multidim(f_2d,dfdx_2d,x_0,40)[0])[1], 0)


if __name__ == '__main__':
    unittest.main()