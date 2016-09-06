import os
from baxcat.geweke import Geweke

import pytest

resdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')


@pytest.mark.inference
@pytest.mark.parametrize('seed', [1337])
def test_geweke(seed):
    dtypes = ['categorical']*5 + ['continuous']*5
    gwk = Geweke(10, 10, dtypes, seed, ct_kernel=0, m=1)
    gwk.run(5000, 5, 1)
    assert not gwk.output(resdir)
