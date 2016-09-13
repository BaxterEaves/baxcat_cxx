import os
from baxcat.geweke import Geweke

import pytest

DIR = os.path.dirname(os.path.abspath(__file__))
RESDIR = os.path.join(DIR, 'result', 'geweke')

if not os.path.exists(RESDIR):
    os.makedirs(RESDIR)


@pytest.mark.inference
@pytest.mark.parametrize('seed', [1337])
def test_geweke(seed):
    dtypes = ['categorical']*5 + ['continuous']*5
    gwk = Geweke(10, 10, dtypes, seed, ct_kernel=0, m=1)
    gwk.run(5000, 5, 1)
    assert not gwk.output(RESDIR)
