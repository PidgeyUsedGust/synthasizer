from synthasizer.utilities import nzs


def test_nzs():
    s = nzs()
    s.add(1)
    s.add(0)
    s.add(2)
    assert len(s) == 2
    assert 0 not in s
    s.update([0, 1, 2, 3])
    assert len(s) == 3
    assert 0 not in s


if __name__ == "__main__":
    test_nzs()