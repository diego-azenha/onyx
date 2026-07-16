import pytest

from sbrt.utils.ring_buffer import RingBuffer


def test_push_returns_none_until_full():
    rb = RingBuffer(3)
    assert rb.push(1.0) is None
    assert rb.push(2.0) is None
    assert rb.push(3.0) is None
    assert rb.push(4.0) == 1.0  # evicts oldest


def test_peek_ages_after_pushes():
    rb = RingBuffer(4)
    for x in (10, 20, 30, 40):
        rb.push(x)
    assert [rb.peek(i) for i in range(4)] == [40, 30, 20, 10]


def test_peek_out_of_range_raises():
    rb = RingBuffer(2)
    rb.push(1.0)
    with pytest.raises(IndexError):
        rb.peek(2)


def test_len_tracks_fill_level():
    rb = RingBuffer(3)
    assert len(rb) == 0
    rb.push(1.0)
    assert len(rb) == 1
    rb.push(2.0)
    rb.push(3.0)
    rb.push(4.0)
    assert len(rb) == 3
