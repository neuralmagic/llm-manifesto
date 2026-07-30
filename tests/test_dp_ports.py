"""Tests for implicit port derivation with and without local DP fanout."""

from manifesto.dp_ports import derive_ports


def test_no_dp_has_one_port():
    ports = derive_ports(rank_count=1)

    assert ports.public == [8000]
    assert ports.backend == [8000]
    assert ports.rank_count == 1


def test_dp_has_one_port_per_local_rank():
    ports = derive_ports(rank_count=4, public_base=8000, backend_base=8200)

    assert ports.public == [8000, 8001, 8002, 8003]
    assert ports.backend == [8200, 8201, 8202, 8203]
    assert ports.rank_count == 4
