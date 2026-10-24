"""Phase 1 tests for src/variables.py (plant-unit map of the 52 variables)."""
from src import data, variables

ALLOWED_UNITS = {"feed", "reactor", "condenser", "separator", "compressor", "stripper", "purge", "product"}


def test_fifty_two_entries_tagged_like_the_columns():
    assert len(variables.VARIABLE_INFO) == 52
    assert list(variables.VARIABLE_INFO) == data.VARIABLES
    assert all(v.tag == tag for tag, v in variables.VARIABLE_INFO.items())


def test_plant_units_come_from_the_allowed_list():
    assert {v.plant_unit for v in variables.VARIABLE_INFO.values()} <= ALLOWED_UNITS


def test_kind_follows_the_tag_ranges():
    for i in range(1, 42):
        expected = "measured" if i <= 22 else "analyzer"
        assert variables.VARIABLE_INFO[f"xmeas_{i}"].kind == expected
    assert all(variables.VARIABLE_INFO[f"xmv_{i}"].kind == "manipulated" for i in range(1, 12))


def test_spot_checks_against_the_flowsheet():
    info = variables.VARIABLE_INFO
    assert info["xmeas_7"].plant_unit == "reactor" and info["xmeas_7"].name == "Reactor pressure"
    assert info["xmeas_15"].plant_unit == "stripper"
    assert info["xmv_10"].plant_unit == "reactor" and info["xmv_10"].name == "Reactor cooling water flow"
    assert info["xmeas_1"].name == "A feed" and info["xmeas_1"].unit == "kscmh"
    assert info["xmeas_22"].plant_unit == "condenser"  # D&V name says "separator"; the cooling is on the condenser
    assert info["xmv_11"].plant_unit == "condenser"
    assert info["xmeas_37"].plant_unit == "product"    # product analysis, stream 11
    assert info["xmeas_29"].plant_unit == "purge"      # purge gas analysis, stream 9
    assert info["xmv_3"].name == "A feed flow"         # xmv order is D, E, A, A+C (xmeas is A, D, E, A+C)


def test_every_unit_has_at_least_one_variable():
    assert {v.plant_unit for v in variables.VARIABLE_INFO.values()} == ALLOWED_UNITS
