import pytest

from dima_auto_editor.naming import natural_key, natural_sorted, parse_numbered_stem


def test_natural_sorting_orders_numbers_by_value():
    assert natural_sorted(["10.mp4", "2.mp4", "01.mp4"]) == ["01.mp4", "2.mp4", "10.mp4"]


def test_natural_sorting_places_sub_indices_after_their_scene():
    names = ["04.png", "03_2.png", "10.mp4", "03.mp4", "03_10.png", "3_1.png"]
    assert natural_sorted(names) == ["03.mp4", "3_1.png", "03_2.png", "03_10.png", "04.png", "10.mp4"]


def test_natural_key_is_case_insensitive():
    assert natural_key("Scene2.PNG") < natural_key("scene10.png")


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("01", (1, 0)),
        ("2", (2, 0)),
        ("10", (10, 0)),
        ("03_2", (3, 2)),
        ("03-2", (3, 2)),
        ("03a", (3, 1)),
        ("03b", (3, 2)),
        ("01 intro", (1, 0)),
        ("01_take2", (1, 0)),
    ],
)
def test_parse_numbered_stem(stem, expected):
    assert parse_numbered_stem(stem) == expected


@pytest.mark.parametrize("stem", ["intro", "", "final_cut"])
def test_parse_numbered_stem_rejects_unnumbered(stem):
    assert parse_numbered_stem(stem) is None
