"""Reading-order reconstruction for OCR boxes (pure geometry, no models)."""
from doclamar.ocr import OcrBox, layout_lines


def box(x0, y0, x1, text, h=20):
    return OcrBox(x0, y0, x1, y0 + h, text)


def column(x0, x1, y_start, label, n, step=30):
    return [box(x0, y_start + i * step, x1, f"{label}{i}") for i in range(n)]


def text_lines(boxes):
    return [line for line in layout_lines(boxes) if line]


def test_single_column_reads_top_to_bottom():
    boxes = column(100, 900, 100, "L", 8)
    assert text_lines(reversed(boxes)) == [f"L{i}" for i in range(8)]


def test_two_columns_are_not_interleaved():
    # Same baselines in both columns: sorting by y alone would give L0 R0 L1 R1 ...
    boxes = column(100, 480, 200, "L", 10) + column(520, 900, 200, "R", 10)
    assert text_lines(boxes) == [f"L{i}" for i in range(10)] + [f"R{i}" for i in range(10)]


def test_full_width_title_and_figure_split_the_columns_into_bands():
    title = box(100, 100, 900, "TITLE")
    figure = box(100, 520, 900, "Figure 1: full width caption")
    top = column(100, 480, 200, "a", 8) + column(520, 900, 200, "b", 8)       # above the figure
    bottom = column(100, 480, 600, "c", 6) + column(520, 900, 600, "d", 6)    # below it
    lines = text_lines([*bottom, figure, *top, title])
    expected = (["TITLE"] + [f"a{i}" for i in range(8)] + [f"b{i}" for i in range(8)]
                + ["Figure 1: full width caption"] + [f"c{i}" for i in range(6)] + [f"d{i}" for i in range(6)])
    assert lines == expected


def test_three_columns():
    boxes = column(50, 300, 100, "A", 6) + column(350, 600, 100, "B", 6) + column(650, 900, 100, "C", 6)
    assert text_lines(boxes) == [f"{c}{i}" for c in "ABC" for i in range(6)]


def test_boxes_on_one_baseline_merge_into_a_line_in_x_order():
    boxes = [box(400, 101, 600, "world"), box(100, 99, 380, "hello"), box(100, 140, 600, "next line")]
    assert text_lines(boxes) == ["hello world", "next line"]


def test_single_column_with_a_short_right_aligned_page_number_stays_single_column():
    boxes = column(100, 900, 100, "L", 8) + [box(850, 400, 880, "7")]
    lines = text_lines(boxes)
    assert lines[:8] == [f"L{i}" for i in range(8)] and "7" in lines


def test_empty_and_whitespace_boxes():
    assert layout_lines([]) == []
    assert layout_lines([box(0, 0, 10, "  ")]) == []
