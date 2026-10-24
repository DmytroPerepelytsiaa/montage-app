from __future__ import annotations

from typing import Any, Iterable, Sequence

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem, QWidget

from ..manifest import timeline_summary_rows
from ..models import Issue, Severity, Timeline
from .theme import SEVERITY_COLORS

_SEVERITY_ORDER = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}


def issue_counts(issues: Iterable[Issue]) -> dict[Severity, int]:
    counts = {severity: 0 for severity in Severity}
    for issue in issues:
        counts[issue.severity] += 1
    return counts


class IssuesList(QListWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWordWrap(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)

    def set_issues(self, issues: Sequence[Issue]) -> None:
        self.clear()
        for issue in sorted(issues, key=lambda item: _SEVERITY_ORDER[item.severity]):
            item = QListWidgetItem(f"[{issue.severity.value.upper()}] {issue.message}")
            item.setForeground(QBrush(SEVERITY_COLORS[issue.severity]))
            item.setToolTip(f"{issue.code}\n{issue.path}".strip())
            item.setData(Qt.ItemDataRole.UserRole, issue)
            self.addItem(item)

    def issues(self) -> list[Issue]:
        return [self.item(row).data(Qt.ItemDataRole.UserRole) for row in range(self.count())]


TIMELINE_COLUMNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("#", ("#",)),
    ("Scene", ("scene",)),
    ("Start", ("start",)),
    ("End", ("end",)),
    ("Duration", ("duration",)),
    ("Cut / transition", ("transition", "cut")),
    ("SFX", ("sfx",)),
    ("Notes", ("notes",)),
)
TIMELINE_HEADERS = tuple(header for header, _ in TIMELINE_COLUMNS)


def summary_cells(row: dict[str, Any]) -> list[str]:
    return [next((str(row[key]) for key in keys if row.get(key) not in (None, "")), "") for _, keys in TIMELINE_COLUMNS]


class TimelineTable(QTableWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(0, len(TIMELINE_HEADERS), parent)
        self.setHorizontalHeaderLabels(TIMELINE_HEADERS)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)

    def show_timeline(self, timeline: Timeline | None) -> None:
        self.show_rows(timeline_summary_rows(timeline) if timeline is not None else [])

    def show_rows(self, rows: Sequence[dict[str, Any]]) -> None:
        self.setRowCount(0)
        self.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for column, text in enumerate(summary_cells(row)):
                item = QTableWidgetItem(text)
                item.setToolTip(text)
                self.setItem(row_index, column, item)
