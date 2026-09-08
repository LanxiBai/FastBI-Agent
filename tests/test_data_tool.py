"""Tests for the Excel/CSV data analysis tool."""

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pandas as pd

from tools.data_tool import (
    DataToolError,
    analyze_data_file,
    analyze_dataframe,
    load_data,
)


class DataToolTests(TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame(
            {
                "date": ["2026-01-01", "2026-01-02", "2026-01-02"],
                "region": ["East", "West", "West"],
                "revenue": [100.0, None, 200.0],
            }
        )

    def test_reads_and_analyzes_csv_with_missing_values(self) -> None:
        csv_file = BytesIO(self.dataframe.to_csv(index=False).encode("utf-8"))

        dataframe, summary = analyze_data_file(csv_file, "sales.csv")

        self.assertEqual(dataframe.shape, (3, 3))
        self.assertEqual(summary["shape"], {"rows": 3, "columns": 3})
        self.assertEqual(summary["missing"]["revenue"]["count"], 1)
        self.assertAlmostEqual(summary["missing"]["revenue"]["rate"], 1 / 3)
        self.assertEqual(summary["numeric_summary"]["revenue"]["median"], 150.0)
        self.assertEqual(summary["categorical_summary"]["region"]["top"], "West")
        self.assertIn("date", summary["date_columns"])

    def test_reads_and_analyzes_xlsx(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sales.xlsx"
            self.dataframe.to_excel(path, index=False)

            dataframe, summary = analyze_data_file(path)

        self.assertEqual(dataframe.shape, (3, 3))
        self.assertEqual(summary["numeric_summary"]["revenue"]["count"], 2)

    def test_reads_gb18030_csv(self) -> None:
        content = "城市,销售额\n上海,100\n北京,200\n".encode("gb18030")

        dataframe = load_data(BytesIO(content), "sales.csv")

        self.assertEqual(dataframe["城市"].tolist(), ["上海", "北京"])

    def test_counts_duplicate_rows(self) -> None:
        dataframe = pd.DataFrame(
            {"region": ["East", "East"], "revenue": [100, 100]}
        )

        summary = analyze_dataframe(dataframe)

        self.assertEqual(summary["duplicates"], 1)

    def test_rejects_unsupported_file_type(self) -> None:
        with self.assertRaisesRegex(DataToolError, "Unsupported file type"):
            load_data(BytesIO(b"some text"), "notes.txt")

    def test_rejects_empty_file(self) -> None:
        with self.assertRaisesRegex(DataToolError, "file is empty"):
            load_data(BytesIO(), "empty.csv")

    def test_rejects_header_only_dataset(self) -> None:
        with self.assertRaisesRegex(DataToolError, "contains no data rows"):
            load_data(BytesIO(b"name,value\n"), "header-only.csv")

    def test_reports_invalid_excel_file(self) -> None:
        with self.assertRaisesRegex(DataToolError, "Excel file could not be read"):
            load_data(BytesIO(b"not an Excel workbook"), "broken.xlsx")
