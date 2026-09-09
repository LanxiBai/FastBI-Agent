"""Tests for the Plotly chart generation tool."""

from unittest import TestCase

import pandas as pd
import plotly.graph_objects as go

from tools.chart_tool import ChartToolError, generate_chart, suggest_chart


class ChartToolTests(TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame(
            {
                "date": ["2026-01-02", "2026-01-01", "2026-01-03"],
                "product": ["Basic", "Pro", "Enterprise"],
                "sales": [120.0, 200.0, 160.0],
            }
        )

    def test_generates_date_sales_line_chart(self) -> None:
        figure = generate_chart(self.dataframe, "line", "date", "sales")

        self.assertIsInstance(figure, go.Figure)
        self.assertEqual(figure.data[0].type, "scatter")
        self.assertEqual(list(figure.data[0].y), [200.0, 120.0, 160.0])

    def test_generates_product_sales_bar_chart(self) -> None:
        figure = generate_chart(self.dataframe, "bar", "product", "sales")

        self.assertIsInstance(figure, go.Figure)
        self.assertEqual(figure.data[0].type, "bar")
        self.assertEqual(list(figure.data[0].x), ["Basic", "Pro", "Enterprise"])

    def test_generates_sales_histogram(self) -> None:
        figure = generate_chart(self.dataframe, "histogram", "sales")

        self.assertIsInstance(figure, go.Figure)
        self.assertEqual(figure.data[0].type, "histogram")
        self.assertEqual(list(figure.data[0].x), [120.0, 200.0, 160.0])

    def test_recommends_line_chart_for_date_and_numeric_fields(self) -> None:
        suggestion = suggest_chart(self.dataframe)

        self.assertEqual(suggestion["chart_type"], "line")
        self.assertEqual(suggestion["x_column"], "date")
        self.assertEqual(suggestion["y_column"], "sales")

    def test_recommends_bar_chart_for_category_and_numeric_fields(self) -> None:
        suggestion = suggest_chart(self.dataframe[["product", "sales"]])

        self.assertEqual(suggestion["chart_type"], "bar")
        self.assertEqual(suggestion["x_column"], "product")
        self.assertEqual(suggestion["y_column"], "sales")

    def test_recommends_histogram_for_numeric_only_data(self) -> None:
        suggestion = suggest_chart(self.dataframe[["sales"]])

        self.assertEqual(suggestion["chart_type"], "histogram")
        self.assertEqual(suggestion["x_column"], "sales")
        self.assertIsNone(suggestion["y_column"])

    def test_prefers_derived_date_from_year_month_day_columns(self) -> None:
        dataframe = pd.DataFrame(
            {
                "year": [2026, 2026],
                "month": [9, 8],
                "day": [9, 31],
                "existing_date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
                "sales": [200.0, 100.0],
            }
        )

        suggestion = suggest_chart(dataframe)
        figure = generate_chart(dataframe, "line", "date", "sales")

        self.assertEqual(suggestion["chart_type"], "line")
        self.assertEqual(suggestion["x_column"], "date")
        self.assertEqual(list(figure.data[0].y), [100.0, 200.0])

    def test_rejects_invalid_field_combinations(self) -> None:
        with self.assertRaisesRegex(ChartToolError, "must be date/time"):
            generate_chart(self.dataframe, "line", "product", "sales")
        with self.assertRaisesRegex(ChartToolError, "does not exist"):
            generate_chart(self.dataframe, "bar", "unknown", "sales")

    def test_rejects_empty_dataframe(self) -> None:
        with self.assertRaisesRegex(ChartToolError, "empty dataset"):
            generate_chart(pd.DataFrame(), "histogram", "sales")

    def test_rejects_data_without_numeric_fields(self) -> None:
        dataframe = pd.DataFrame({"product": ["Basic", "Pro"]})

        with self.assertRaisesRegex(ChartToolError, "numeric column is required"):
            suggest_chart(dataframe)
