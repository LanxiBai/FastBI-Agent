"""Numerical and validation tests for constrained pandas requests."""

from unittest import TestCase

import pandas as pd

from tools.pandas_analysis_tool import PandasAnalysisError, execute_analysis


class PandasAnalysisTests(TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame({
            "Product": ["A", "B", "A", "B", "C"],
            "Quantity": [2.0, 8.0, 4.0, 6.0, None],
            "Region": ["East", "East", "West", "West", "East"],
        })
        self.request = {
            "group_by": "Product", "metric": "Quantity",
            "aggregation": "sum", "sort": "desc",
        }

    def test_all_five_grouped_aggregations(self) -> None:
        for operation, expected in {
            "sum": {"A": 6.0, "B": 14.0, "C": None},
            "mean": {"A": 3.0, "B": 7.0, "C": None},
            "count": {"A": 2, "B": 2, "C": 0},
            "min": {"A": 2.0, "B": 6.0, "C": None},
            "max": {"A": 4.0, "B": 8.0, "C": None},
        }.items():
            with self.subTest(operation=operation):
                result = execute_analysis(
                    self.dataframe, {**self.request, "aggregation": operation}
                )
                actual = {row["Product"]: row["Quantity"] for row in result["records"]}
                self.assertEqual(actual, expected)

    def test_sort_and_filters_apply_before_aggregation_without_mutation(self) -> None:
        original = self.dataframe.copy(deep=True)
        result = execute_analysis(self.dataframe, {
            **self.request, "sort": "asc", "filters": [
                {"column": "Region", "op": "eq", "value": "East"},
                {"column": "Quantity", "op": "gte", "value": 2},
            ],
        })
        self.assertEqual(result["records"], [
            {"Product": "A", "Quantity": 2.0},
            {"Product": "B", "Quantity": 8.0},
        ])
        pd.testing.assert_frame_equal(original, self.dataframe)

    def test_all_filter_operators(self) -> None:
        for op, count in {"eq": 1, "ne": 3, "gt": 2, "gte": 3, "lt": 1, "lte": 2}.items():
            with self.subTest(op=op):
                result = execute_analysis(self.dataframe, {
                    **self.request,
                    "filters": [{"column": "Quantity", "op": op, "value": 4}],
                })
                self.assertEqual(result["matched_rows"], count)

    def test_global_count_and_sum(self) -> None:
        for operation, expected in [("count", 4), ("sum", 20.0)]:
            result = execute_analysis(self.dataframe, {
                **self.request, "group_by": None, "aggregation": operation,
            })
            self.assertEqual(result["records"], [{"Quantity": expected}])

    def test_empty_filter_result_and_empty_input(self) -> None:
        result = execute_analysis(self.dataframe, {
            **self.request,
            "filters": [{"column": "Quantity", "op": "gt", "value": 100}],
        })
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["records"], [])
        with self.assertRaises(PandasAnalysisError):
            execute_analysis(pd.DataFrame(), self.request)

    def test_invalid_requests_are_rejected(self) -> None:
        changes = [
            {"metric": "Missing"}, {"metric": "Region"},
            {"aggregation": "__import__('os')"}, {"sort": "random"},
            {"code": "print('not allowed')"},
            {"filters": [{"column": "Quantity", "op": "eval", "value": "1"}]},
            {"filters": [{"column": "Quantity", "op": "gt", "value": "bad"}]},
        ]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(PandasAnalysisError):
                execute_analysis(self.dataframe, {**self.request, **change})

    def test_large_results_are_sorted_before_truncation(self) -> None:
        dataframe = pd.DataFrame({"Product": range(120), "Quantity": range(120)})
        result = execute_analysis(dataframe, self.request)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["total_groups"], 120)
        self.assertEqual(len(result["records"]), 100)
        self.assertEqual(result["records"][0]["Quantity"], 119)
