"""Tests for the integrated LangGraph analyst workflow."""

import json
from unittest import TestCase

import pandas as pd
from langchain_core.messages import AIMessage

from agents.analyst_agent import AgentError, run_agent


class FakeChatModel:
    """Return deterministic structured analysis without an external API call."""

    def __init__(self, summary: str = "Analysis completed from the supplied data."):
        self.summary = summary
        self.messages: list[object] = []

    def invoke(self, messages: object, **kwargs: object) -> AIMessage:
        self.messages = list(messages)
        if "ONE safe pandas analysis request" in self.messages[0].content:
            payload = json.loads(self.messages[1].content)
            question = payload["question"]
            columns = payload["dataset_summary"]["columns"]
            request = None
            if "趋势" in question or "下降" in question:
                request = {"group_by": "date", "metric": "sales", "aggregation": "sum"}
            elif "哪个产品" in question and "product" in columns:
                request = {
                    "group_by": "product", "metric": "sales",
                    "aggregation": "sum", "sort": "desc",
                }
            return AIMessage(content=json.dumps({"request": request, "reason": "test"}))
        return AIMessage(
            content=json.dumps(
                {
                    "summary": self.summary,
                    "key_findings": ["Finding supported by the supplied context."],
                    "potential_risks": ["Only the available fields were evaluated."],
                    "recommended_actions": ["Review the identified metric."],
                }
            )
        )


class InvalidChatModel:
    def invoke(self, messages: object, **kwargs: object) -> AIMessage:
        return AIMessage(content="not valid JSON")


class FailingChatModel:
    def invoke(self, messages: object, **kwargs: object) -> AIMessage:
        raise RuntimeError("service unavailable")


class AnalystAgentTests(TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame(
            {
                "year": [2026, 2026, 2026],
                "month": [1, 2, 3],
                "day": [1, 1, 1],
                "product": ["Basic", "Pro", "Pro"],
                "sales": [100.0, 250.0, 150.0],
            }
        )

    def test_summarizes_dataset_without_forcing_chart(self) -> None:
        result = run_agent("总结这份数据", self.dataframe, FakeChatModel())

        self.assertIn("## Summary", result["final_response"])
        self.assertEqual(result["dataset_summary"]["shape"]["rows"], 3)
        self.assertEqual(result["chart_result"]["status"], "not_requested")
        self.assertIsNone(result["chart_figure"])

    def test_generates_line_chart_for_sales_trend(self) -> None:
        result = run_agent(
            "销售额随时间有什么趋势？", self.dataframe, FakeChatModel()
        )

        self.assertEqual(result["chart_request"]["chart_type"], "line")
        self.assertEqual(result["chart_request"]["x_column"], "date")
        self.assertEqual(result["chart_request"]["y_column"], "sales")
        self.assertEqual(result["chart_result"]["status"], "generated")
        self.assertEqual(result["chart_figure"].data[0].type, "scatter")

    def test_generates_bar_chart_for_highest_product_question(self) -> None:
        model = FakeChatModel()

        result = run_agent("哪个产品销售额最高？", self.dataframe, model)

        self.assertEqual(result["chart_request"]["chart_type"], "bar")
        self.assertEqual(result["chart_request"]["x_column"], "product")
        self.assertEqual(result["chart_request"]["y_column"], "sales")
        self.assertEqual(result["question_context"]["records"][0]["product"], "Pro")
        self.assertEqual(result["question_context"]["records"][0]["sales"], 400.0)

    def test_generates_histogram_for_distribution_question(self) -> None:
        result = run_agent(
            "销售额的分布情况怎么样？", self.dataframe, FakeChatModel()
        )

        self.assertEqual(result["chart_request"]["chart_type"], "histogram")
        self.assertEqual(result["chart_request"]["x_column"], "sales")
        self.assertEqual(result["chart_figure"].data[0].type, "histogram")

    def test_unsupported_causal_question_is_sent_with_guardrails(self) -> None:
        model = FakeChatModel("现有字段无法判断销售变化是否由库存不足导致。")

        result = run_agent("销售下降是库存不足导致的吗？", self.dataframe, model)

        self.assertIn("无法判断", result["analysis_result"]["summary"])
        system_prompt = model.messages[0].content
        self.assertIn("Never invent causes", system_prompt)
        self.assertIn("insufficient", system_prompt)

    def test_incompatible_chart_falls_back_to_text_analysis(self) -> None:
        dataframe = pd.DataFrame({"sales": [100.0, 200.0]})

        result = run_agent("哪个产品销售额最高？", dataframe, FakeChatModel())

        self.assertEqual(result["chart_result"]["status"], "unavailable")
        self.assertIsNone(result["chart_figure"])
        self.assertIn("## Summary", result["final_response"])

    def test_invalid_llm_response_is_reported(self) -> None:
        with self.assertRaisesRegex(AgentError, "required JSON"):
            run_agent("总结这份数据", self.dataframe, InvalidChatModel())

    def test_llm_api_failure_is_reported(self) -> None:
        with self.assertRaisesRegex(AgentError, "LLM analysis failed"):
            run_agent("总结这份数据", self.dataframe, FailingChatModel())

    def test_only_compact_sample_is_sent_to_llm(self) -> None:
        model = FakeChatModel()
        dataframe = pd.DataFrame(
            {"product": [f"P{index}" for index in range(30)], "sales": range(30)}
        )

        result = run_agent("总结这份数据", dataframe, model)

        payload = json.loads(model.messages[1].content)
        self.assertEqual(len(payload["sample_rows"]), 5)
        self.assertEqual(result["dataset_summary"]["shape"]["rows"], 30)


class PlannedChatModel(FakeChatModel):
    """Stub only model I/O; real graph and pandas still execute."""

    def __init__(self, request: dict):
        super().__init__()
        self.request = request
        self.analyst_payload = None

    def invoke(self, messages: object, **kwargs: object) -> AIMessage:
        messages = list(messages)
        if "ONE safe pandas analysis request" in messages[0].content:
            return AIMessage(content=json.dumps({"request": self.request}))
        self.analyst_payload = json.loads(messages[1].content)
        # The fake analyst echoes computed evidence, never a hard-coded answer.
        self.summary = json.dumps(
            self.analyst_payload["calculation_result"], ensure_ascii=False
        )
        return super().invoke(messages, **kwargs)


class ProductQuantityTests(TestCase):
    def setUp(self) -> None:
        self.dataframe = pd.DataFrame({
            "Product": ["A", "B", "A", "B", "C", "A"],
            "Sales": [900, 1, 800, 1, 700, 600],
            "Quantity": [2, 8, 3, 7, 4, 1],
            "Region": ["East", "East", "West", "West", "East", "East"],
        })
        self.request = {
            "group_by": "Product", "metric": "Quantity",
            "aggregation": "sum", "sort": "desc",
        }

    def test_requested_product_quantity_questions_use_real_totals(self) -> None:
        for question in ("各产品总销量是多少？", "哪个产品销量最高？"):
            with self.subTest(question=question):
                model = PlannedChatModel(self.request)
                result = run_agent(question, self.dataframe, model)
                expected = [
                    {"Product": "B", "Quantity": 15},
                    {"Product": "A", "Quantity": 6},
                    {"Product": "C", "Quantity": 4},
                ]
                self.assertEqual(result["calculation_result"]["records"], expected)
                self.assertEqual(
                    model.analyst_payload["calculation_result"]["records"], expected
                )
                self.assertIn('"Quantity": 15', result["final_response"])
                if "最高" in question:
                    self.assertEqual(result["chart_request"]["y_column"], "Quantity")
                    self.assertEqual(list(result["chart_figure"].data[0].y), [15, 6, 4])

    def test_filtered_mean_chart_uses_same_calculation(self) -> None:
        request = {
            **self.request, "aggregation": "mean",
            "filters": [{"column": "Region", "op": "eq", "value": "East"}],
        }
        result = run_agent(
            "East 各产品平均销量对比", self.dataframe, PlannedChatModel(request)
        )
        self.assertEqual(list(result["chart_figure"].data[0].y), [8, 4, 1.5])

    def test_unsafe_request_is_not_executed_and_text_still_returns(self) -> None:
        result = run_agent(
            "哪个产品销量最高？", self.dataframe,
            PlannedChatModel({**self.request, "code": "raise RuntimeError()"}),
        )
        self.assertEqual(result["calculation_result"]["status"], "unavailable")
        self.assertIsNone(result["chart_figure"])
        self.assertIn("Only structured", result["final_response"])
