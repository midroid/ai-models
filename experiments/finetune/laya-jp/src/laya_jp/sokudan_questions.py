"""Question dicts copied from the vendored sokudan runners.

tests/test_payloads.py checks these against data/external/sokudan/bench_ja.py
and bench_en.py. Edit the vendored file and this module together, then update
data/manifest.json. Do not reword the options to chase a score.
"""

from __future__ import annotations

from typing import Any

DEPARTMENTS_JA: dict[str, str] = {
    "請求": "支払い・返金・請求書・料金の二重引き落としなど金銭処理に関するもの",
    "技術": "不具合・障害・エラー・ログイン不能など製品が動かないことに関するもの",
    "営業": "料金プラン・新規契約・見積もり・増席など購入判断に関するもの",
    "その他": "上のどれにも当てはまらない一般的な連絡",
}

URGENCY_JA: list[str] = ["急がない", "早めに", "業務が止まっている"]

DEPARTMENTS_EN: dict[str, str] = {
    "Billing": "payments, refunds, invoices, double charges -- anything about money",
    "Technical": "bugs, outages, errors, being unable to log in -- the product not working",
    "Sales": "pricing plans, new contracts, quotes, adding seats -- a purchase decision",
    "Other": "general correspondence that fits none of the above",
}

URGENCY_EN: list[str] = ["Not urgent", "Soon", "Work is blocked"]


def bench_questions_ja() -> dict[str, dict[str, Any]]:
    return {
        "department": {
            "type": "choice",
            "instructions": "この問い合わせはどの部署が担当すべきか",
            "criteria": dict(DEPARTMENTS_JA),
        },
        "urgency": {
            "type": "score",
            "instructions": "この依頼の緊急度は",
            "criteria": list(URGENCY_JA),
        },
        "churn": {
            "type": "noul",
            "instructions": "送信者は解約・契約終了を示唆しているか",
        },
    }


def bench_questions_en() -> dict[str, dict[str, Any]]:
    return {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this message?",
            "criteria": dict(DEPARTMENTS_EN),
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": list(URGENCY_EN),
        },
        "churn": {
            "type": "noul",
            "instructions": "Is the sender hinting that they may stop using the service?",
        },
    }
