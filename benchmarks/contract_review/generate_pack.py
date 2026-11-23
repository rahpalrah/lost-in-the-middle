"""Generate the synthetic contract pack used by the contract-review benchmark.

Business scenario: a buyer's contract-review desk handles a SaaS vendor's contract
pack - a Master Services Agreement (MSA) with a Data Processing Agreement (Exhibit A)
and an Order Form (Exhibit B) - plus an unrelated NDA with a different counterparty.

The documents are synthetic but written like real commercial contracts: numbered
articles, defined terms, cross-references, boilerplate, and deliberate distractors
(several different notice periods and day counts, two liability caps, conflicting
terms between the MSA and its exhibits resolved by an order-of-precedence clause).
Every fact a question depends on is planted exactly once, so the right answer is
known. Generation is deterministic (fixed seed).

Usage:
    python benchmarks/contract_review/generate_pack.py OUT_DIR
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from fpdf import FPDF

SEED = 20261121
FILLER_SCALE = 0.6  # share of boilerplate paragraphs kept; sets total length
VENDOR = "Northwind Cloud Systems, Inc."
CUSTOMER = "Contoso Retail Group Ltd."

# Facts the questions test. Keys are referenced from questions.json via "fact".
FACTS: dict[str, str] = {
    "term": (
        "The initial term of this Agreement begins on the Effective Date of March 1, 2026 "
        'and continues for thirty-six (36) months (the "Initial Term").'
    ),
    "renewal": (
        "Upon expiry of the Initial Term, this Agreement renews automatically for "
        "successive renewal terms of twelve (12) months each unless either party gives "
        "written notice of non-renewal at least one hundred twenty (120) days before the "
        "end of the then-current term."
    ),
    "precedence": (
        "In the event of any conflict, the following order of precedence applies: (a) the "
        "Data Processing Agreement in Exhibit A, solely with respect to the processing of "
        "Personal Data; (b) the Order Form in Exhibit B, solely with respect to fees, "
        "quantities and payment terms; and (c) the body of this Agreement."
    ),
    "msa_payment": (
        "Unless otherwise stated in an Order Form, Customer shall pay each undisputed "
        "invoice within thirty (30) days of the invoice date."
    ),
    "price_cap": (
        "Vendor may increase the subscription fees for any renewal term by giving at least "
        "sixty (60) days written notice, provided that any increase shall not exceed the "
        "lesser of five percent (5%) or the percentage increase in the CPI-U index over the "
        "preceding twelve (12) months."
    ),
    "sla": (
        "Vendor shall make the Services available at least 99.9% of the time in each "
        "calendar month, excluding Scheduled Maintenance."
    ),
    "credits": (
        "If Monthly Uptime falls below 99.9%, Customer is entitled to a service credit of "
        "ten percent (10%) of the monthly fees; if Monthly Uptime falls below 99.0%, the "
        "service credit is twenty-five percent (25%) of the monthly fees. Service credits "
        "in any month shall not exceed thirty percent (30%) of the monthly fees."
    ),
    "msa_breach_notice": (
        "Vendor shall notify Customer of any Security Incident affecting Customer Data "
        "without undue delay and in any event within seventy-two (72) hours of becoming "
        "aware of it."
    ),
    "liability_cap": (
        "Except for the Excluded Claims, each party's total aggregate liability arising out "
        "of or relating to this Agreement shall not exceed the total fees paid or payable by "
        "Customer in the twelve (12) months preceding the event giving rise to the claim."
    ),
    "data_super_cap": (
        "Vendor's total aggregate liability for breaches of Article 11 (Data Protection and "
        "Security) shall not exceed three (3) times the annual subscription fees payable "
        "under all Order Forms then in effect."
    ),
    "insurance": (
        "Vendor shall maintain cyber liability and technology errors and omissions insurance "
        "with limits of not less than ten million US dollars (US$10,000,000) per claim."
    ),
    "change_of_control": (
        "Vendor may assign this Agreement in connection with a change of control by giving "
        "Customer at least thirty (30) days prior written notice, and Customer may terminate "
        "this Agreement without payment of any early termination fee by written notice given "
        "within sixty (60) days after receiving such notice."
    ),
    "convenience": (
        "Customer may terminate this Agreement for convenience at any time after the first "
        "twelve (12) months of the Initial Term by giving Vendor not less than ninety (90) "
        "days prior written notice, provided that Customer pays an early termination fee "
        "equal to fifty percent (50%) of the fees that would have been payable for the "
        "remainder of the then-current term."
    ),
    "vendor_termination": (
        "Vendor may terminate this Agreement for convenience only at the end of a renewal "
        "term by giving not less than one hundred eighty (180) days prior written notice."
    ),
    "governing_law": (
        "This Agreement is governed by the laws of the State of New York, without regard to "
        "its conflict of laws rules, and the parties submit to the exclusive jurisdiction of "
        "the state and federal courts located in New York County, New York."
    ),
    "dpa_breach_notice": (
        "The Processor shall notify the Controller of a Personal Data Breach without undue "
        "delay and in any event within forty-eight (48) hours after becoming aware of it."
    ),
    "subprocessors": (
        "The Processor shall give the Controller at least thirty (30) days prior written "
        "notice of any new Sub-processor, and the Controller may object on reasonable data "
        "protection grounds within fifteen (15) days of such notice."
    ),
    "data_location": (
        "Customer Personal Data shall be stored and processed only in data centres located "
        "in Frankfurt, Germany and Dublin, Ireland."
    ),
    "order_fees": (
        "Subscription: Northwind Commerce Cloud, Enterprise Edition, 1,200 named users. "
        "Annual Subscription Fee: US$480,000, invoiced annually in advance."
    ),
    "order_payment": "Payment Terms: net forty-five (45) days from the invoice date.",
    "nda_law": (
        "This Non-Disclosure Agreement is governed by the laws of the State of California, "
        "and the confidentiality obligations survive for five (5) years after termination."
    ),
}

# MSA outline: (article title, [(subsection title, [fact keys], filler paragraphs)]).
MSA: list[tuple[str, list[tuple[str, list[str], int]]]] = [
    ("Definitions and Interpretation", [("Definitions", [], 14), ("Interpretation", [], 5)]),
    ("Term", [("Initial Term", ["term"], 4), ("Renewal", ["renewal"], 5)]),
    ("Order of Precedence", [("Conflicts", ["precedence"], 4)]),
    (
        "Services",
        [
            ("Provision of Services", [], 10),
            ("Service Changes", [], 7),
            ("Customer Responsibilities", [], 8),
        ],
    ),
    (
        "Fees and Payment",
        [("Invoicing", ["msa_payment"], 6), ("Taxes", [], 6), ("Disputed Invoices", [], 6)],
    ),
    ("Fee Changes", [("Renewal Pricing", ["price_cap"], 5), ("Audit of Fees", [], 6)]),
    ("Implementation", [("Implementation Plan", [], 9), ("Acceptance Testing", [], 8)]),
    (
        "Service Levels",
        [("Availability", ["sla"], 6), ("Service Credits", ["credits"], 6), ("Support", [], 9)],
    ),
    ("Personnel and Subcontractors", [("Key Personnel", [], 8), ("Subcontracting", [], 7)]),
    (
        "Intellectual Property",
        [("Ownership", [], 9), ("Licence to Customer", [], 8), ("Feedback", [], 5)],
    ),
    (
        "Data Protection and Security",
        [
            ("Security Program", [], 10),
            ("Security Incidents", ["msa_breach_notice"], 6),
            ("Audits", [], 7),
        ],
    ),
    ("Confidentiality", [("Obligations", [], 9), ("Exceptions", [], 6)]),
    (
        "Warranties",
        [("Mutual Warranties", [], 7), ("Service Warranty", [], 7), ("Disclaimer", [], 5)],
    ),
    ("Indemnification", [("Vendor Indemnity", [], 9), ("Indemnification Procedure", [], 7)]),
    (
        "Limitation of Liability",
        [
            ("Exclusion of Indirect Damages", [], 6),
            ("Liability Cap", ["liability_cap"], 5),
            ("Data Protection Cap", ["data_super_cap"], 5),
        ],
    ),
    ("Insurance", [("Required Coverage", ["insurance"], 6), ("Certificates", [], 5)]),
    ("Business Continuity", [("Disaster Recovery", [], 9), ("Backups", [], 7)]),
    ("Assignment", [("General", [], 6), ("Change of Control", ["change_of_control"], 6)]),
    (
        "Term and Termination",
        [
            ("Termination for Cause", [], 9),
            ("Termination for Convenience", ["convenience", "vendor_termination"], 7),
            ("Effect of Termination", [], 9),
            ("Transition Assistance", [], 8),
        ],
    ),
    (
        "Compliance",
        [("Anti-Corruption", [], 7), ("Export Control", [], 6), ("Accessibility", [], 5)],
    ),
    ("Dispute Resolution", [("Escalation", [], 7), ("Governing Law", ["governing_law"], 4)]),
    (
        "General Provisions",
        [
            ("Notices", [], 7),
            ("Force Majeure", [], 7),
            ("Entire Agreement", [], 5),
            ("Counterparts", [], 4),
        ],
    ),
]

DPA: list[tuple[str, list[str], int]] = [
    ("Scope and Roles", [], 7),
    ("Processing Instructions", [], 7),
    ("Personal Data Breach", ["dpa_breach_notice"], 5),
    ("Sub-processors", ["subprocessors"], 6),
    ("Data Location and Transfers", ["data_location"], 6),
    ("Return and Deletion", [], 6),
]

ORDER_FORM: list[tuple[str, list[str], int]] = [
    ("Subscription Details", ["order_fees"], 2),
    ("Payment", ["order_payment"], 2),
    ("Special Terms", [], 3),
]

# Building blocks for realistic, varied contract boilerplate.
_SUBJECTS = [
    "Each party",
    "Vendor",
    "Customer",
    "Neither party",
    "The receiving party",
    "Vendor and its Affiliates",
    "Customer's Authorised Users",
    "The disclosing party",
]
_OBLIGATIONS = [
    "shall use commercially reasonable efforts to",
    "shall promptly",
    "may, at its option,",
    "shall not unreasonably",
    "shall, upon written request,",
    "shall in good faith",
    "is responsible for ensuring that it will",
    "agrees to",
]
_ACTIONS = [
    "maintain accurate records of all activities performed under this Agreement",
    "cooperate with the other party to resolve operational issues",
    "provide reasonable assistance in connection with regulatory inquiries",
    "comply with all applicable laws and industry standards",
    "document any material changes to the Services in the release notes",
    "keep the other party informed of the status of open support requests",
    "review and update its internal policies at least once in each contract year",
    "ensure that its personnel receive appropriate training",
    "deliver the reports described in the applicable statement of work",
    "consult with the other party before making changes that affect integrations",
    "withhold consent to the extent permitted by this Agreement",
    "provide copies of relevant third-party certifications",
]
_QUALIFIERS = [
    "in accordance with Good Industry Practice",
    "subject to the confidentiality obligations in Article 12",
    "to the extent reasonably practicable",
    "within a reasonable time after receiving written notice",
    "as further described in the Documentation",
    "except as otherwise agreed in writing by the parties",
    "at no additional charge to Customer",
    "during the Subscription Term",
]
_FOLLOWUPS = [
    "Any failure to do so shall be addressed through the escalation process in Article 21.",
    "Such obligations apply equally to any Affiliate performing obligations hereunder.",
    "The parties acknowledge that this provision has been specifically negotiated.",
    "Nothing in this Section limits any other right or remedy available at law.",
    "Requests under this Section shall be made through the designated account manager.",
    "Reasonable costs incurred shall be borne by the requesting party unless agreed otherwise.",
    "Records shall be retained for the period required by applicable law.",
    "This obligation continues for so long as the relevant Services are provided.",
]


def _filler(rng: random.Random, sentences: int) -> str:
    parts = []
    for _ in range(sentences):
        parts.append(
            f"{rng.choice(_SUBJECTS)} {rng.choice(_OBLIGATIONS)} {rng.choice(_ACTIONS)} "
            f"{rng.choice(_QUALIFIERS)}."
        )
        if rng.random() < 0.35:
            parts.append(rng.choice(_FOLLOWUPS))
    return " ".join(parts)


class _Doc(FPDF):
    def __init__(self, title: str) -> None:
        super().__init__(format="Letter")
        self.doc_title = title
        self.set_auto_page_break(auto=True, margin=20)
        self.set_margins(22, 20, 22)

    def header(self) -> None:
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 6, self.doc_title, align="R", new_x="LMARGIN", new_y="NEXT")

    def footer(self) -> None:
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.cell(0, 6, f"Page {self.page_no()}", align="C")

    def heading(self, text: str, size: int = 12) -> None:
        self.ln(3)
        self.set_font("Helvetica", "B", size)
        self.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def para(self, text: str) -> None:
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)


def _section(doc: _Doc, rng: random.Random, title: str, facts: list[str], paras: int) -> None:
    doc.heading(title, 10)
    paras = max(1, round(paras * FILLER_SCALE))
    slots = sorted(rng.sample(range(paras + len(facts)), len(facts))) if facts else []
    fact_iter = iter(facts)
    for i in range(paras + len(facts)):
        if i in slots:
            doc.para(FACTS[next(fact_iter)])
        else:
            doc.para(_filler(rng, rng.randint(3, 6)))


def build_contract_pack(path: Path) -> None:
    """Write the MSA + DPA + Order Form pack as one PDF (exhibits attached)."""
    rng = random.Random(SEED)
    doc = _Doc(f"Master Services Agreement - {VENDOR} / {CUSTOMER} - CONFIDENTIAL")
    doc.add_page()
    doc.set_font("Helvetica", "B", 18)
    doc.ln(40)
    doc.multi_cell(0, 10, "MASTER SERVICES AGREEMENT", align="C", new_x="LMARGIN", new_y="NEXT")
    doc.set_font("Helvetica", "", 12)
    doc.multi_cell(
        0,
        7,
        f'between\n{VENDOR} ("Vendor")\nand\n{CUSTOMER} ("Customer")\n\n'
        "Agreement No. NW-CRG-2026-0147",
        align="C",
        new_x="LMARGIN",
        new_y="NEXT",
    )
    doc.add_page()
    for n, (article, sections) in enumerate(MSA, start=1):
        doc.heading(f"ARTICLE {n}. {article.upper()}")
        for m, (title, facts, paras) in enumerate(sections, start=1):
            _section(doc, rng, f"{n}.{m} {title}", facts, paras)

    doc.add_page()
    doc.heading("EXHIBIT A - DATA PROCESSING AGREEMENT", 13)
    doc.para(
        f"This Data Processing Agreement forms part of the Agreement between {CUSTOMER} as "
        f"Controller and {VENDOR} as Processor."
    )
    for m, (title, facts, paras) in enumerate(DPA, start=1):
        _section(doc, rng, f"A.{m} {title}", facts, paras)

    doc.add_page()
    doc.heading("EXHIBIT B - ORDER FORM NO. 1", 13)
    for m, (title, facts, paras) in enumerate(ORDER_FORM, start=1):
        _section(doc, rng, f"B.{m} {title}", facts, paras)
    doc.output(str(path))


def build_nda(path: Path) -> None:
    """A short, unrelated NDA - used to check documents don't leak into each other."""
    rng = random.Random(SEED + 1)
    doc = _Doc("Mutual Non-Disclosure Agreement - Contoso / Fabrikam Logistics")
    doc.add_page()
    doc.heading("MUTUAL NON-DISCLOSURE AGREEMENT", 14)
    for m, (title, facts, paras) in enumerate(
        [
            ("Purpose", [], 3),
            ("Confidential Information", [], 4),
            ("Governing Law", ["nda_law"], 2),
        ],
        start=1,
    ):
        _section(doc, rng, f"{m}. {title}", facts, paras)
    doc.output(str(path))


def main(argv: list[str]) -> int:
    out = Path(argv[1] if len(argv) > 1 else "benchmarks/contract_review/data")
    out.mkdir(parents=True, exist_ok=True)
    build_contract_pack(out / "northwind_msa_pack.pdf")
    build_nda(out / "fabrikam_nda.pdf")
    print(f"Wrote {out / 'northwind_msa_pack.pdf'} and {out / 'fabrikam_nda.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
