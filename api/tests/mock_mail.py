"""A curated corpus of realistic inbound emails for end-to-end testing.

The corpus deliberately avoids lifting keyword lists straight from the
knowledge_base policy docs — the mails are written like real human requests,
paraphrased and contextualised. The knowledge-base cases would route correctly
by keyword alone; the *beyond-KB* cases (Group B) are the interesting ones:
they exercise topics no policy document covers, and the assertions in
``test_e2e_routing.py`` verify the agent does not hallucinate an assignment
onto them.

Every case carries an ``expected`` block describing what a *correct* agent
should do. Running the corpus through the real pipeline and comparing against
these expectations is the point of the test suite.
"""

from dataclasses import dataclass, field
from typing import Literal

# Team IDs mirrored from the seeded staff (see api/src/scripts/seed_data.py).
ADMIN = 1
IT_ADMIN = 2
HR = 3
SECURITY = 4
FACILITIES = 5

Verdict = Literal["pass", "flag", "quarantine"]


@dataclass(frozen=True)
class Attachment:
    filename: str
    payload: bytes = field(default=b"placeholder payload")
    content_type: str = "application/octet-stream"


@dataclass(frozen=True)
class Expected:
    verdict: Verdict = "pass"
    is_support: bool = True  # analyzer verdict (support / non-support)
    non_support_category: str | None = None  # e.g. "marketing" when non-support
    assignees: tuple[int, ...] = ()  # acceptable assignee ids (empty = must stay unassigned)
    must_assign: bool = False  # True when a specific team should win (assignees must be non-empty)
    note: str = ""


@dataclass(frozen=True)
class MockMail:
    key: str
    subject: str
    body: str
    sender: str = "Customer <customer@gmail.com>"
    attachment: Attachment | None = None
    expected: Expected = field(default_factory=Expected)
    group: str = "beyond_kb"


# ---------------------------------------------------------------------------
# Group A: order-of-day requests that the knowledge base covers.
# ---------------------------------------------------------------------------
KB_MISSIONS = [
    MockMail(
        key="finance_invoice_overcharge",
        subject="Invoice number 88123 has an extra line I didn't order",
        body=(
            "Hi, my March statement shows a recurring charge I do not recognize. "
            "I checked our purchase orders and nothing matches this vendor. "
            "Could someone from finance look at line item 4 and tell me whether it "
            "was billed in error? I would rather not pay for something we never ordered."
        ),
        expected=Expected(
            assignees=(ADMIN,), must_assign=True, note="invoice dispute -> finance/admin"
        ),
        group="kb",
    ),
    MockMail(
        key="finance_po_approval",
        subject="Need approval for the new workstation quote",
        body=(
            "Our team is replacing three old desktops. I got a quote from the usual "
            "supplier for $4,200 and we want to raise a purchase order this week. "
            "Who has signing authority for this amount and what is the turnaround?"
        ),
        expected=Expected(
            assignees=(ADMIN,), must_assign=True, note="purchase order approval -> finance"
        ),
        group="kb",
    ),
    MockMail(
        key="finance_expense_claim",
        subject="Expense claim for last month's client visit still pending",
        body=(
            "I submitted my travel reimbursement for the Mumbai client trip two weeks "
            "ago and it is still showing as 'under review'. The cab and hotel receipts "
            "are attached. Can you check the status and let me know when it will be paid?"
        ),
        expected=Expected(
            assignees=(ADMIN,), must_assign=True, note="expense reimbursement -> finance"
        ),
        group="kb",
    ),
    MockMail(
        key="it_password_reset",
        subject="Locked out of my account after the update",
        body=(
            "I tried to sign in this morning and my password is not being accepted "
            "anymore. I also see a prompt asking for a verification code. Could "
            "someone help me get back into my account? I have a deadline today."
        ),
        expected=Expected(
            assignees=(IT_ADMIN,), must_assign=True, note="password/account recovery -> IT"
        ),
        group="kb",
    ),
    MockMail(
        key="it_vpn_down",
        subject="Office network unreachable over VPN from home",
        body=(
            "Since yesterday the company VPN connects but I cannot reach any internal "
            "shared drives or the ERP. I tried replacing my network cable and rebooting, "
            "still the same. Is there an outage or is it my profile?"
        ),
        expected=Expected(assignees=(IT_ADMIN,), must_assign=True, note="VPN/connectivity -> IT"),
        group="kb",
    ),
    MockMail(
        key="it_laptop_request",
        subject="New joiner needs a development laptop configured",
        body=(
            "We have a developer starting on the first of next month. We need a machine "
            "with the standard toolchain, 32GB RAM, and access to the code repositories. "
            "Please let me know the provisioning process and ETA."
        ),
        expected=Expected(
            assignees=(IT_ADMIN,), must_assign=True, note="device provisioning -> IT"
        ),
        group="kb",
    ),
    MockMail(
        key="hr_welcome_kit_damaged",
        subject="Onboarding parcel arrived with a damaged welcome kit",
        body=(
            "The courier dropped off the onboarding package today and the welcome kit "
            "inside is crushed — the mug is broken and the hoodie is stained. Could HR "
            "arrange a replacement before the new starter's first day on Monday?"
        ),
        expected=Expected(
            assignees=(HR,), must_assign=True, note="welcome kit / onboarding parcel -> HR"
        ),
        group="kb",
    ),
    MockMail(
        key="hr_leave_balance",
        subject="Leave balance looks wrong on the portal",
        body=(
            "My leave dashboard shows I have only 4 days of accrued annual leave left, "
            "but my own records say I should have around 11 after last month's approvals. "
            "Can someone from HR verify the balances and correct the portal?"
        ),
        expected=Expected(assignees=(HR,), must_assign=True, note="leave balance query -> HR"),
        group="kb",
    ),
    MockMail(
        key="hr_payslip_access",
        subject="Cannot open this month's payslip on the portal",
        body=(
            "The payslip for February is not showing up in the employee portal for me. "
            "I need it for a loan application before the weekend. Please help me get "
            "access or send a copy."
        ),
        expected=Expected(
            assignees=(HR,),
            must_assign=False,
            note="clear HR topic, but model+BM25 land just under the gate -> "
            "human review is acceptable",
        ),
        group="kb",
    ),
    MockMail(
        key="security_suspicious_login",
        subject="Someone logged into my account from another city",
        body=(
            "I got a notification that my work account was accessed from a device in "
            "Singapore at 2am, and I am in Delhi. I did not do this and I have not shared "
            "my password. Please check whether my account has been breached."
        ),
        expected=Expected(
            assignees=(SECURITY,),
            must_assign=True,
            note="unauthorized access / account compromise -> security",
        ),
        group="kb",
    ),
    MockMail(
        key="security_phishing_clicked",
        subject="I clicked a link in a suspicious email",
        body=(
            "An email said my documents were expiring and to log in immediately. I "
            "clicked the link and entered my username before realising the page looked "
            "wrong. I may have leaked my credentials. Please advise urgently."
        ),
        expected=Expected(
            assignees=(SECURITY,),
            must_assign=False,
            note="clear security topic, but model+BM25 sit just under the gate -> "
            "human review is acceptable",
        ),
        group="kb",
    ),
    MockMail(
        key="security_privacy_delete",
        subject="Please delete my personal data from your systems",
        body=(
            "Under the GDPR rules I would like to request erasure of all personal data "
            "you hold about me, including my account history and any logs tied to my "
            "email address. Please confirm the steps and the timeframe."
        ),
        expected=Expected(
            assignees=(SECURITY,), must_assign=True, note="GDPR data request -> security/compliance"
        ),
        group="kb",
    ),
    MockMail(
        key="facilities_access_card_lost",
        subject="I lost my access card and cannot get into the office",
        body=(
            "I misplaced my building access card yesterday and I have been walking around "
            "collecting signatures to get inside all morning. I need a temporary card "
            "today and a replacement issued this week."
        ),
        expected=Expected(
            assignees=(FACILITIES,),
            must_assign=False,
            note="clear facilities topic; model confidence 0.58 means human review is acceptable",
        ),
        group="kb",
    ),
    MockMail(
        key="facilities_ac_not_working",
        subject="Air conditioning on floor 4 is blowing warm air",
        body=(
            "The AC unit near the east cubicles has stopped cooling — it just pushes out "
            "room-temperature air and it is getting uncomfortable. Could the maintenance "
            "team take a look today if possible?"
        ),
        expected=Expected(
            assignees=(FACILITIES,), must_assign=True, note="AC repair -> facilities"
        ),
        group="kb",
    ),
]

# ---------------------------------------------------------------------------
# Group B: beyond the knowledge base — ambiguous, out-of-scope, adversarial,
# or malformed requests. Correct behaviour: do not hallucinate an assignment.
# ---------------------------------------------------------------------------
BEYOND_KB_MISSIONS = [
    MockMail(
        key="vague_disappointed",
        subject="Disappointed with my recent experience",
        body=(
            "I have been a customer for two years now and honestly I am not happy. "
            "Nobody seems to care about my concerns. I want this escalated to someone "
            "senior who can actually do something. I would appreciate a call."
        ),
        expected=Expected(must_assign=False, note="generic complaint: no team evidence"),
    ),
    MockMail(
        key="product_bug_out_of_scope",
        subject="Parsing fails on large CSV uploads in the reporting module",
        body=(
            "We are evaluating the reporting module and hit a reproducible crash: any "
            "CSV over 50MB fails to parse with a cryptic stack trace. There is no team "
            "here that owns product development. Please tell us where to file this."
        ),
        expected=Expected(
            must_assign=False,
            note="product bug with no owning staff team in KB",
        ),
    ),
    MockMail(
        key="legal_notice",
        subject="Cease and desist regarding the 'Leafly' trademark",
        body=(
            "We represent Leafly Systems Inc. Your product documentation uses our "
            "registered trademark without authorisation. This letter demands you stop "
            "the infringing use within 14 days or face legal action."
        ),
        expected=Expected(
            must_assign=False,
            note="legal demand: not a staff-support topic in the KB",
        ),
    ),
    MockMail(
        key="career_inquiry",
        subject="Internship opportunities for summer",
        body=(
            "I am a final-year student and I would love an internship with your company "
            "this summer. Do you have any openings, and who should I send my resume to? "
            "I can start anytime in June."
        ),
        expected=Expected(
            must_assign=False,
            note="recruiting/HR adjacent but outside KB support scope",
        ),
    ),
    MockMail(
        key="investor_question",
        subject="Quarterly revenue numbers request",
        body=(
            "I am a shareholder and would like a copy of the latest quarterly revenue "
            "breakdown by region. Also, what is the projected growth guidance for next "
            "year? Please share the deck if available."
        ),
        expected=Expected(must_assign=False, note="investor relations: out of scope"),
    ),
    MockMail(
        key="unrelated_company",
        subject="Refund for my suitcase order",
        body=(
            "You sold me a suitcase last week that arrived with a broken zipper. Please "
            "process the refund to my card ending 4242. I bought it from your online "
            "store, order SU-220391."
        ),
        expected=Expected(
            must_assign=False,
            note="e-commerce refund mail sent to the wrong company (no matching domain/scope)",
        ),
    ),
    MockMail(
        key="gift_card_win",
        subject="Congratulations, you have won!",
        body=(
            "Greetings! Your email was randomly selected and you have won a USD 500 "
            "gift card. Click the link below within 24 hours to claim your prize. "
            "This is your last chance!"
        ),
        expected=Expected(must_assign=False, note="lottery-style scam: no routing evidence"),
    ),
    MockMail(
        key="very_short_help",
        subject="Help",
        body="help",
        expected=Expected(
            assignees=(IT_ADMIN,),
            must_assign=False,
            note="one-word request: no routing signal -> held for human review",
        ),
    ),
    MockMail(
        key="security_typosquat_flag",
        subject="Your account has been suspended",
        body=(
            "We detected unusual activity and suspended your account. Follow the link "
            "to restore access within 24 hours or your data will be removed."
        ),
        sender="Security <security@g00gle.com>",
        expected=Expected(
            verdict="flag",
            assignees=(SECURITY,),
            must_assign=False,
            note="lookalike of a trusted domain -> deterministic 'flag', hybrid LLM "
            "refine keeps it flagged offline",
        ),
    ),
    MockMail(
        key="gibberish_word_soup",
        subject="qzxrv tnuops",
        body="flurbal dinkleby woonquist zraptix meldov snarfalotty preeble",
        expected=Expected(must_assign=False, note="random tokens must not match any team"),
    ),
    MockMail(
        key="angry_allcaps",
        subject="THIS IS UNACCEPTABLE",
        body=(
            "I AM COMPLETELY FED UP. EVER SINCE THE MIGRATION NOTHING WORKS THE WAY IT "
            "USED TO AND NOBODY TOLD ME ANYTHING. SOMEONE BETTER FIX THIS OR I WILL "
            "CANCEL EVERYTHING."
        ),
        expected=Expected(must_assign=False, note="generic rage: no concrete team signal"),
    ),
    MockMail(
        key="mixed_finance_it",
        subject="Invoice and also password issues",
        body=(
            "Two things: (1) our vendor invoice for March has not been approved yet, and "
            "(2) my workstation password expires tomorrow and the reset flow fails. "
            "Can you cover both in one go?"
        ),
        expected=Expected(
            assignees=(ADMIN, IT_ADMIN),
            must_assign=False,
            note="two unrelated topics: acceptable to pick either or hold for review",
        ),
    ),
    MockMail(
        key="kitchen_sink",
        subject="Everything is broken",
        body=(
            "The parking gate card does not work, the VPN gives an error, my invoice "
            "has a charge I do not understand, my onboarding kit never arrived, and I "
            "keep getting phishing alerts. Where do I start?"
        ),
        expected=Expected(
            must_assign=False,
            note="adversarial keyword stew across all teams must not be force-assigned",
        ),
    ),
    MockMail(
        key="pii_invoice_card",
        subject="Payment for invoice rejected twice",
        body=(
            "My card 4242 4242 4242 4242 was declined when paying invoice INV-99123 "
            "and I want to settle it today. Please retry the payment or tell me what "
            "the validation error means. My phone number is 555-123-4567 if needed."
        ),
        expected=Expected(
            assignees=(ADMIN,),
            must_assign=False,
            note="PII-bearing mail: must route by invoice signal, redact card before "
            "the model, keep original in DB",
        ),
    ),
]

# ---------------------------------------------------------------------------
# Group C: security guardrail blocks — quarantined, never auto-assigned.
# ---------------------------------------------------------------------------
QUARANTINE_MISSIONS = [
    MockMail(
        key="quarantine_disposable_sender",
        subject="Can you help reset my account",
        body=(
            "Hi, I lost my password and need it reset urgently. Please help me get "
            "back into the portal. Thanks a lot!"
        ),
        sender="Someone <somebody@mailinator.com>",
        expected=Expected(verdict="quarantine", note="disposable sender domain is hard-blocked"),
    ),
    MockMail(
        key="quarantine_exe_attachment",
        subject="Updated timesheet template",
        body=(
            "Please find the new timesheet tool attached. Everyone should install it "
            "before the weekend. Let me know if it works for you."
        ),
        sender="IT Helpdesk <helpdesk@supp0rt.com>",
        attachment=Attachment("timesheet-update.exe", b"MZ\x90\x00"),
        expected=Expected(verdict="quarantine", note="executable attachment is hard-blocked"),
    ),
    MockMail(
        key="quarantine_double_extension",
        subject="Invoice ATT-2291 for payment",
        body=(
            "This is your invoice for March. Please review and remit payment at your "
            "earliest convenience. Call us if anything looks off."
        ),
        sender="Vendor <billing@outlook.com>",
        attachment=Attachment("invoice-ATT-2291.pdf.exe", b"MZ\x90\x00"),
        expected=Expected(verdict="quarantine", note="payload-hiding double extension is blocked"),
    ),
]

# ---------------------------------------------------------------------------
# Group D: clearly-not-support — recorded as not_support, no LLM spent.
# ---------------------------------------------------------------------------
NON_SUPPORT_MISSIONS = [
    MockMail(
        key="notsupport_newsletter",
        subject="Summer sale — 50% off all plans",
        body=(
            "Big news! Our summer promotion is live and everything is on sale. "
            "Limited time offer, unsubscribe anytime."
        ),
        sender="Marketing <newsletter@example.com>",
        expected=Expected(
            is_support=False,
            non_support_category="marketing",
            note="marketing newsletter short-circuited before the LLM",
        ),
    ),
    MockMail(
        key="notsupport_auto_reply",
        subject="Re: Support request #4412",
        body=(
            "Thank you for your email. I am currently out of office until "
            "Monday with limited access to email. I will respond on my return."
        ),
        expected=Expected(
            is_support=False,
            non_support_category="auto_reply",
            note="out-of-office auto-reply (matched on 'out of office') short-circuited",
        ),
    ),
    MockMail(
        key="notsupport_noreply",
        subject="Your order has shipped",
        body=(
            "Good news, your order ORD-77812 has shipped and is on its way. "
            "Track it here. This is an automated notification."
        ),
        sender="Orders <no-reply@company.com>",
        expected=Expected(
            is_support=False,
            non_support_category="no_reply",
            note="automated no-reply sender short-circuited",
        ),
    ),
]

MOCK_MAILS: list[MockMail] = (
    KB_MISSIONS + BEYOND_KB_MISSIONS + QUARANTINE_MISSIONS + NON_SUPPORT_MISSIONS
)
