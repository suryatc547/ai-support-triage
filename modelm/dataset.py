"""Domain training dataset for the custom support ticket classifier.

Contains curated examples spanning IT Support, Finance & Procurement,
Human Resources, Security & Compliance, and Facilities & Admin.
"""

TRAINING_DATA = [
    # -------------------------------------------------------------------------
    # IT Support
    # -------------------------------------------------------------------------
    ("VPN connection keeps dropping when working remotely", "it_support"),
    ("My laptop screen is flickering and shutting down randomly", "it_support"),
    ("Need software license for IntelliJ IDEA and Docker Desktop", "it_support"),
    ("Cannot connect to office Wi-Fi network on my MacBook", "it_support"),
    ("Printer on the 3rd floor is jammed and shows error 502", "it_support"),
    ("Need a second monitor and HDMI adapter for my workstation", "it_support"),
    ("Password reset required for company Active Directory account", "it_support"),
    ("Blue screen of death on Windows 11 after recent update", "it_support"),
    ("Keyboard keys are sticking, request replacement wireless keyboard", "it_support"),
    ("Need access to the internal development server via SSH", "it_support"),
    ("Outlook email client is not syncing incoming messages", "it_support"),
    ("HotSpot is not connecting to device helpdesk needed", "it_support"),
    ("Switching to a new project, please arrange a laptop for me", "it_support"),
    ("VPN is not working and the Wi-Fi is very slow, helpdesk", "it_support"),
    ("The printer is jammed and not printing, please fix", "it_support"),
    ("Need mouse and ergonomic wrist rest for my desk", "it_support"),
    ("Audio and microphone not detected during Zoom video call", "it_support"),
    ("Request for local admin permissions to install developer SDKs", "it_support"),
    (
        "MacBook battery draining rapidly and overheating during compilation",
        "it_support",
    ),
    ("Help setting up remote desktop connection to the test environment", "it_support"),
    ("VPN broken and cannot connect to office network", "it_support"),
    ("VPN is broken and authentication fails on MacBook", "it_support"),
    ("Broken laptop charger cable and damaged power adapter", "it_support"),
    # -------------------------------------------------------------------------
    # Finance & Procurement
    # -------------------------------------------------------------------------
    ("Need approval for vendor invoice #48291 from AWS Cloud", "finance_procurement"),
    (
        "Expense reimbursement for client dinner travel pending approval",
        "finance_procurement",
    ),
    (
        "Purchase requisition for 10 developer workstations and monitors",
        "finance_procurement",
    ),
    (
        "When will the Q3 travel allowance reimbursement be processed?",
        "finance_procurement",
    ),
    (
        "Vendor payment inquiry from TechCorp for consulting services",
        "finance_procurement",
    ),
    ("Cost center budget allocation inquiry for marketing team", "finance_procurement"),
    (
        "Renew annual subscription for GitHub Enterprise organization",
        "finance_procurement",
    ),
    (
        "Invoice approval needed for office supplies bulk purchase",
        "finance_procurement",
    ),
    (
        "Need purchase order PO-99201 generated for server hardware order",
        "finance_procurement",
    ),
    ("Expense report query rejected due to missing GST receipt", "finance_procurement"),
    (
        "Corporate credit card limit increase request for project travel",
        "finance_procurement",
    ),
    ("Quarterly financial audit documentation submission", "finance_procurement"),
    (
        "Approval required for software subscription renewal contract",
        "finance_procurement",
    ),
    (
        "Vendor billing discrepancy on latest Google Workspace invoice",
        "finance_procurement",
    ),
    ("Please approve invoice #44 for software licenses", "finance_procurement"),
    (
        "Requesting budget approval for attending tech conference in Berlin",
        "finance_procurement",
    ),
    ("Petty cash voucher claim for team lunch celebration", "finance_procurement"),
    ("Procurement of new testing mobile devices for QA team", "finance_procurement"),
    ("Invoice submission for legal consulting retainer fee", "finance_procurement"),
    (
        "Query regarding tax deduction on annual vendor contract payment",
        "finance_procurement",
    ),
    # -------------------------------------------------------------------------
    # Human Resources
    # -------------------------------------------------------------------------
    (
        "Question about medical health insurance policy coverage for dependents",
        "human_resources",
    ),
    ("Salary discrepancy in this month's payslip breakdown", "human_resources"),
    ("Request for paternity leave and parental time off guidelines", "human_resources"),
    ("New employee onboarding checklist and documentation query", "human_resources"),
    ("Updating home address and emergency contact details in HRMS", "human_resources"),
    ("Query about annual performance review and appraisal cycle", "human_resources"),
    (
        "Request for employment verification letter for visa application",
        "human_resources",
    ),
    ("Unused paid time off PTO carry forward policy clarification", "human_resources"),
    (
        "Employee referral bonus payout status for software engineer hire",
        "human_resources",
    ),
    (
        "Query regarding notice period and resignation handover process",
        "human_resources",
    ),
    ("Training program and education reimbursement subsidy request", "human_resources"),
    ("Bereavement leave application for immediate family member", "human_resources"),
    (
        "Update bank account details for monthly payroll direct deposit",
        "human_resources",
    ),
    ("Employee stock option plan ESOP vesting schedule query", "human_resources"),
    (
        "Request for official copy of signed employment contract amendment",
        "human_resources",
    ),
    (
        "Attendance record correction for days worked during business trip",
        "human_resources",
    ),
    (
        "Maternity leave extension request and doctor certificate submission",
        "human_resources",
    ),
    (
        "Clarification on company remote work policy and core working hours",
        "human_resources",
    ),
    ("Health wellness allowance reimbursement claim submission", "human_resources"),
    ("Relocation assistance allowance inquiry for office transfer", "human_resources"),
    (
        "I received my welcome kits today however the parcel and kit has minor damages",
        "human_resources",
    ),
    (
        "Welcome kit parcel arrived with damaged contents, need replacement",
        "human_resources",
    ),
    (
        "Delivery damage on my new hire welcome kit package",
        "human_resources",
    ),
    (
        "When will my onboarding welcome kit and parcel be shipped",
        "human_resources",
    ),
    (
        "Parcel containing my welcome kit was damaged during transit",
        "human_resources",
    ),
    (
        "Received damaged onboarding parcel, how to proceed",
        "human_resources",
    ),
    (
        "Welcome kit items missing or damaged in delivery parcel",
        "human_resources",
    ),
    # -------------------------------------------------------------------------
    # Security & Compliance
    # -------------------------------------------------------------------------
    (
        "Received a suspicious phishing email asking for corporate credentials",
        "security_compliance",
    ),
    (
        "Urgent: Potential data breach suspected on staging database server",
        "security_compliance",
    ),
    (
        "Lost my phone with Microsoft Authenticator MFA two-factor authentication",
        "security_compliance",
    ),
    (
        "Immediate account access revocation for terminated contractor",
        "security_compliance",
    ),
    (
        "Antivirus alert detected potential malware Trojan on desktop PC",
        "security_compliance",
    ),
    (
        "GDPR data deletion request received from European customer",
        "security_compliance",
    ),
    (
        "Security audit vulnerability report remediation follow-up",
        "security_compliance",
    ),
    (
        "Unauthorized login attempt detected on my corporate email account",
        "security_compliance",
    ),
    (
        "Report suspicious USB drive found in company conference room",
        "security_compliance",
    ),
    (
        "Ransomware warning pop-up appeared while browsing internal wiki",
        "security_compliance",
    ),
    (
        "SOC2 compliance policy acknowledgment and audit trail documentation",
        "security_compliance",
    ),
    (
        "Request for penetration test approval on external staging API",
        "security_compliance",
    ),
    (
        "Security team approval needed for third-party SaaS cloud integration",
        "security_compliance",
    ),
    ("Multiple failed login alerts triggered on VPN gateway", "security_compliance"),
    (
        "Confidential customer data accidentally emailed to wrong recipient",
        "security_compliance",
    ),
    (
        "Request for security risk assessment of new mobile vendor SDK",
        "security_compliance",
    ),
    ("Employee laptop reported stolen from vehicle overnight", "security_compliance"),
    (
        "Suspicious network traffic and outbound port scanning detected",
        "security_compliance",
    ),
    ("MFA token reset requested after replacing mobile handset", "security_compliance"),
    (
        "Access permissions review for production payment gateway credentials",
        "security_compliance",
    ),
    # -------------------------------------------------------------------------
    # Facilities & Admin
    # -------------------------------------------------------------------------
    (
        "My office access smart card RFID badge is not opening main gate",
        "facilities_admin",
    ),
    (
        "Air conditioning AC in meeting room 4B is not working properly",
        "facilities_admin",
    ),
    (
        "Request for reserved car parking pass sticker in basement level 2",
        "facilities_admin",
    ),
    (
        "Desk booking hot desking system shows error when booking workstation",
        "facilities_admin",
    ),
    (
        "Office chair has broken armrest and hydraulic lift mechanism",
        "facilities_admin",
    ),
    (
        "Whiteboard markers and stationary supplies needed in boardroom",
        "facilities_admin",
    ),
    (
        "Spill on carpet near cafeteria area needs immediate cleaning",
        "facilities_admin",
    ),
    (
        "Request for office visitor pass for vendor visiting tomorrow",
        "facilities_admin",
    ),
    (
        "Desk relocation request from 2nd floor east wing to 3rd floor",
        "facilities_admin",
    ),
    ("Lighting fixture flickering in hallway outside quiet room", "facilities_admin"),
    ("Restroom plumbing maintenance issue on 4th floor west wing", "facilities_admin"),
    (
        "Filing cabinet key replacement request for locked document drawer",
        "facilities_admin",
    ),
    (
        "Need standing desk converter installed on assigned desk #204",
        "facilities_admin",
    ),
    (
        "Cafeteria coffee machine is out of order and needs servicing",
        "facilities_admin",
    ),
    (
        "Fire extinguisher inspection schedule and building safety drill",
        "facilities_admin",
    ),
    ("Office pest control schedule inquiry for open plan floor", "facilities_admin"),
    (
        "Meeting room projector bulb replacement required in Apollo Room",
        "facilities_admin",
    ),
    (
        "Hot water dispenser in kitchen pantry area not heating water",
        "facilities_admin",
    ),
    (
        "Request for temporary storage locker during office refurbishment",
        "facilities_admin",
    ),
    ("Window blinds broken in executive conference room 3A", "facilities_admin"),
    # Additional realistic enterprise support queries
    ("Docking station USB ports and dual monitors not detecting laptop", "it_support"),
    ("Cannot access company intranet portal from home network without VPN", "it_support"),
    ("Request for mechanical ergonomic keyboard and vertical mouse", "it_support"),
    ("Need PostgreSQL client and Python SDK installed on corporate machine", "it_support"),
    ("Account locked out after three failed Active Directory password attempts", "it_support"),
    ("Webcam video flickers and freezes during Microsoft Teams meetings", "it_support"),
    ("Need USB-C to HDMI converter cable for conference room presentation", "it_support"),
    (
        "Requesting vendor payment status for invoice #88392 due this Friday",
        "finance_procurement",
    ),
    (
        "Please approve reimbursement claim for international business flight",
        "finance_procurement",
    ),
    (
        "Need purchase order issued for new team software licenses",
        "finance_procurement",
    ),
    (
        "Inquiry regarding monthly GST tax deduction on vendor consulting invoices",
        "finance_procurement",
    ),
    (
        "Corporate credit card transaction declined while traveling overseas",
        "finance_procurement",
    ),
    (
        "Budget approval request for attending cloud architecture summit",
        "finance_procurement",
    ),
    (
        "Vendor billing address update for annual software maintenance contract",
        "finance_procurement",
    ),
    (
        "How do I request a replacement for damaged items in my welcome kit?",
        "human_resources",
    ),
    (
        "Welcome kit courier arrived today but the contents inside were damaged",
        "human_resources",
    ),
    (
        "Onboarding documentation completed, when will my employee ID be issued?",
        "human_resources",
    ),
    (
        "Request for official salary certificate letter for mortgage loan",
        "human_resources",
    ),
    (
        "Adding newborn baby dependent to corporate medical insurance coverage",
        "human_resources",
    ),
    (
        "Inquiry regarding carry forward of accrued privilege leave balance",
        "human_resources",
    ),
    (
        "Annual appraisal review rating query and promotion eligibility criteria",
        "human_resources",
    ),
    (
        "Received an email asking for password verification from external sender",
        "security_compliance",
    ),
    (
        "Suspicious login attempt notification from an unrecognized IP address",
        "security_compliance",
    ),
    (
        "Need emergency access revocation for contractor whose agreement ended",
        "security_compliance",
    ),
    (
        "Employee laptop infected with suspicious browser redirect adware",
        "security_compliance",
    ),
    (
        "Security review questionnaire needed for enterprise client contract",
        "security_compliance",
    ),
    (
        "MFA authenticator app lost access after factory resetting mobile device",
        "security_compliance",
    ),
    (
        "Report suspected confidential file download to personal USB drive",
        "security_compliance",
    ),
    (
        "Air conditioning thermostat in open bay area is stuck at 18 degrees",
        "facilities_admin",
    ),
    (
        "Keycard reader at secondary entrance door is not responding to badges",
        "facilities_admin",
    ),
    (
        "Need whiteboard markers, eraser pads, and flip charts in conference room",
        "facilities_admin",
    ),
    (
        "Office chair hydraulic cylinder sinks continuously, request replacement",
        "facilities_admin",
    ),
    (
        "Request reserved parking slot allocation for electric vehicle charging",
        "facilities_admin",
    ),
    (
        "Water leakage observed near restroom entrance on 2nd floor",
        "facilities_admin",
    ),
    (
        "Request for team seating rearrangement and desk relocation next week",
        "facilities_admin",
    ),
]


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    """Generate normalized character n-grams for Jaccard similarity."""
    import re

    clean = re.sub(r"\s+", " ", text.lower().strip())
    if len(clean) < n:
        return {clean}
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def deduplicate_dataset(
    pairs: list[tuple[str, str]], threshold: float = 0.85
) -> list[tuple[str, str]]:
    """Remove exact duplicates and high-similarity near-duplicates from dataset."""
    import re

    unique_pairs: list[tuple[str, str]] = []
    seen_exact: set[str] = set()
    seen_ngrams: list[set[str]] = []

    for text, label in pairs:
        norm_key = re.sub(r"[^\w\s]", "", text.lower()).strip()
        if not norm_key or norm_key in seen_exact:
            continue

        ng = _char_ngrams(norm_key)
        is_near_dup = False
        for existing_ng in seen_ngrams:
            intersection = len(ng & existing_ng)
            union = len(ng | existing_ng)
            if union > 0 and (intersection / union) >= threshold:
                is_near_dup = True
                break

        if not is_near_dup:
            seen_exact.add(norm_key)
            seen_ngrams.append(ng)
            unique_pairs.append((text, label))

    return unique_pairs
