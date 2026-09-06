"""
Seed script to populate the support staff database with initial users and their expertise.
Run once with: python -m src.scripts.seed_data
"""

from ..models.database import SessionLocal, init_db
from ..models.models import User

SUPPORT_STAFF = [
    {
        "name": "Admin Team",
        "email": "admin@test.com",
        "department": "Finance & Procurement",
        "expertise": (
            "Purchase orders, invoice approvals, vendor payments, expense reimbursements, "
            "budget queries, financial approvals, contract renewals, subscription billing, "
            "procurement requests, purchase requisitions, cost center approvals"
        ),
    },
    {
        "name": "IT Admin",
        "email": "it-admin@test.com",
        "department": "IT Support",
        "expertise": (
            "Hardware requests, laptop setup, software installation, VPN issues, "
            "network connectivity problems, password resets, device provisioning, "
            "email account setup, printer issues, software licenses, system access, "
            "monitor setup, peripheral devices, remote desktop, IT helpdesk"
        ),
    },
    {
        "name": "HR Team",
        "email": "hr@test.com",
        "department": "Human Resources",
        "expertise": (
            "Employee onboarding, offboarding, profile updates, ID card, leave requests, "
            "payroll queries, salary, benefits enrollment, medical insurance, training programs, "
            "policy clarifications, contract amendments, performance reviews, job transfers, "
            "resignation, termination, employee handbook, attendance"
        ),
    },
    {
        "name": "Security Team",
        "email": "security@test.com",
        "department": "Security & Compliance",
        "expertise": (
            "Data breach reports, unauthorized access, suspicious activity, account compromise, "
            "GDPR compliance, data privacy, phishing emails, malware, ransomware, "
            "access revocation, security audit, vulnerability reports, MFA issues, "
            "security policy, incident response, compliance requirements"
        ),
    },
    {
        "name": "Facilities Team",
        "email": "facilities@test.com",
        "department": "Facilities & Admin",
        "expertise": (
            "Office access cards, building entry, desk booking, hot desking, "
            "maintenance requests, office repairs, visitor management, parking passes, "
            "asset allocation, office supplies, meeting room booking, air conditioning, "
            "cleaning requests, furniture, relocation within office"
        ),
    },
]


def seed():
    init_db()
    db = SessionLocal()
    try:
        existing_emails = {u.email for u in db.query(User.email).all()}
        added = 0
        for staff in SUPPORT_STAFF:
            if staff["email"] not in existing_emails:
                db.add(User(**staff))
                added += 1
                print(f"  + Added: {staff['name']} <{staff['email']}>")
            else:
                print(f"  - Skipped (already exists): {staff['email']}")
        db.commit()
        print(f"\nSeeding complete. {added} new user(s) added.")
    finally:
        db.close()


if __name__ == "__main__":
    print("Seeding support staff...\n")
    seed()
