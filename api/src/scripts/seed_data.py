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
            "Employee onboarding, welcome kits, onboarding parcels, courier delivery damages, "
            "offboarding, profile updates, ID card, leave requests, "
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
        added = 0
        updated = 0
        for staff in SUPPORT_STAFF:
            email = staff["email"]
            user = (
                db.query(User)
                .filter(
                    (User.department == staff["department"])
                    | (User.name == staff["name"])
                    | (User.email == email)
                )
                .first()
            )
            if user:
                changed = False
                if user.email != email:
                    user.email = email
                    changed = True
                if user.expertise != staff["expertise"]:
                    user.expertise = staff["expertise"]
                    changed = True
                if user.department != staff["department"]:
                    user.department = staff["department"]
                    changed = True
                if user.name != staff["name"]:
                    user.name = staff["name"]
                    changed = True

                if changed:
                    updated += 1
                    print(f"  * Updated: {staff['name']} <{email}>")
                else:
                    print(f"  - Unchanged: {staff['name']} <{email}>")
            else:
                db.add(User(**staff))
                added += 1
                print(f"  + Added: {staff['name']} <{email}>")
        db.commit()
        print(f"\nSeeding complete. {added} added, {updated} updated.")
    finally:
        db.close()


if __name__ == "__main__":
    print("Seeding support staff...\n")
    seed()
