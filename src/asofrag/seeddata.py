"""The synthetic handbook: 60 clauses in 8 sections, and the changes injected across editions.

Each ``Spec`` is one clause. ``{v}`` in the text is filled from ``vals``: an amended clause has
two values and switches to the second on ``change``. The wording and length of an amended clause
are otherwise identical, which is what makes it hard for a similarity index to tell versions apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

V1, V2, V3, V4, V5 = (
    date(2024, 1, 1),
    date(2024, 7, 1),
    date(2025, 1, 1),
    date(2025, 6, 1),
    date(2026, 1, 1),
)
EDITIONS = (V1, V2, V3, V4, V5)
HORIZON_END = date(2026, 6, 30)  # last as-of date used by the question set

SECTIONS = {
    "conduct": "Conduct",
    "holidays": "Holidays",
    "leave": "Leave",
    "expenses": "Expenses",
    "remote": "Remote work",
    "travel": "Travel",
    "overtime": "Overtime",
    "equipment": "Equipment",
}


@dataclass(frozen=True)
class Spec:
    clause_id: str
    section: str
    title: str
    text: str
    vals: tuple[str, ...] = ("",)
    key: str = ""
    questions: tuple[str, ...] = ()
    change: date | None = None  # amended: switches from vals[0] to vals[1] on this edition
    intro: date = V1
    repeal: date | None = None  # absent from the edition dated here onwards
    move: tuple[date, str] | None = None  # moves to another section on this edition
    pin: date | None = None  # fixed as-of date for the first before-change question

    def value_at(self, when: date) -> str:
        return self.vals[1 if self.change and when >= self.change else 0]

    def exists_at(self, when: date) -> bool:
        return self.intro <= when and (self.repeal is None or when < self.repeal)

    def section_at(self, when: date) -> str:
        return self.move[1] if self.move and when >= self.move[0] else self.section

    def render(self, when: date) -> str:
        return self.text.format(v=self.value_at(when))

    def gold_value(self, when: date) -> str:
        return self.key.format(v=self.value_at(when))


def S(clause_id, section, title, text, vals=("",), key="", qs=(), **kw) -> Spec:
    return Spec(clause_id, section, title, text, tuple(vals), key, tuple(qs), **kw)


CATALOG: tuple[Spec, ...] = (
    # ---- Conduct -------------------------------------------------------------------------
    S(
        "CONDUCT-1.1",
        "conduct",
        "Vendor gifts",
        "Employees may accept a gift from a vendor worth up to ${v} in a calendar year. Anything "
        "above that value must be declined or reported to the compliance team.",
        ("50", "75"),
        "${v}",
        change=V2,
        qs=(
            "What is the most a vendor gift can be worth in a calendar year before it must be "
            "declined?",
            "Up to what value may an employee accept a gift from a vendor?",
        ),
    ),
    S(
        "CONDUCT-1.2",
        "conduct",
        "Outside business interests",
        "Employees must disclose an outside business interest within {v} days of taking it on. The "
        "compliance team reviews each disclosure and replies in writing.",
        ("10",),
        "{v} days",
        qs=("Within how many days must an outside business interest be disclosed?",),
    ),
    S(
        "CONDUCT-1.3",
        "conduct",
        "Harassment complaints",
        "Human resources acknowledges a harassment complaint within {v} business days. An "
        "investigator with no reporting line to either party is assigned to the case.",
        ("2",),
        "{v} business days",
        qs=("How soon does human resources acknowledge a harassment complaint?",),
    ),
    S(
        "CONDUCT-1.4",
        "conduct",
        "Social media",
        "Staff must not post confidential client names on social media accounts. Personal accounts "
        "may state the employer but must not speak for it.",
    ),
    S(
        "CONDUCT-1.5",
        "conduct",
        "Privacy training",
        "Every new hire completes data privacy training within {v} days of joining. Refresher "
        "courses are assigned each January.",
        ("30",),
    ),
    S(
        "CONDUCT-1.6",
        "conduct",
        "Smoking breaks",
        "Employees may take {v} smoking breaks of ten minutes at the designated outdoor shelter. "
        "Breaks beyond that allowance are unpaid.",
        ("two",),
        "{v}",
        repeal=V2,
        qs=(
            "How many smoking breaks may an employee take at the outdoor shelter?",
            "Is there a time allowance for smoking breaks at the designated shelter?",
        ),
    ),
    S(
        "CONDUCT-1.7",
        "conduct",
        "Retaliation",
        "Retaliation against a person who reports a concern in good faith is grounds for "
        "dismissal. Managers must pass every report to human resources the same day.",
    ),
    S(
        "CONDUCT-1.8",
        "conduct",
        "Relatives",
        "Relatives may not work in the same reporting line. Existing cases are reviewed by the "
        "compliance team.",
    ),
    S(
        "CONDUCT-1.9",
        "conduct",
        "Generative AI tools",
        "Employees must log every external generative AI tool they use with the security team "
        "within {v} days of first use. Customer data must never be entered into such tools.",
        ("14",),
        "{v} days",
        intro=V5,
        qs=(
            "Within how many days must an external generative AI tool be logged with the security "
            "team?",
            "How long after first use must an external generative AI tool be logged?",
        ),
    ),
    # ---- Holidays ------------------------------------------------------------------------
    S(
        "HOLIDAY-2.1",
        "holidays",
        "Public holidays",
        "The company observes {v} paid public holidays each calendar year, listed in the annual "
        "calendar. A holiday that falls on a weekend moves to the nearest weekday.",
        ("11",),
        "{v} paid public holidays",
        qs=("How many paid public holidays does the company observe each year?",),
    ),
    S(
        "HOLIDAY-2.2",
        "holidays",
        "Floating holidays",
        "Employees choose {v} floating holidays per year to use on days of personal significance. "
        "Unused floating holidays lapse on 31 December.",
        ("2", "3"),
        "{v} floating holidays",
        change=V3,
        qs=(
            "How many floating holidays can an employee pick for days of personal significance?",
            "What is the yearly allowance of floating holidays?",
        ),
    ),
    S(
        "HOLIDAY-2.3",
        "holidays",
        "Holiday pay",
        "Work on a public holiday is paid at {v} times the base rate. The pay applies to the hours "
        "worked on the day only.",
        ("2",),
        "{v} times",
        qs=("At what multiple of the base rate is work on a public holiday paid?",),
    ),
    S(
        "HOLIDAY-2.4",
        "holidays",
        "Holiday cover",
        "Teams publish their holiday coverage rota {v} weeks before each public holiday. Rota "
        "swaps are agreed between colleagues.",
        ("3",),
    ),
    S(
        "HOLIDAY-2.5",
        "holidays",
        "Year-end closure",
        "The offices close between 25 December and 1 January. Staff who must work in that period "
        "receive an extra day off in lieu.",
    ),
    S(
        "HOLIDAY-2.6",
        "holidays",
        "Anniversary half day",
        "Employees receive a half day off on their work anniversary. The half day must be taken "
        "within the same month.",
        repeal=V5,
        qs=("Do employees get time off on their work anniversary?",),
    ),
    S(
        "HOLIDAY-2.7",
        "holidays",
        "Early closing",
        "Offices close at 3 pm on the last working day before a public holiday. Customer support "
        "continues on a reduced rota.",
    ),
    # ---- Leave ---------------------------------------------------------------------------
    S(
        "LEAVE-3.1",
        "leave",
        "Vacation accrual",
        "Full-time employees accrue {v} days of paid vacation per year. Accrual is prorated for "
        "part-time staff.",
        ("20",),
        "{v} days",
        qs=("How many days of paid vacation do full-time employees accrue per year?",),
    ),
    S(
        "LEAVE-3.2",
        "leave",
        "Parental leave",
        "Employees receive {v} weeks of paid parental leave after the birth or adoption of a "
        "child. The leave may be taken in one block or in two blocks within the first year.",
        ("12", "16"),
        "{v} weeks",
        change=V4,
        pin=date(2025, 3, 12),
        qs=(
            "How many weeks of paid parental leave follow the birth or adoption of a child?",
            "What is the duration of paid parental leave for a new parent?",
        ),
    ),
    S(
        "LEAVE-3.3",
        "leave",
        "Sick leave",
        "Employees receive {v} paid sick days per year. A medical certificate is required after "
        "three consecutive days.",
        ("10",),
    ),
    S(
        "LEAVE-3.4",
        "leave",
        "Bereavement leave",
        "Employees receive {v} days of paid bereavement leave on the death of an immediate family "
        "member. Additional unpaid time may be agreed with the manager.",
        ("3", "5"),
        "{v} days",
        change=V3,
        qs=(
            "How many days of paid bereavement leave follow the death of an immediate family "
            "member?",
            "How many paid days does bereavement leave give after the death of an immediate "
            "relative?",
        ),
    ),
    S(
        "LEAVE-3.5",
        "leave",
        "Jury duty",
        "Employees called for jury duty keep full pay for up to {v} days of jury service. The "
        "court summons must be shown to human resources.",
        ("10",),
        "{v} days",
        qs=("For how many days of jury service do employees keep full pay?",),
    ),
    S(
        "LEAVE-3.6",
        "leave",
        "Carry-over cap",
        "Employees may carry over up to {v} unused vacation days into the next year. Days above "
        "the cap are forfeited on 31 March.",
        ("5", "8"),
        "{v} unused vacation days",
        change=V2,
        qs=(
            "Up to how many unused vacation days can an employee carry over into the next year?",
            "What is the cap on vacation days an employee can carry over?",
        ),
    ),
    S(
        "LEAVE-3.7",
        "leave",
        "Sabbatical",
        "Employees with seven years of service may apply for an unpaid sabbatical of up to {v} "
        "months. The role is held open for the whole period.",
        ("3",),
    ),
    S(
        "LEAVE-3.8",
        "leave",
        "Military leave",
        "Reservists called to active duty are granted unpaid leave and keep their seniority. "
        "Benefits continue for {v} months.",
        ("6",),
    ),
    S(
        "LEAVE-3.9",
        "leave",
        "Volunteer time",
        "Employees may spend {v} paid hours a year volunteering with a registered charity. The "
        "charity must be approved by the community team.",
        ("16",),
        "{v} paid hours",
        intro=V2,
        qs=("May an employee volunteer for a registered charity on company time?",),
    ),
    # ---- Expenses ------------------------------------------------------------------------
    S(
        "EXPENSE-4.1",
        "expenses",
        "Meal per diem",
        "The daily meal per diem on overnight business trips is ${v}. The per diem is paid "
        "without receipts.",
        ("55", "65"),
        "${v}",
        change=V3,
        qs=(
            "What is the daily meal per diem on an overnight business trip?",
            "How much per diem is paid for meals each day away on business?",
        ),
    ),
    S(
        "EXPENSE-4.2",
        "expenses",
        "Receipt threshold",
        "A receipt is required for any single expense above ${v}. Lost receipts need a signed "
        "statement from the claimant.",
        ("25", "30"),
        "${v}",
        change=V4,
        qs=(
            "Above what amount is a receipt required for a single expense?",
            "From what dollar amount does a single expense need a receipt?",
        ),
    ),
    S(
        "EXPENSE-4.3",
        "expenses",
        "Submission deadline",
        "Expense reports must be submitted within {v} days of the end of the trip. Late reports "
        "are paid in the following month.",
        ("30",),
        "{v} days",
        qs=("Within how many days of a trip must an expense report be submitted?",),
    ),
    S(
        "EXPENSE-4.4",
        "expenses",
        "Client dinners",
        "Dinner with a client is reimbursed up to ${v} per person. The most senior attendee pays "
        "and submits the claim.",
        ("80", "95"),
        "${v}",
        change=V5,
        qs=(
            "What is the reimbursement limit per person for a dinner with a client?",
            "Up to how much is a client dinner reimbursed for each person?",
        ),
    ),
    S(
        "EXPENSE-4.5",
        "expenses",
        "Corporate cards",
        "Corporate cards carry a monthly limit of ${v}. Personal purchases on the card are repaid "
        "within five days.",
        ("5000",),
    ),
    S(
        "EXPENSE-4.6",
        "expenses",
        "Airfare class",
        "Flights shorter than six hours are booked in economy class. Longer flights may be booked "
        "in premium economy.",
    ),
    S(
        "EXPENSE-4.7",
        "expenses",
        "Trade press",
        "The company reimburses up to ${v} a month for trade press subscriptions. Claims are "
        "submitted with the monthly invoice.",
        ("30",),
        "${v}",
        repeal=V4,
        qs=("Does the company pay for trade press subscriptions?",),
    ),
    S(
        "EXPENSE-4.8",
        "expenses",
        "Tips",
        "Tips of up to {v} percent of a restaurant bill are reimbursed. Tips above that need "
        "approval from a manager.",
        ("15",),
    ),
    S(
        "EXPENSE-4.9",
        "expenses",
        "Wellness",
        "The company reimburses up to ${v} a month for a gym or fitness class membership. Claims "
        "need the monthly receipt.",
        ("40",),
        "${v}",
        intro=V3,
        qs=("Does the company pay for a gym or fitness class membership?",),
    ),
    # ---- Remote work ---------------------------------------------------------------------
    S(
        "REMOTE-5.1",
        "remote",
        "Remote days",
        "Employees may work remotely up to {v} days per week. The remaining days are spent in the "
        "office with the team.",
        ("2", "3"),
        "{v} days per week",
        change=V2,
        qs=(
            "How many days per week may an employee work remotely?",
            "What is the weekly limit on remote working days?",
        ),
    ),
    S(
        "REMOTE-5.2",
        "remote",
        "Internet stipend",
        "Remote employees receive a home internet stipend of ${v} per month. The stipend is added "
        "to the monthly payroll.",
        ("40", "50"),
        "${v}",
        change=V5,
        qs=(
            "What is the monthly home internet stipend for remote employees?",
            "How much does the company pay toward home internet each month?",
        ),
    ),
    S(
        "REMOTE-5.3",
        "remote",
        "Core hours",
        "Remote employees must be reachable during core hours from {v} to 15:00 local time. "
        "Meetings are not scheduled outside core hours.",
        ("10:00",),
    ),
    S(
        "REMOTE-5.4",
        "remote",
        "VPN",
        "Work on company systems from home requires the VPN to be connected. Access from a "
        "personal device needs the approval of the security team.",
    ),
    S(
        "REMOTE-5.5",
        "remote",
        "Chair allowance",
        "A one-time ergonomic chair allowance of ${v} is available to remote employees. The chair "
        "stays with the employee.",
        ("250",),
        "${v}",
        move=(V4, "equipment"),
        qs=("How much is the one-time ergonomic chair allowance?",),
    ),
    S(
        "REMOTE-5.6",
        "remote",
        "Working abroad",
        "Remote work from another country is limited to {v} days a year. Longer periods need "
        "approval from the tax and legal teams.",
        ("30",),
        "{v} days",
        qs=("What is the yearly limit on working from another country?",),
    ),
    S(
        "REMOTE-5.7",
        "remote",
        "Workspace safety",
        "Remote employees complete a home workspace safety checklist at onboarding. The checklist "
        "is renewed every two years.",
    ),
    S(
        "REMOTE-5.8",
        "remote",
        "Coworking space",
        "Employees without room at home may claim up to ${v} a month for a coworking space. "
        "Claims need the space contract.",
        ("120",),
        "${v}",
        intro=V4,
        qs=("Does the company pay for a desk in a shared coworking space?",),
    ),
    # ---- Travel --------------------------------------------------------------------------
    S(
        "TRAVEL-6.1",
        "travel",
        "Hotel cap",
        "Hotel stays are reimbursed up to ${v} per night in standard cities. Higher caps apply in "
        "cities on the premium list.",
        ("180", "210"),
        "${v}",
        change=V3,
        qs=(
            "What is the nightly hotel limit in a standard city?",
            "How much is reimbursed for a hotel night in a standard city?",
        ),
    ),
    S(
        "TRAVEL-6.2",
        "travel",
        "Mileage",
        "Business mileage in a personal car is reimbursed at ${v} per mile. Commuting miles are "
        "not reimbursed.",
        ("0.67", "0.70"),
        "${v}",
        change=V2,
        qs=(
            "What rate per mile is paid for business mileage in a personal car?",
            "How much is reimbursed per mile when an employee drives a personal car on business?",
        ),
    ),
    S(
        "TRAVEL-6.3",
        "travel",
        "Booking lead time",
        "Flights are booked at least {v} days before departure. Exceptions need the approval of "
        "the department head.",
        ("14",),
        "{v} days",
        qs=("How many days before departure must flights be booked?",),
    ),
    S(
        "TRAVEL-6.4",
        "travel",
        "Rental cars",
        "Rental cars are booked in the compact or mid-size class. Insurance is purchased through "
        "the corporate rental agreement.",
    ),
    S(
        "TRAVEL-6.5",
        "travel",
        "Airport parking",
        "Airport parking is reimbursed at the long-stay rate of up to ${v} per day. Valet parking "
        "is not reimbursed.",
        ("22",),
        "${v}",
        move=(V3, "expenses"),
        qs=("What is the daily limit for reimbursing long-stay airport parking?",),
    ),
    S(
        "TRAVEL-6.6",
        "travel",
        "International travel",
        "Travel outside the home country needs written approval {v} weeks in advance. Approval "
        "covers the itinerary and the visa plan.",
        ("4",),
        "{v} weeks",
        qs=("How many weeks in advance does international travel need written approval?",),
    ),
    S(
        "TRAVEL-6.7",
        "travel",
        "Travel insurance",
        "Travel insurance is provided automatically for every approved business trip. Claims are "
        "filed with the insurer within {v} days.",
        ("60",),
    ),
    S(
        "TRAVEL-6.8",
        "travel",
        "Loyalty points",
        "Loyalty points earned on business travel belong to the traveller. They may not be used "
        "to upgrade a company-paid ticket.",
    ),
    # ---- Overtime ------------------------------------------------------------------------
    S(
        "OVERTIME-7.1",
        "overtime",
        "Overtime premium",
        "Overtime hours are paid at {v} times the regular hourly rate. Overtime applies to "
        "non-exempt employees only.",
        ("1.50", "1.75"),
        "{v} times",
        change=V4,
        qs=(
            "What multiple of the regular hourly rate is paid for overtime hours?",
            "How are overtime hours paid compared with the normal hourly rate?",
        ),
    ),
    S(
        "OVERTIME-7.2",
        "overtime",
        "Overtime approval",
        "Overtime must be approved by a manager before it is worked. Unapproved hours are paid "
        "but may lead to a warning.",
    ),
    S(
        "OVERTIME-7.3",
        "overtime",
        "On-call stipend",
        "Employees on the on-call rota receive a stipend of ${v} for each week on call. Call-outs "
        "are paid separately as overtime.",
        ("150",),
        "${v}",
        qs=("How much is the weekly on-call stipend?",),
    ),
    S(
        "OVERTIME-7.4",
        "overtime",
        "Compensatory time",
        "Salaried employees may bank up to {v} hours of compensatory time. Banked hours expire "
        "after six months.",
        ("24",),
    ),
    S(
        "OVERTIME-7.5",
        "overtime",
        "Overtime meals",
        "A meal allowance of ${v} is paid when overtime runs past eight in the evening. The "
        "allowance is claimed through the expense system.",
        ("20",),
        "${v}",
        qs=("How much is the meal allowance when overtime runs late?",),
    ),
    S(
        "OVERTIME-7.6",
        "overtime",
        "Weekend shifts",
        "Weekend shifts carry a premium of ${v} per shift for hourly employees. The premium is "
        "separate from overtime.",
        ("60",),
    ),
    S(
        "OVERTIME-7.7",
        "overtime",
        "Rest periods",
        "Employees must have at least {v} hours of rest between two shifts. A shorter rest needs "
        "the written agreement of the employee.",
        ("11",),
    ),
    # ---- Equipment -----------------------------------------------------------------------
    S(
        "EQUIP-8.1",
        "equipment",
        "Laptop refresh",
        "Laptops are replaced every {v} years. Earlier replacement is granted when a device fails "
        "and cannot be repaired.",
        ("3", "4"),
        "{v} years",
        change=V5,
        qs=(
            "How often are laptops replaced?",
            "After how many years does an employee get a new laptop?",
        ),
    ),
    S(
        "EQUIP-8.2",
        "equipment",
        "Monitors",
        "Each employee may order one external monitor costing up to ${v}. Additional monitors "
        "need a business case.",
        ("300",),
    ),
    S(
        "EQUIP-8.3",
        "equipment",
        "Return of equipment",
        "Equipment must be returned within {v} days of an employee's last working day. Unreturned "
        "equipment is deducted from final pay.",
        ("7",),
        "{v} days",
        qs=("Within how many days of the last working day must equipment be returned?",),
    ),
    S(
        "EQUIP-8.4",
        "equipment",
        "Lost devices",
        "A lost or stolen device must be reported to the security team within {v} hours. The "
        "device is remotely wiped on receipt of the report.",
        ("24",),
        "{v} hours",
        qs=("Within how many hours must a lost or stolen device be reported?",),
    ),
    S(
        "EQUIP-8.5",
        "equipment",
        "Phone stipend",
        "A monthly stipend of ${v} covers the cost of a personal mobile phone used for work. The "
        "stipend is paid by direct deposit.",
        ("35",),
        "${v}",
        repeal=V3,
        qs=(
            "Is there a stipend for a personal mobile phone?",
            "What does the company pay toward a personal mobile phone used for work?",
        ),
    ),
    S(
        "EQUIP-8.6",
        "equipment",
        "Headsets",
        "A headset is provided to every employee who takes video calls. Replacement headsets are "
        "issued once a year.",
    ),
    S(
        "EQUIP-8.7",
        "equipment",
        "Software licences",
        "Software licences are requested through the IT service desk. Licences unused for {v} "
        "days are reclaimed.",
        ("90",),
    ),
    S(
        "EQUIP-8.8",
        "equipment",
        "Security keys",
        "Every employee is issued a hardware security key for multi-factor sign-in. A lost key is "
        "replaced within {v} working days.",
        ("3",),
        "{v} working days",
        intro=V4,
        qs=("Is every employee issued a hardware security key for multi-factor sign-in?",),
    ),
)

# How the 60 questions are drawn from the catalog ---------------------------------------------
BEFORE_TWICE = (  # amended clauses asked twice before the change
    "LEAVE-3.2", "LEAVE-3.4", "EXPENSE-4.1", "EXPENSE-4.2", "EXPENSE-4.4",
    "REMOTE-5.2", "TRAVEL-6.1", "OVERTIME-7.1", "HOLIDAY-2.2", "EQUIP-8.1",
)  # fmt: skip
BEFORE_AND_AFTER = (  # asked once before and twice after the change
    "CONDUCT-1.1", "LEAVE-3.6", "REMOTE-5.1", "TRAVEL-6.2",
)  # fmt: skip
UNCHANGED_ASKED = (
    "CONDUCT-1.2", "CONDUCT-1.3", "HOLIDAY-2.1", "HOLIDAY-2.3", "LEAVE-3.1", "LEAVE-3.5",
    "EXPENSE-4.3", "REMOTE-5.5", "REMOTE-5.6", "TRAVEL-6.3", "TRAVEL-6.5", "TRAVEL-6.6",
    "OVERTIME-7.3", "OVERTIME-7.5", "EQUIP-8.3", "EQUIP-8.4",
)  # fmt: skip
# clause id -> number of questions dated after its repeal (phrasings are reused in order)
REPEALED_ASKED = {"CONDUCT-1.6": 2, "EQUIP-8.5": 2, "EXPENSE-4.7": 1, "HOLIDAY-2.6": 1}
# clause id -> number of questions dated before its introduction
INTRODUCED_ASKED = {
    "CONDUCT-1.9": 2, "LEAVE-3.9": 1, "EXPENSE-4.9": 1, "REMOTE-5.8": 1, "EQUIP-8.8": 1,
}  # fmt: skip

FABRICATED_QUOTES = (
    "Unused entitlements roll over indefinitely and never expire.",
    "Claims are settled within one business day of approval.",
    "The limit is waived for employees with more than five years of service.",
)
