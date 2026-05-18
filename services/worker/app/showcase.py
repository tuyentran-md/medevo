from __future__ import annotations

from app.models import ShowcaseRecord


SHOWCASES: list[ShowcaseRecord] = [
    ShowcaseRecord(
        id="showcase-sepsis-supportive-care",
        title="Illustrative Pediatric Sepsis Supportive Care",
        description=(
            "Demo-only guideline excerpt focused on escalation, antibiotics, and monitoring."
        ),
        input_mode="guideline",
        input_text=(
            "Children with suspected sepsis should receive cultures before antibiotics when this "
            "does not delay treatment. Broad-spectrum antibiotics should be started within one hour "
            "for septic shock and within three hours for stable sepsis. Reassess lactate, urine output, "
            "and perfusion trends to guide escalation. De-escalate antimicrobial coverage once culture "
            "data clarify the likely pathogen. Escalate to ICU support if shock persists despite fluid "
            "and vasoactive therapy."
        ),
        tags=["showcase", "guideline", "pediatrics"],
    ),
    ShowcaseRecord(
        id="showcase-bronchiolitis-oxygen",
        title="Illustrative Bronchiolitis Oxygen Strategy",
        description=(
            "Demo-only excerpt for supportive respiratory decision-making under evidence drift."
        ),
        input_mode="guideline",
        input_text=(
            "Infants with bronchiolitis should receive supportive care with hydration and nasal suction. "
            "Supplemental oxygen is recommended when saturation persistently falls below the target range. "
            "Routine bronchodilators should not be continued without clear observed benefit. High-flow nasal "
            "oxygen may be considered when standard low-flow support fails. Chest radiographs should be reserved "
            "for atypical cases or suspected complications."
        ),
        tags=["showcase", "respiratory", "pediatrics"],
    ),
    ShowcaseRecord(
        id="showcase-antibiotic-paper",
        title="Illustrative Antibiotic Stewardship Paper Conclusion",
        description=(
            "Demo-only paper-style conclusion converted into guideline-equivalent claims."
        ),
        input_mode="paper",
        input_text=(
            "Conclusion: In hospitalized children with complicated intra-abdominal infection, protocolized "
            "early source control combined with short-course targeted antibiotics reduced overall exposure "
            "without increasing treatment failure. The evidence supports narrowing therapy once culture data "
            "are available and avoiding automatic prolonged intravenous courses."
        ),
        tags=["showcase", "paper", "stewardship"],
    ),
]


def get_showcase(showcase_id: str) -> ShowcaseRecord | None:
    for showcase in SHOWCASES:
        if showcase.id == showcase_id:
            return showcase
    return None
