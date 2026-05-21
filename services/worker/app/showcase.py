from __future__ import annotations

from app.models import ShowcaseRecord


SHOWCASES: list[ShowcaseRecord] = [
    ShowcaseRecord(
        id="hrt-chronic-disease-prevention",
        title="Illustrative Hormone Therapy for Chronic-Disease Prevention",
        description=(
            "Demo-only guideline excerpt tracking how recommendations on menopausal "
            "hormone therapy for chronic-disease prevention drift across eras when the "
            "evidence base is left ungated, versus held by the provenance gate."
        ),
        input_mode="guideline",
        input_text=(
            "Menopausal hormone therapy with estrogen should be offered to postmenopausal "
            "women for the primary prevention of coronary heart disease. Hormone replacement "
            "therapy reduces the long-term risk of cardiovascular events in menopausal women. "
            "Estrogen plus progestin should be considered to prevent chronic disease and "
            "all-cause mortality after menopause. Hormone therapy lowers the incidence of "
            "osteoporotic fracture and should be continued for chronic-disease prevention. "
            "Routine use of menopausal hormone therapy for chronic-disease prevention should "
            "be reassessed against observed cardiovascular and breast-cancer outcomes."
        ),
        tags=["showcase", "guideline", "hormone-therapy"],
    ),
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
