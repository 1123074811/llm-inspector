"""
runner/dynamic_kg_questions.py — v16 Phase 6: Dynamic Knowledge Graph Question Generation

Generates verification questions at run-time from Wikidata triples,
avoiding question leakage and contamination.

Reference:
    Pellissier Tanon et al. (2020) "YAGO 4: A Reason-able Knowledge Base" ESWC
    Wikidata SPARQL: https://query.wikidata.org/sparql
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


# High-confidence Wikidata properties for question generation
# Each entry: (property_id, english_label, question_template, answer_type)
_VERIFIED_PROPERTIES = [
    ("P569", "date of birth", "When was {entity} born?", "date"),
    ("P570", "date of death", "When did {entity} die?", "date"),
    ("P19", "place of birth", "Where was {entity} born?", "entity"),
    ("P20", "place of death", "Where did {entity} die?", "entity"),
    ("P27", "country of citizenship", "What country is {entity} a citizen of?", "entity"),
    ("P106", "occupation", "What is {entity}'s occupation?", "entity"),
    ("P108", "employer", "Who employs {entity}?", "entity"),
    ("P166", "award received", "What award did {entity} receive?", "entity"),
    ("P69", "educated at", "Where was {entity} educated?", "entity"),
    ("P54", "member of sports team", "What sports team does {entity} belong to?", "entity"),
    ("P101", "field of work", "What field does {entity} work in?", "entity"),
    ("P735", "given name", "What is {entity}'s given name?", "text"),
    ("P734", "family name", "What is {entity}'s family name?", "text"),
    ("P1412", "languages spoken", "What language does {entity} speak?", "entity"),
    ("P131", "located in", "Where is {entity} located?", "entity"),
    ("P6", "head of government", "Who is the head of government of {entity}?", "entity"),
    ("P150", "contains", "What does {entity} contain?", "entity"),
    ("P36", "capital", "What is the capital of {entity}?", "entity"),
    ("P571", "inception", "When was {entity} founded/established?", "date"),
    ("P112", "founded by", "Who founded {entity}?", "entity"),
]

# Verified entities for question generation (high-confidence Wikidata items)
_SEED_ENTITIES = [
    {"qid": "Q42", "label": "Douglas Adams"},
    {"qid": "Q937", "label": "Albert Einstein"},
    {"qid": "Q7186", "label": "Marie Curie"},
    {"qid": "Q8023", "label": "Nelson Mandela"},
    {"qid": "Q9682", "label": "Elisabeth II"},
    {"qid": "Q5582", "label": "Martin Luther King Jr."},
    {"qid": "Q5879", "label": "Johann Sebastian Bach"},
    {"qid": "Q9259", "label": "Nikola Tesla"},
    {"qid": "Q12489", "label": "Rome"},
    {"qid": "Q84", "label": "London"},
    {"qid": "Q60", "label": "New York City"},
    {"qid": "Q148", "label": "People's Republic of China"},
    {"qid": "Q30", "label": "United States"},
    {"qid": "Q142", "label": "France"},
    {"qid": "Q183", "label": "Germany"},
    {"qid": "Q17", "label": "Japan"},
    {"qid": "Q408", "label": "Australia"},
    {"qid": "Q668", "label": "India"},
    {"qid": "Q159", "label": "Russia"},
    {"qid": "Q145", "label": "United Kingdom"},
]


@dataclass
class DynamicKGQuestion:
    """A dynamically generated knowledge graph verification question."""
    question_id: str
    question_text: str
    expected_answer: str
    answer_type: str       # "date", "entity", "text"
    entity_qid: str        # Wikidata Q-number
    entity_label: str
    property_id: str       # Wikidata P-number
    property_label: str
    confidence: float = 1.0
    source: str = "wikidata_dynamic"

    def to_test_case(self) -> dict:
        """Convert to suite-compatible test case dict."""
        return {
            "id": self.question_id,
            "category": "knowledge",
            "dimension": "knowledge",
            "name": f"KG_{self.entity_label}_{self.property_label}",
            "user_prompt": self.question_text,
            "judge_method": "semantic",
            "max_tokens": 64,
            "n_samples": 1,
            "temperature": 0.0,
            "params": {
                "keywords": [self.expected_answer],
                "answer_type": self.answer_type,
            },
            "weight": 0.5,  # Lower weight for dynamic questions
            "irt_a": 0.6,
            "irt_b": -0.5,
            "irt_c": 0.0,
            "source_ref": f"wikidata:{self.entity_qid}/{self.property_id}",
            "license": "CC0",
            "difficulty": 0.3,
            "calibrated": False,
            "dynamic": True,
        }


def generate_kg_question(
    entity_qid: str,
    prop_id: str,
    entity_label: str = "",
    prop_label: str = "",
    verified_answer: str = "",
) -> DynamicKGQuestion | None:
    """
    v16 Phase 6: Generate a verification question from a Wikidata triple.

    e.g. (Q42, P569) → "When was Douglas Adams born?" with verified answer "1952-03-11"

    Args:
        entity_qid: Wikidata entity Q-number (e.g. "Q42").
        prop_id: Wikidata property P-number (e.g. "P569").
        entity_label: Human-readable entity name.
        prop_label: Human-readable property name.
        verified_answer: The verified answer string.

    Returns:
        DynamicKGQuestion or None if generation fails.
    """
    # Find question template
    template = None
    for pid, plabel, tmpl, atype in _VERIFIED_PROPERTIES:
        if pid == prop_id:
            template = tmpl
            prop_label = prop_label or plabel
            answer_type = atype
            break

    if template is None:
        logger.warning("No template for property", prop_id=prop_id)
        return None

    if not entity_label:
        entity_label = entity_qid

    question_text = template.format(entity=entity_label)
    question_id = f"kg_{entity_qid}_{prop_id}"

    return DynamicKGQuestion(
        question_id=question_id,
        question_text=question_text,
        expected_answer=verified_answer,
        answer_type=answer_type if 'answer_type' in dir() else "text",
        entity_qid=entity_qid,
        entity_label=entity_label,
        property_id=prop_id,
        property_label=prop_label,
    )


def generate_random_questions(
    n: int = 5,
    seed: int | None = None,
) -> list[DynamicKGQuestion]:
    """
    v16 Phase 6: Generate n random KG verification questions.

    Used at run startup to create dynamic, non-leaked verification questions.

    Args:
        n: Number of questions to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of DynamicKGQuestion objects.
    """
    if seed is not None:
        random.seed(seed)

    questions: list[DynamicKGQuestion] = []
    attempts = 0
    max_attempts = n * 5

    while len(questions) < n and attempts < max_attempts:
        attempts += 1
        entity = random.choice(_SEED_ENTITIES)
        prop = random.choice(_VERIFIED_PROPERTIES)

        q = generate_kg_question(
            entity_qid=entity["qid"],
            prop_id=prop[0],
            entity_label=entity["label"],
            prop_label=prop[1],
            verified_answer="",  # Answer filled at runtime by KG query
        )
        if q is not None:
            questions.append(q)

    logger.info(
        "Generated dynamic KG questions",
        requested=n,
        generated=len(questions),
    )
    return questions


def verify_dual_source(
    entity_name: str,
    wikidata_result: Any | None = None,
    dbpedia_result: Any | None = None,
) -> tuple[bool, bool]:
    """
    v16 Phase 6: Cross-verify between Wikidata and DBpedia.

    Args:
        entity_name: Entity to verify.
        wikidata_result: VerificationResult from WikidataClient.
        dbpedia_result: VerificationResult from DBpedia client.

    Returns:
        (is_consistent, kg_conflict) tuple.
    """
    if wikidata_result is None and dbpedia_result is None:
        return False, False

    if wikidata_result is None or dbpedia_result is None:
        # Only one source available — no conflict, but lower confidence
        return True, False

    # Both sources available — check for conflict
    wd_verified = getattr(wikidata_result, 'is_verified', False)
    db_verified = getattr(dbpedia_result, 'is_verified', False)

    if wd_verified and db_verified:
        return True, False  # Both agree: verified
    elif not wd_verified and not db_verified:
        return True, False  # Both agree: not verified
    else:
        return False, True  # Conflict: one says verified, other doesn't
