"""
config.py — Specialized agent registry.

This file describes WHICH agents exist, WHAT they can answer,
and WHICH slice of the single Pinecone index ("legal-rag") each one
queries, via a metadata filter.

Cases and civil codes are NOT split into separate Pinecone namespaces or
indexes — they all live in one index, and each agent's `pinecone_filter`
picks out its slice via `country` / `law` / `doc_type` metadata (see
ingestion.py, which is what actually writes those fields on every vector).

The supervisor (agents.py) reads this file to decide which agent
to route the user's question to, and to build each SpecializedAgent's
Pinecone query filter.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AgentDescription:
    agent_id: str                # unique identifier
    countries: List[str]         # jurisdictions covered (for the routing prompt)
    legal_areas: List[str]       # legal domains covered (for the routing prompt)
    content_types: List[str]     # "case" / "civil_code" (for the routing prompt)
    description: str             # text read by the supervisor for routing decisions
    pinecone_filter: Dict = field(default_factory=dict)
    #   Passed as-is to pine_index.query(filter=...). Built from the same
    #   vocabulary ingestion.py writes: law is "Divorce" or "Inheritance",
    #   and doc_type is "Legal Cases" or "Civil Codes" — doc_type is now
    #   purely structural (case-law vs statute); it no longer encodes the
    #   legal area itself, `law` is the only field that does that, for
    #   both doc_types alike.
    #
    #   The only agent left combining case-law and civil code in one query
    #   is Estonia Inheritance, which omits doc_type from the filter (so
    #   `law="Inheritance"` alone pulls in both doc_types) — its case-law
    #   corpus is too small (31 documents) to justify splitting further.


AGENT_REGISTRY: List[AgentDescription] = [

    # ── ITALY (4 agents: case-law and legislation split by legal area) ────

    AgentDescription(
        agent_id="italy_divorce_cases",
        countries=["Italy"],
        legal_areas=["Divorce"],
        content_types=["case"],
        description=(
     "Italian case law and judicial decisions concerning divorce and "
     "separation disputes. Select this agent when the question asks how "
     "Italian courts have interpreted or decided issues involving spousal "
     "maintenance and post-divorce support, fault-based separation, child "
     "custody and placement, parental responsibility, assignment of the "
     "family home, or division of marital assets. Covers decisions from the "
     "Court of Cassation, Courts of Appeal, and ordinary Tribunals. Use the "
     "Italian divorce legislation agent instead when the question concerns "
     "statutory rules or legislative provisions rather than judicial decisions."
        ),
        pinecone_filter={"country": "Italy", "law": "Divorce", "doc_type": "Legal Cases"},
    ),

    AgentDescription(
        agent_id="italy_divorce_law",
        countries=["Italy"],
        legal_areas=["Divorce"],
        content_types=["civil_code"],
        description=(
     "Italian statutes and legislative provisions governing marriage, "
     "separation, and divorce. Select this agent when the question asks what "
     "Italian law provides regarding divorce procedures, spousal maintenance, "
     "marital property regimes, parental responsibility, shared child custody "
     "and support, or assignment of the family home. Use the Italian divorce "
     "case-law agent instead when the question concerns how courts have "
     "interpreted or applied these rules in specific disputes."
        ),
        pinecone_filter={"country": "Italy", "law": "Divorce", "doc_type": "Civil Codes"},
    ),

    AgentDescription(
        agent_id="italy_inheritance_cases",
        countries=["Italy"],
        legal_areas=["Inheritance"],
        content_types=["case"],
        description=(
        "Italian case law and judicial decisions concerning inheritance and "
        "succession disputes. Select this agent when the question asks how "
        "Italian courts have interpreted or decided issues involving forced "
        "heirship and reserved shares, validity or interpretation of wills, "
        "intestate succession, acceptance or renunciation of inheritance, "
        "estate division, gifts affecting heirs' shares, or disputes among "
        "heirs. Covers decisions from the Court of Cassation, Courts of Appeal, "
        "and ordinary Tribunals. Use the Italian inheritance legislation agent "
        "instead when the question concerns statutory rules or Civil Code "
        "provisions rather than judicial decisions."
        ),
        pinecone_filter={"country": "Italy", "law": "Inheritance", "doc_type": "Legal Cases"},
    ),

    AgentDescription(
    agent_id="italy_inheritance_law",
    countries=["Italy"],
    legal_areas=["Inheritance"],
    content_types=["civil_code"],
    description=(
        "Italian statutes and Civil Code provisions governing inheritance "
        "and succession. Select this agent when the question asks what Italian "
        "law provides regarding intestate or testamentary succession, will "
        "requirements, forced heirship and reserved shares, classes and shares "
        "of heirs, acceptance or renunciation of inheritance, estate division, "
        "or gifts affecting heirs' rights. Use the Italian inheritance case-law "
        "agent instead when the question concerns how courts have interpreted "
        "or applied these rules in specific disputes."
      ),
      pinecone_filter={"country": "Italy","law": "Inheritance","doc_type": "Civil Codes",},
    ),

    # ── SLOVENIA (4 agents: case-law and legislation split by legal area) ─
    # Volume justifies the same split as Italy: 101 divorce cases / 100
    # inheritance cases / 30 divorce articles / 209 inheritance articles.

    AgentDescription(
        agent_id="slovenia_divorce_cases",
        countries=["Slovenia"],
        legal_areas=["Divorce"],
        content_types=["case"],
        description=(
     "Slovenian case law and judicial decisions concerning divorce and "
     "separation disputes. Select this agent when the question asks how "
     "Slovenian courts have interpreted or decided issues involving spousal "
     "maintenance, division of marital assets, child custody and support, "
     "parental responsibility, or assignment of the family home. Covers "
     "decisions from the Supreme Court and lower Slovenian courts. Use the "
     "Slovenian divorce legislation agent instead when the question concerns "
     "statutory rules or legislative provisions rather than judicial decisions."
        ),
        pinecone_filter={"country": "Slovenia", "law": "Divorce", "doc_type": "Legal Cases"},
    ),

    AgentDescription(
        agent_id="slovenia_divorce_law",
        countries=["Slovenia"],
        legal_areas=["Divorce"],
        content_types=["civil_code"],
        description=(
     "Slovenian statutes and legislative provisions governing marriage, "
     "separation, and divorce. Select this agent when the question asks what "
     "Slovenian law provides regarding divorce procedures, spousal maintenance, "
     "marital property regimes, parental responsibility, child custody and "
     "support, or assignment of the family home. Use the Slovenian divorce "
     "case-law agent instead when the question concerns how courts have "
     "interpreted or applied these rules in specific disputes."
        ),
        pinecone_filter={"country": "Slovenia", "law": "Divorce", "doc_type": "Civil Codes"},
    ),

    AgentDescription(
        agent_id="slovenia_inheritance_cases",
        countries=["Slovenia"],
        legal_areas=["Inheritance"],
        content_types=["case"],
        description=(
     "Slovenian case law and judicial decisions concerning inheritance and "
     "succession disputes. Select this agent when the question asks how "
     "Slovenian courts have interpreted or decided issues involving compulsory "
     "shares and protected heirs, validity or interpretation of wills, intestate "
     "succession, acceptance or renunciation of inheritance, estate division, "
     "gifts affecting heirs' shares, or disputes among heirs. Covers decisions "
     "from the Supreme Court and lower Slovenian courts. Use the Slovenian "
     "inheritance legislation agent instead when the question concerns statutory "
     "rules or Civil Code provisions rather than judicial decisions."
        ),
        pinecone_filter={"country": "Slovenia", "law": "Inheritance", "doc_type": "Legal Cases"},
    ),

    AgentDescription(
        agent_id="slovenia_inheritance_law",
        countries=["Slovenia"],
        legal_areas=["Inheritance"],
        content_types=["civil_code"],
        description=(
     "Slovenian statutes and Civil Code provisions governing inheritance "
     "and succession. Select this agent when the question asks what Slovenian "
     "law provides regarding intestate or testamentary succession, will "
     "requirements, compulsory shares and protected heirs, classes and shares "
     "of heirs, acceptance or renunciation of inheritance, estate division, "
     "or gifts affecting heirs' rights. Use the Slovenian inheritance case-law "
     "agent instead when the question concerns how courts have interpreted "
     "or applied these rules in specific disputes."
       ), 
        pinecone_filter={"country": "Slovenia", "law": "Inheritance", "doc_type": "Civil Codes"},
    ),

    # ── ESTONIA (3 agents) ──────────────────────────────────────────────
    # Divorce is split case-law/legislation like Italy and Slovenia (91
    # divorce cases / 47 divorce articles — both large enough on their
    # own). Inheritance stays merged into one agent: only 31 inheritance
    # case-law documents exist, too thin to justify a dedicated agent.

    AgentDescription(
        agent_id="estonia_divorce_cases",
        countries=["Estonia"],
        legal_areas=["Divorce"],
        content_types=["case"],
        description=(
     "Estonian case law and judicial decisions concerning divorce and "
     "separation disputes. Select this agent when the question asks how "
     "Estonian courts have interpreted or decided issues involving division "
     "of marital assets, spousal maintenance, child custody and support, "
     "parental responsibility, or assignment of the family home. Covers "
     "decisions from the Supreme Court and lower Estonian courts. Use the "
     "Estonian divorce legislation agent instead when the question concerns "
     "statutory rules or legislative provisions rather than judicial decisions."
        ),
        pinecone_filter={"country": "Estonia", "law": "Divorce", "doc_type": "Legal Cases"},
    ),

    AgentDescription(
        agent_id="estonia_divorce_law",
        countries=["Estonia"],
        legal_areas=["Divorce"],
        content_types=["civil_code"],
        description=(
     "Estonian statutes and legislative provisions governing marriage, "
     "separation, and divorce. Select this agent when the question asks what "
     "Estonian law provides regarding divorce procedures, spousal maintenance, "
     "marital property regimes, parental responsibility, child custody and "
     "support, or assignment of the family home. Use the Estonian divorce "
     "case-law agent instead when the question concerns how courts have "
     "interpreted or applied these rules in specific disputes."
        ),
        pinecone_filter={"country": "Estonia", "law": "Divorce", "doc_type": "Civil Codes"},
    ),

    AgentDescription(
        agent_id="estonia_inheritance_agent",
        countries=["Estonia"],
        legal_areas=["Inheritance"],
        content_types=["case", "civil_code"],
        description=(
     "Estonian inheritance and succession law, combining statutes and "
     "legislative provisions with case law and judicial decisions. Select "
     "this agent for questions about compulsory portions and protected heirs, "
     "intestate or testamentary succession, will requirements and validity, "
     "classes and shares of heirs, acceptance or renunciation of inheritance, "
     "estate division, gifts affecting heirs' rights, disputes among heirs, "
     "or how Estonian courts have interpreted and applied succession rules. "
     "Covers both legislation and decisions from the Supreme Court and lower "
     "Estonian courts."
    ),
        pinecone_filter={"country": "Estonia", "law": "Inheritance"},
    ),

]

AGENT_MAP = {a.agent_id: a for a in AGENT_REGISTRY}
