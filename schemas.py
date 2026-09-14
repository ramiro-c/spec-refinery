from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["gemini", "openrouter"]
RoleName = Literal["supervisor", "interrogator", "writer"]

class Citation(BaseModel):
    document_id: str
    title: str = ""
    excerpt: str = ""

class DocumentAssessment(BaseModel):
    """The interrogator's per-document verdict for one retrieved rule."""

    document_id: str
    tipo: Literal["choque", "contexto"]

class AcceptanceCriterion(BaseModel):
    dado: str = ""
    cuando: str = ""
    entonces: str = ""

class SpecStatus(BaseModel):
    se_puede_cerrar: bool
    vaguedad: int = Field(ge=0, le=10)
    razon: str


class Interrogation(BaseModel):
    """What the interrogator decided after grilling the PM this round."""

    # First on purpose: it is a fact about the human's message, and answering
    # it before reasoning about quality keeps the two from bleeding together.
    human_wants_close: bool = Field(
        default=False,
        description=(
            "Did the human ASK to close the spec in their last message? This "
            "is a fact about what they wrote, never your opinion about whether "
            "closing is wise. If they asked, it is true even when the spec is "
            "incomplete: you have no authority to refuse. Domain talk about "
            "closing something (a price closing in the cart) is not a request."
        ),
    )
    decisiones: list[Decision] = Field(
        default_factory=list,
        description=(
            "Arguments the PM settled in this round, including any company "
            "rule they decided to change, deprecate or except. Record it here "
            "and stop asking about it."
        ),
    )
    preguntas: list[str] = Field(
        default_factory=list,
        description=(
            "Open questions for the PM. Ask as many as this request genuinely "
            "needs, in Rioplatense Spanish. Empty only when nothing is left to "
            "settle."
        ),
    )
    se_puede_cerrar: bool = Field(
        description=(
            "True only when an engineer could implement this and a tester could "
            "write the acceptance criteria without asking anything else."
        )
    )
    vaguedad: int = Field(
        ge=0,
        le=10,
        description="0 when nothing is left to clarify, 10 for a one-line wish.",
    )
    razon: str = Field(description="Why, in one sentence, in Spanish.")
    clasificaciones: list[DocumentAssessment] = Field(
        default_factory=list,
        description=(
            "One entry per document retrieved this turn: \"choque\" when the "
            "request genuinely contradicts that rule, \"contexto\" when the "
            "document informs the request without being contradicted."
        ),
    )


class Decision(BaseModel):
    """Something the PM settled, including overruling a company rule.

    A retrieved rule is evidence, not a veto: the PM may adapt the request to
    it, decide to change it, or accept a scoped exception. Whichever way it
    goes, it stops being an open question and becomes part of the spec.
    """

    tema: str = Field(description="What was being argued, in a few words.")
    decision: str = Field(description="What the PM decided.")
    impacto: str = Field(
        default="",
        description=(
            "What this costs: rules to rewrite, services to touch, risks taken."
        ),
    )


class SpecDocument(BaseModel):
    pedido: str
    que_entendimos: str = ""
    choques: list[Citation] = Field(default_factory=list)
    contexto: list[Citation] = Field(default_factory=list)
    decisiones: list[Decision] = Field(default_factory=list)
    servicios: list[str] = Field(default_factory=list)
    criterios: list[AcceptanceCriterion] = Field(default_factory=list)
    preguntas: list[str] = Field(default_factory=list)
    estado: SpecStatus
    cerrada: bool = False
