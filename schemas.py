from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ProviderName = Literal["gemini", "openrouter"]
RoleName = Literal["supervisor", "interrogator", "writer"]

class Citation(BaseModel):
    document_id: str
    title: str = ""
    excerpt: str = ""

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
    human_wants_close: bool = Field(
        default=False,
        description=(
            "True only when the PM asks to stop refining and close the spec. "
            "Domain talk about closing something (a price closing in the cart) "
            "is not a request to close."
        ),
    )

class SpecDocument(BaseModel):
    pedido: str
    que_entendimos: str = ""
    choques: list[Citation] = Field(default_factory=list)
    servicios: list[str] = Field(default_factory=list)
    criterios: list[AcceptanceCriterion] = Field(default_factory=list)
    preguntas: list[str] = Field(default_factory=list)
    estado: SpecStatus
