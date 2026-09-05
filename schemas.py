from __future__ import annotations
from pydantic import BaseModel, Field

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
    vaguedad: int = Field(ge=0)
    razon: str

class SpecDocument(BaseModel):
    pedido: str
    que_entendimos: str = ""
    choques: list[Citation] = Field(default_factory=list)
    servicios: list[str] = Field(default_factory=list)
    criterios: list[AcceptanceCriterion] = Field(default_factory=list)
    preguntas: list[str] = Field(default_factory=list)
    estado: SpecStatus
