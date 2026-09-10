from ui import _assistant_reply


def test_assistant_reply_lists_questions_after_collision():
    spec = {
        "preguntas": [
            "El pedido se pisa con «El precio se cierra en el carrito» "
            "(adr-cart-price.md). ¿El comprar ahora saltea el carrito?",
            "¿Qué productos entran?",
            "¿Qué es más rápido?",
        ],
        "choques": [
            {
                "document_id": "adr-cart-price.md",
                "title": "El precio se cierra en el carrito",
            }
        ],
        "estado": {"se_puede_cerrar": False, "vaguedad": 4, "razon": "faltan slots"},
    }
    text = _assistant_reply(spec)
    assert "Hay un choque con El precio se cierra en el carrito." in text
    assert "adr-cart-price.md" in text
    assert "2. ¿Qué productos entran?" in text
    assert "3. ¿Qué es más rápido?" in text


def test_assistant_reply_does_not_announce_a_collision_nobody_asks_about():
    """Later rounds keep the citation but ask about other gaps."""
    spec = {
        "preguntas": ["¿Quién pide esto?", "¿Qué queda fuera de alcance?"],
        "choques": [
            {
                "document_id": "spec-cyber-banner.md",
                "title": "Banner Cyber Monday del año pasado",
            }
        ],
        "estado": {"se_puede_cerrar": False, "vaguedad": 3, "razon": "faltan slots"},
    }
    text = _assistant_reply(spec)
    assert "Hay un choque" not in text
    assert "Necesito aclarar el pedido." in text
    assert "Me quedan 2 preguntas." in text


def test_assistant_reply_without_questions_when_closable():
    spec = {
        "preguntas": [],
        "choques": [],
        "estado": {"se_puede_cerrar": True, "vaguedad": 0, "razon": "ok"},
    }
    assert "cerrar la spec" in _assistant_reply(spec)


def test_assistant_reply_skips_blank_questions():
    spec = {
        "preguntas": ["  ", "¿Alcance?"],
        "choques": [],
        "estado": {"se_puede_cerrar": False, "vaguedad": 1, "razon": "alcance"},
    }
    text = _assistant_reply(spec)
    assert "Necesito aclarar el pedido." in text
    assert "1. ¿Alcance?" in text
    assert "2." not in text
