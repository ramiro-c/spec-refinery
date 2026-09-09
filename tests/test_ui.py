from ui import _assistant_reply


def test_assistant_reply_lists_questions_after_collision():
    spec = {
        "preguntas": [
            "¿El comprar ahora saltea el carrito?",
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
    assert "1. ¿El comprar ahora saltea el carrito?" in text
    assert "2. ¿Qué productos entran?" in text
    assert "3. ¿Qué es más rápido?" in text


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
