import contextlib

import ui
from ui import _assistant_reply, _spec_esta_abierta


class _FakeSessionState(dict):
    """Attribute access over a dict, like Streamlit's session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class _FakeStreamlit:
    """Minimal streamlit surface that records the close-controls gate output.

    Only the calls ``ui.main()`` makes are implemented; layout primitives are
    context managers (``nullcontext``) and widgets return inert values.
    """

    def __init__(self, *, thread_id=None, spec=None, messages=None):
        self.session_state = _FakeSessionState(
            thread_id=thread_id,
            spec=spec,
            messages=list(messages) if messages is not None else [],
        )
        self.buttons: list[str] = []
        self.captions: list[str] = []
        self.reruns = 0

    # Layout / context managers -------------------------------------------
    def set_page_config(self, **kwargs):
        pass

    def columns(self, spec, gap=None):
        return [contextlib.nullcontext(), contextlib.nullcontext()]

    def container(self, **kwargs):
        return contextlib.nullcontext()

    def chat_message(self, role):
        return contextlib.nullcontext()

    def spinner(self, text=None):
        return contextlib.nullcontext()

    # Text primitives ------------------------------------------------------
    def title(self, *args, **kwargs):
        pass

    def subheader(self, *args, **kwargs):
        pass

    def write(self, *args, **kwargs):
        pass

    def markdown(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def progress(self, *args, **kwargs):
        pass

    def caption(self, text="", *args, **kwargs):
        self.captions.append(str(text))

    # Widgets --------------------------------------------------------------
    def button(self, label, **kwargs):
        self.buttons.append(str(label))
        return False

    def chat_input(self, *args, **kwargs):
        return None

    def rerun(self):
        self.reruns += 1


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


def test_close_controls_hidden_once_the_spec_is_closed():
    """A closed spec means no close caption and no close button."""
    assert _spec_esta_abierta({"cerrada": True}) is False
    assert _spec_esta_abierta({"cerrada": False}) is True
    assert _spec_esta_abierta({}) is True


def test_main_hides_close_controls_when_spec_cerrada(monkeypatch):
    """The real render gate: a closed spec emits no close button/caption."""
    fake = _FakeStreamlit(
        thread_id="t1", spec={"cerrada": True}, messages=[("user", "hola")]
    )
    monkeypatch.setattr(ui, "st", fake)

    ui.main()

    assert "Cerrar spec" not in fake.buttons
    assert not any("Para cerrar" in caption for caption in fake.captions)


def test_main_shows_close_controls_when_spec_abierta(monkeypatch):
    """The real render gate: an open spec emits the close button and caption."""
    fake = _FakeStreamlit(
        thread_id="t1", spec={"cerrada": False}, messages=[("user", "hola")]
    )
    monkeypatch.setattr(ui, "st", fake)

    ui.main()

    assert "Cerrar spec" in fake.buttons
    assert any("Para cerrar" in caption for caption in fake.captions)


def test_main_shows_close_controls_for_a_spec_without_the_flag(monkeypatch):
    """A spec predating `cerrada` defaults to open, so the controls show."""
    fake = _FakeStreamlit(thread_id="t1", spec={}, messages=[("user", "hola")])
    monkeypatch.setattr(ui, "st", fake)

    ui.main()

    assert "Cerrar spec" in fake.buttons
    assert any("Para cerrar" in caption for caption in fake.captions)


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
