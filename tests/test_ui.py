import contextlib

import ui
from demo import CYBER_TICKET
from ui import _assistant_reply, _spec_is_open


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
        self.buttons_pressed: set[str] = set()
        self.captions: list[str] = []
        self.subheaders: list[str] = []
        self.events: list[tuple[str, str]] = []
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

    def subheader(self, text="", *args, **kwargs):
        self.subheaders.append(str(text))
        self.events.append(("subheader", str(text)))

    def write(self, *args, **kwargs):
        pass

    def markdown(self, text="", *args, **kwargs):
        self.events.append(("markdown", str(text)))

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
        return str(label) in self.buttons_pressed

    def chat_input(self, *args, **kwargs):
        return None

    def rerun(self):
        self.reruns += 1


def test_assistant_reply_lists_questions_after_collision():
    spec = {
        "questions": [
            "El pedido se pisa con «El precio se cierra en el carrito» "
            "(adr-cart-price.md). ¿El comprar ahora saltea el carrito?",
            "¿Qué productos entran?",
            "¿Qué es más rápido?",
        ],
        "clashes": [
            {
                "document_id": "adr-cart-price.md",
                "title": "El precio se cierra en el carrito",
            }
        ],
        "status": {"can_close": False, "vagueness": 4, "reason": "faltan slots"},
    }
    text = _assistant_reply(spec)
    assert "Hay un choque con El precio se cierra en el carrito." in text
    assert "adr-cart-price.md" in text
    assert "2. ¿Qué productos entran?" in text
    assert "3. ¿Qué es más rápido?" in text


def test_assistant_reply_does_not_announce_a_collision_nobody_asks_about():
    """Later rounds keep the citation but ask about other gaps."""
    spec = {
        "questions": ["¿Quién pide esto?", "¿Qué queda fuera de alcance?"],
        "clashes": [
            {
                "document_id": "spec-cyber-banner.md",
                "title": "Banner Cyber Monday del año pasado",
            }
        ],
        "status": {"can_close": False, "vagueness": 3, "reason": "faltan slots"},
    }
    text = _assistant_reply(spec)
    assert "Hay un choque" not in text
    assert "Necesito aclarar el pedido." in text
    assert "Me quedan 2 preguntas." in text


def test_assistant_reply_without_questions_when_closable():
    spec = {
        "questions": [],
        "clashes": [],
        "status": {"can_close": True, "vagueness": 0, "reason": "ok"},
    }
    assert "cerrar la spec" in _assistant_reply(spec)


def test_close_controls_hidden_once_the_spec_is_closed():
    """A closed spec means no close caption and no close button."""
    assert _spec_is_open({"closed": True}) is False
    assert _spec_is_open({"closed": False}) is True
    assert _spec_is_open({}) is True


def test_main_hides_close_controls_when_spec_closed(monkeypatch):
    """The real render gate: a closed spec emits no close button/caption."""
    fake = _FakeStreamlit(
        thread_id="t1", spec={"closed": True}, messages=[("user", "hola")]
    )
    monkeypatch.setattr(ui, "st", fake)

    ui.main()

    assert "Cerrar spec" not in fake.buttons
    assert not any("Para cerrar" in caption for caption in fake.captions)


def test_main_shows_close_controls_when_spec_open(monkeypatch):
    """The real render gate: an open spec emits the close button and caption."""
    fake = _FakeStreamlit(
        thread_id="t1", spec={"closed": False}, messages=[("user", "hola")]
    )
    monkeypatch.setattr(ui, "st", fake)

    ui.main()

    assert "Cerrar spec" in fake.buttons
    assert any("Para cerrar" in caption for caption in fake.captions)


def test_main_shows_close_controls_for_a_spec_without_the_flag(monkeypatch):
    """A spec predating `closed` defaults to open, so the controls show."""
    fake = _FakeStreamlit(thread_id="t1", spec={}, messages=[("user", "hola")])
    monkeypatch.setattr(ui, "st", fake)

    ui.main()

    assert "Cerrar spec" in fake.buttons
    assert any("Para cerrar" in caption for caption in fake.captions)


def test_assistant_reply_skips_blank_questions():
    spec = {
        "questions": ["  ", "¿Alcance?"],
        "clashes": [],
        "status": {"can_close": False, "vagueness": 1, "reason": "alcance"},
    }
    text = _assistant_reply(spec)
    assert "Necesito aclarar el pedido." in text
    assert "1. ¿Alcance?" in text
    assert "2." not in text


def test_assistant_reply_does_not_announce_context_as_a_clash():
    """The lead keys off real `clashes`; a document in `context` is silent."""
    question = "¿Cuándo se reserva el stock? (adr-stock-reserve.md)"
    context_only = {
        "questions": [question],
        "clashes": [],
        "context": [
            {
                "document_id": "adr-stock-reserve.md",
                "title": "El stock se reserva al confirmar",
            }
        ],
        "status": {"can_close": False, "vagueness": 3, "reason": "contexto"},
    }
    text = _assistant_reply(context_only)
    assert "Hay un choque" not in text
    assert "Necesito aclarar el pedido." in text

    same_doc_as_clash = {**context_only, "clashes": context_only["context"]}
    assert "Hay un choque con El stock se reserva al confirmar." in _assistant_reply(
        same_doc_as_clash
    )


def test_render_spec_separates_clashes_from_context(monkeypatch):
    """Real clashes render under one header, retrieved context under another."""
    fake = _FakeStreamlit()
    monkeypatch.setattr(ui, "st", fake)
    spec = {
        "request": "checkout sin carrito",
        "clashes": [
            {
                "document_id": "adr-cart-price.md",
                "title": "El precio se cierra en el carrito",
            }
        ],
        "context": [
            {
                "document_id": "adr-stock-reserve.md",
                "title": "El stock se reserva al confirmar",
            }
        ],
    }
    ui._render_spec(spec)

    assert "Choques reales" in fake.subheaders
    assert "Contexto recuperado" in fake.subheaders

    clash_header = fake.events.index(("subheader", "Choques reales"))
    context_header = fake.events.index(("subheader", "Contexto recuperado"))
    clash_doc = next(
        index
        for index, (kind, text) in enumerate(fake.events)
        if kind == "markdown" and "adr-cart-price.md" in text
    )
    context_doc = next(
        index
        for index, (kind, text) in enumerate(fake.events)
        if kind == "markdown" and "adr-stock-reserve.md" in text
    )
    assert clash_header < clash_doc < context_header
    assert context_header < context_doc


def test_demo_button_always_starts_a_fresh_thread(monkeypatch):
    """A leftover thread id must not survive the demo button.

    With `thread_id` set and an empty chat, `_execute_prompt` would otherwise
    POST to `/threads/{id}/messages` (continuation) instead of `/threads`.
    """
    fake = _FakeStreamlit(thread_id="t-existing", spec={"closed": True})
    fake.buttons_pressed.add("Usar el ticket de demo")
    monkeypatch.setattr(ui, "st", fake)

    ui.main()

    assert fake.session_state.thread_id is None
    assert fake.session_state.pending_prompt == CYBER_TICKET


def test_demo_button_reset_preserves_round_budget(monkeypatch):
    """The thread reset must not clobber the configured round budget."""
    fake = _FakeStreamlit(thread_id="t-existing", spec={"closed": True})
    fake.session_state.max_rounds = 9
    fake.session_state.max_questions = 7
    fake.buttons_pressed.add("Usar el ticket de demo")
    monkeypatch.setattr(ui, "st", fake)

    ui.main()

    assert fake.session_state.thread_id is None
    assert fake.session_state.max_rounds == 9
    assert fake.session_state.max_questions == 7
