from pygasflow.utils.doc_utils import modify_panel_code


def test_compressible():
    code = """
from pygasflow.interactive.compressible import compressible_app
app = compressible_app()
app.show()"""
    new_code = modify_panel_code(code)
    lines = new_code.split("\n")
    assert len(lines) == 7
    assert lines[-1] == "app"
