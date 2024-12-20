import my_library as ml


def test_module_docstring():

    assert "useful" in ml.__doc__
