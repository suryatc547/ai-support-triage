from unittest.mock import patch

from src.configs.security_config import _load_domain_list


def test_load_domain_list(tmp_path):
    # Create a dummy config file with comments, uppercase, and spaces
    dummy_conf = tmp_path / "dummy.conf"
    dummy_conf.write_text(
        "domain1.com\n# A comment\n  domain2.net  \n\nDOMAIN3.ORG", encoding="utf-8"
    )

    with patch("src.configs.security_config.os.path.dirname", return_value=str(tmp_path)):
        domains = _load_domain_list("dummy.conf")

    assert domains == ["domain1.com", "domain2.net", "domain3.org"]


def test_load_domain_list_file_not_found(tmp_path):
    with patch("src.configs.security_config.os.path.dirname", return_value=str(tmp_path)):
        domains = _load_domain_list("nonexistent.conf")
    assert domains == []
