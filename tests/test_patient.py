"""Inventory lookup with synthetic identifiers and explicit aliases."""

import unittest

from dose_view import get_patient_data


class PatientLookupTests(unittest.TestCase):
    def test_exact_id_takes_precedence_over_alias(self):
        exact, other = {"objects": []}, {"objects": ["synthetic"]}
        inventory = {"example": exact, "alternate": other}
        self.assertIs(get_patient_data("example", inventory), exact)
        self.assertIs(get_patient_data("example", inventory, aliases={"example": "alternate"}), exact)

    def test_identifiers_are_not_guessed(self):
        inventory = {"case": {"objects": []}, "other-site": {"objects": []}}
        self.assertIsNone(get_patient_data("site-case", inventory))
        self.assertIsNone(get_patient_data("other", inventory))
        self.assertIsNone(get_patient_data("unknown", inventory))

    def test_explicit_aliases_and_missing_targets(self):
        entry = {"objects": []}
        inventory = {"example-record": entry}
        aliases = {"example-alias": "example-record", "unresolved": "missing-record"}
        self.assertIs(get_patient_data("example-alias", inventory, aliases=aliases), entry)
        self.assertIsNone(get_patient_data("unresolved", inventory, aliases=aliases))
        self.assertIsNone(get_patient_data("unknown", inventory, aliases=aliases))


if __name__ == "__main__":
    unittest.main()
