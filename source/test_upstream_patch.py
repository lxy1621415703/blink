"""Check the isolated upstream fix and its failure classification without cloning."""

import copy
import hashlib
import unittest

from run_upstream import (
    PATCH_NEW, PATCH_OLD, is_known_source_membership_failure,
    patch_controller_membership, validate_membership_records,
)


def fixture():
    return [{
        "instance": "migrations0.txt",
        "migrations": [{"switch": "s0", "source": "c0", "destination": "c1", "load": 2.0}],
        "controllers": [
            {"controller": "c0", "observed_switches": [], "observed_load": 2.0},
            {"controller": "c1", "observed_switches": ["s0"], "observed_load": 2.0},
        ],
    }]


class UpstreamPatchTests(unittest.TestCase):
    def test_patch_changes_only_the_target_and_records_hashes(self):
        original = b"# public fixture\nreturn " + PATCH_OLD + b"\n# preserved\n"
        patched, evidence = patch_controller_membership(original)
        self.assertEqual(patched, original.replace(PATCH_OLD, PATCH_NEW, 1))
        self.assertEqual(evidence["replacement_count"], 1)
        self.assertEqual(evidence["original_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(evidence["patched_sha256"], hashlib.sha256(patched).hexdigest())
        self.assertIn("-return control_const.get_constraint_switches(False)", evidence["unified_diff"])
        self.assertIn("+return control_const.get_constraint_switches(resiliency)", evidence["unified_diff"])

    def test_patch_rejects_missing_ambiguous_and_already_patched_input(self):
        for value in [b"unrelated source", PATCH_OLD + b"\n" + PATCH_OLD, PATCH_NEW]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                patch_controller_membership(value)

    def test_validator_detects_missing_source_membership(self):
        records = fixture()
        result = validate_membership_records(records, True)
        self.assertFalse(result["consistent"])
        difference = result["instances"][0]["differences"][0]
        self.assertEqual(difference["controller"], "c0")
        self.assertEqual(difference["missing_switches"], ["s0"])
        corrected = copy.deepcopy(records)
        corrected[0]["controllers"][0]["observed_switches"] = ["s0"]
        self.assertTrue(validate_membership_records(corrected, True)["consistent"])

    def test_validator_checks_destination_only_and_load(self):
        records = fixture()
        records[0]["controllers"][0]["observed_load"] = 0.0
        self.assertTrue(validate_membership_records(records, False)["consistent"])
        records[0]["controllers"][1]["observed_load"] = 9.0
        self.assertFalse(validate_membership_records(records, False)["consistent"])

    def test_only_the_demonstrated_source_error_is_accepted(self):
        record = {"returncode": 1, "timed_out": False, "error": None,
                  "vector_first_fit_rounds": 1, "current_bottleneck_first_rounds": None}
        validation = validate_membership_records(fixture(), True)["instances"][0]
        context = "in remove_migration_from_constraints\nin remove_migration_from_controller\n"
        expected = context + "MigrationScheduling.exceptions.SwitchNotFound: Switch s0 not found.\n"
        self.assertTrue(is_known_source_membership_failure(record, expected, validation))
        self.assertTrue(is_known_source_membership_failure(record, context + "KeyError: 'c0'\n", validation))
        for stderr in [context + "TypeError: unsupported API\n",
                       context + "KeyError: 'c9'\n", expected.replace("s0", "s9")]:
            self.assertFalse(is_known_source_membership_failure(record, stderr, validation))
        record["timed_out"] = True
        self.assertFalse(is_known_source_membership_failure(record, expected, validation))


if __name__ == "__main__":
    unittest.main()

