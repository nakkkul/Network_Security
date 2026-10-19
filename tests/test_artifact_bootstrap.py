import hashlib
import tempfile
import unittest
from pathlib import Path

import artifact_bootstrap as bootstrap


class ArtifactBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.file_bytes = {
            name: ("approved test fixture:" + name).encode("utf-8")
            for name in bootstrap.EXPECTED_SHA256
        }
        self.expected = {
            name: hashlib.sha256(content).hexdigest()
            for name, content in self.file_bytes.items()
        }
        self.env = {bootstrap.MODEL_VERSION_ENV: "phase7"}

    def tearDown(self):
        self.temp.cleanup()

    def target(self):
        return self.root / "Artifacts" / f"phase7_final_fit_{bootstrap.RUN_ID}"

    def create_bundle(self):
        target = self.target()
        target.mkdir(parents=True)
        for name, data in self.file_bytes.items():
            (target / name).write_bytes(data)
        return target

    def test_legacy_is_noop_and_does_not_create_artifacts(self):
        result = bootstrap.ensure_phase7_artifacts(
            env={bootstrap.MODEL_VERSION_ENV: "legacy"},
            base_dir=self.root,
            expected_sha256=self.expected,
        )
        self.assertEqual(result, "legacy_skipped")
        self.assertFalse((self.root / "Artifacts").exists())

    def test_existing_valid_bundle_is_verified(self):
        self.create_bundle()
        result = bootstrap.ensure_phase7_artifacts(
            env=self.env, base_dir=self.root, expected_sha256=self.expected
        )
        self.assertEqual(result, "verified_existing")

    def test_missing_bundle_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "missing required file"):
            bootstrap.ensure_phase7_artifacts(
                env=self.env, base_dir=self.root, expected_sha256=self.expected
            )
        self.assertFalse((self.root / "Artifacts").exists())

    def test_partial_bundle_fails_closed(self):
        target = self.target()
        target.mkdir(parents=True)
        first_name = next(iter(self.file_bytes))
        (target / first_name).write_bytes(self.file_bytes[first_name])
        with self.assertRaisesRegex(RuntimeError, "missing required file"):
            bootstrap.ensure_phase7_artifacts(
                env=self.env, base_dir=self.root, expected_sha256=self.expected
            )

    def test_hash_mismatch_fails_closed(self):
        target = self.create_bundle()
        (target / next(iter(self.file_bytes))).write_bytes(b"tampered content")
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            bootstrap.ensure_phase7_artifacts(
                env=self.env, base_dir=self.root, expected_sha256=self.expected
            )

    def test_unknown_model_version_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported"):
            bootstrap.ensure_phase7_artifacts(
                env={bootstrap.MODEL_VERSION_ENV: "typo"},
                base_dir=self.root,
                expected_sha256=self.expected,
            )

    def test_configured_artifact_directory_is_supported(self):
        target = self.root / "custom-approved-artifacts"
        target.mkdir()
        for name, data in self.file_bytes.items():
            (target / name).write_bytes(data)
        env = dict(self.env)
        env[bootstrap.ARTIFACT_DIR_ENV] = str(target)
        result = bootstrap.ensure_phase7_artifacts(
            env=env, base_dir=self.root, expected_sha256=self.expected
        )
        self.assertEqual(result, "verified_existing")

    def test_missing_configured_artifact_directory_fails(self):
        env = dict(self.env)
        env[bootstrap.ARTIFACT_DIR_ENV] = str(self.root / "missing")
        with self.assertRaisesRegex(RuntimeError, "missing required file"):
            bootstrap.ensure_phase7_artifacts(
                env=env, base_dir=self.root, expected_sha256=self.expected
            )


if __name__ == "__main__":
    unittest.main()
