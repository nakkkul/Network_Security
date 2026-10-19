# Check the real local Git state before creating `v2-development`

The public GitHub commit `bb4f0b9` exists and is titled **Prepare V1 resume release**. This package does not contain `.git`, and the execution environment cannot inspect the user's local repository. Therefore, do these checks in VS Code Terminal from the root of the local Git clone before creating a branch.

```cmd
git status --short
git branch --show-current
git show -s --format="%H %s" resume-v1
git ls-remote origin refs/heads/resume-v1
```

1. `git status --short` should be empty. If it is not empty, stop and inspect those changes; do not discard them.
2. `git show ... resume-v1` should show the expected V1 release commit/message or a reviewed descendant. Confirm the full commit begins with `bb4f0b9` if that is still the tip of the release branch.
3. The hash returned by `git ls-remote` is the current remote `resume-v1` tip. Compare it to `git rev-parse resume-v1`. If they differ, fetch/pull and inspect rather than guessing.

If the working tree is clean and the local `resume-v1` is aligned with the remote V1 release:

```cmd
git switch resume-v1
git pull --ff-only origin resume-v1
git switch -c v2-development
```

Then extract/copy only the `v2_phase2_evaluation` folder from this package into the repository root. Do not copy it to `main` or `resume-v1`.
