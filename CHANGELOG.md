# Changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and [SemVer](https://semver.org/).

## [Unreleased]

### Added
- Runnable `examples/` (crash recovery, conversation forking).
- CI: lint, format, complexity, tests on 3.10–3.13, Bandit, pip-audit, CodeQL, build check.
- LICENSE, CONTRIBUTING, SECURITY.

### Fixed
- Python 3.10 import error (`typing.Self` -> `typing_extensions.Self`).

## [0.1.0]
- Initial release: durable steps, tool replay safety, tasks/sagas, branching, approvals, pluggable storage.
