# Security policy

miter trains models from data you pass to it, and loads models from JSON files (`load_model`) and
pickles. Treat a model file like code: load JSON models only from sources you trust, and **never
unpickle data from an untrusted source** (this is true of any pickle, not only miter's). The JSON
loader validates its input and is meant to reject a malformed file with an error, never a crash.

## Supported versions

Fixes go into the latest release on PyPI. Older versions do not get backports.

## Reporting a vulnerability

Please report security problems privately through GitHub:
**[Report a vulnerability](https://github.com/misterekkk/miter/security/advisories/new)**
(the repository's Security tab). Do not open a public issue for them.

Useful details: the miter version, the platform, and if possible a model file or a short script
that reproduces the problem. You should get an answer within a week. Once a fix is released, the
advisory is published with credit to the reporter, unless you prefer otherwise.

## Scope

In scope: crashes, memory errors or hangs caused by a JSON model file or by input data passed to
`fit` or `predict`, and wrong predictions caused by malformed input that was accepted.

Out of scope: pickles from untrusted sources (unpickling runs arbitrary code by design).
