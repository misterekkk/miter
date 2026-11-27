---
title: Saving models
description: Pickle and JSON, both exact.
---

```python
import pickle
blob = pickle.dumps(clf)                    # exact: bit-identical predictions after loading
clf2 = pickle.loads(blob)

clf.save_model("model.json")                # miter's JSON format, also exact
clf3 = MiterClassifier.load_model("model.json")
```

- **What the files hold:** both formats store the full model. That is the quantizers (as 17-digit
  doubles), category maps, CTR tables and trees, with the integer weights and thresholds of oblique
  splits.
- **Exactness:** predictions after loading equal the original bit for bit.
- **Labels:** `load_model` restores the model and its parameters but not a classifier's original
  labels. Labels come back as `0..K-1`. Use pickle if you need them.
- **Security:** as with any pickle, unpickle only data you trust. A JSON file is data and safe to
  read.
