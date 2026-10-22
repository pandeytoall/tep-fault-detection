# Two tracks, never ranked together, with an unseen-fault test

Unsupervised models (normal runs only, like PCA) and the supervised model (trained on labelled faulty runs) are reported in separate tables and never ranked together. On TEP's 20 known faults a supervised model looks far better than PCA, but a real plant rarely has labels for its next fault. One ranking would reward having labels, not detecting faults. To show that with our own numbers rather than claim it, the supervised model is retrained once without faults 1, 8, 13, 14 and 16 (the lowest-numbered fault of each Downs & Vogel type, skipping the near-undetectable 3/9/15; fixed before any results) and compared on those faults with the full supervised model and PCA.

**Considered options:** one combined ranking (simpler, misleading); unsupervised only (drops the "what labels buy you" result); leave-one-fault-out (20 retrains, too costly for the time budget).
