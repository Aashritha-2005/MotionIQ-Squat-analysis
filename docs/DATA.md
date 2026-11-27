# Data

[← Back to README](../README.md)

**Primary dataset:** Duddela Sai Prashanth, Priya Kamath, Rakshith Bhandary, S Ashlesh Pai, Pratham PN, Advith P,
Abhishek Mashetty. *A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad Form Labels.*
Mendeley Data, **v3** (10 Aug 2026). DOI [10.17632/kgbb3yn47p.3](https://doi.org/10.17632/kgbb3yn47p.3).
License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

What I used and changed:
- **Squat videos only:** from the "exercise quality-wise" folder, 3 views (front, side, diagonal) × good/bad, at 15 fps.
- **25 subjects, not 26:** the dataset page says 26, but the squat portion has no subject_025.
- **10 clips quarantined:** 5 pairs of squat clips were byte-identical (same SHA-256) under different labels or subjects, and both copies of each pair were dropped.
- **4 clips dropped for blur:** in these, the dataset's privacy blur covered the body on ≥30% of frames.
- **Final set:** 136 clips (69 good, 67 bad), 60,123 frames, 1,875 reps.
- **Split, by subject:** 20 subjects for training (16 train + 4 internal validation) and 5 held out (004, 009, 010, 021, 024; 30 clips). The held-out subjects were used only for the reported numbers.
- **Not redistributed:** no dataset videos are in this repository. `data/labels.csv` (committed) contains my annotations on top of the dataset's file names and good/bad labels.

**My annotations:**
- **Phase boundaries:** auto-generated from a pose-based depth signal and spot-checked.
- **Per-rep fault tags:** I added tags (`shallow_depth`, `forward_lean`, `knee_valgus`, plus `other` / `unclear`) by hand on top of the dataset's clip-level good/bad label. A fault is tagged only on views where it is observable, and is marked `n/a` elsewhere.

**External test clips** (Wikimedia Commons; used only for testing; not committed):
- "Kettlebell Racked Squats (side view)" by Taco fleur, CC BY-SA 4.0. [Commons page](https://commons.wikimedia.org/wiki/File:Kettlebell_Racked_Squats_(side_view).webm)
- "Squats Wikipedia" by CRiles23, CC BY-SA 3.0. [Commons page](https://commons.wikimedia.org/wiki/File:Squats_Wikipedia.webm)
- "Muscle Strengthening at Home - Half squat" by the US Centers for Disease Control and Prevention, public domain. [Commons page](https://commons.wikimedia.org/wiki/File:Muscle_Strengthening_at_Home_-_Half_squat.webm)
