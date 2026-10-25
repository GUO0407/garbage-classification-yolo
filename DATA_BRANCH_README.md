# Grasp-order dataset branch

This orphan branch contains the complete local dataset snapshot for the
closed-loop grasp-order work. It intentionally contains data only; application
code lives on `feature/grasp-order-closed-loop`.

## Snapshot

- Source dataset commit: `42dadbb97c0b82a449c679fb4582cf5bc5746761`
- `train/images`: 375
- `val/images`: 71
- `test/images`: 77
- `demo`: 86 images
- `unlabeled/images`: 2 recent RealSense captures awaiting annotation
- Reviewed grasp-order scenes: 523/523
- Direct precedence relations: 282

The `train`, `val`, and `test` directories contain images, YOLO OBB labels,
and `grasp_order` JSON sidecars. The two unfinished captures remain under
`unlabeled` and are not part of test evaluation.

All image formats covered by `.gitattributes` are stored with Git LFS. Install
Git LFS before cloning or checking out this branch.

This branch is independent from `main`; publishing it does not modify the
original code branch or dataset contents on Hugging Face.
