# Model Artifacts

This repository is prepared for source-code hosting. Model and release binaries are not committed.

## Local artifact names

```text
models/facenet_5class_best.pth
android_project/app/src/main/assets/facenet-5class.onnx
outputs/FEREmotion-5Class-GuideBox-offline-v1.7.apk
```

## Why they are ignored

The backup contained large artifacts around 90-112 MB. GitHub rejects files over its hard size limit and warns on large files, so these files should be distributed outside the normal git history.

Recommended options:

- GitHub Releases for APK and ONNX downloads.
- Git LFS for model files if the repository must version binaries.
- A private artifact store if the model cannot be redistributed publicly.

## Original backup notes

The source backup referenced these checksums, but the clean GitHub folder does not copy the binaries:

```text
facenet-5class.onnx
facenet_5class_best.pth
FEREmotion-5Class-GuideBox-offline-v1.7.apk
```
