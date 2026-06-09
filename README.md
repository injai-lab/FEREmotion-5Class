# FEREmotion 5-Class GuideBox

Android offline emotion recognition demo using a 5-class FaceNet-based FER model.

The app analyzes a centered guide-box crop from the live camera feed and predicts:

- Anger
- Disgust
- Happy
- Neutral
- Surprise

## Repository layout

```text
android_project/                 Android CameraX + ONNX Runtime app
android_project/tools/           ONNX export script
training/                        PyTorch training script
models/                          Local-only model checkpoint drop zone
docs/                            Artifact and release notes
```

Large binary artifacts are intentionally not tracked:

- `*.pth` PyTorch checkpoints
- `*.onnx` exported models
- `*.tar.gz` backups
- APK/AAB build outputs

Use Git LFS or a GitHub Release if you want to publish the model binary.

## Model artifact expected by the app

The Android app expects this local file before runtime:

```text
android_project/app/src/main/assets/facenet-5class.onnx
```

That file is ignored by git because it is about 90 MB. Copy or export the model there locally before building an APK that should run inference.

## Android build

Create `android_project/local.properties` from the example:

```text
sdk.dir=C\:/path/to/Android/Sdk
```

Then build:

```powershell
cd android_project
.\gradlew.bat --no-daemon :app:assembleDebug
```

If Gradle cannot find Java or the Android SDK, set them explicitly:

```powershell
$env:JAVA_HOME='C:\path\to\jdk-17'
$env:ANDROID_SDK_ROOT='C:\path\to\Android\Sdk'
cd android_project
.\gradlew.bat --no-daemon :app:assembleDebug
```

## Train

```powershell
pip install -r requirements-train.txt
python training\fer_5class.py --output-dir models
```

Default datasets:

- `Piro17/dataset-affecthqnet-fer2013`
- `Piro17/fer2013test`

The script keeps the original 7-class label IDs `0, 1, 3, 4, 6` and remaps them to the 5 classes above. Fear and sad are excluded.

## Export ONNX

After training, put the checkpoint here:

```text
models/facenet_5class_best.pth
```

Then export:

```powershell
pip install -r requirements-export.txt
python android_project\tools\export_facenet_5class_onnx.py
```

Default output:

```text
android_project/app/src/main/assets/facenet-5class.onnx
```

## Recorded result from the backup log

The original backup log reported:

```text
TEST ACCURACY: 84.43%
anger:    81.2%
disgust:  56.8%
happy:    91.0%
neutral:  79.9%
surprise: 84.5%
```

## GitHub upload checklist

- Do not commit `local.properties`.
- Do not commit `.onnx`, `.pth`, `.tar.gz`, APK, or Gradle build folders.
- Add a license before making the repository public.
- Publish model files through GitHub Releases or Git LFS if public distribution is needed.
