package com.codex.feremotion;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.PorterDuff;
import android.graphics.PorterDuffXfermode;
import android.graphics.Rect;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.os.Bundle;
import android.os.SystemClock;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.TextView;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.Collections;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class MainActivity extends AppCompatActivity {
    private static final String MODEL_ASSET = "facenet-5class.onnx";
    private static final int MODEL_SIZE = 160;
    private static final long ANALYSIS_INTERVAL_MS = 1200L;
    private static final float GUIDE_WIDTH_RATIO = 0.68f;
    private static final float GUIDE_TOP_RATIO = 0.23f;
    private static final float GUIDE_HEIGHT_FROM_WIDTH = 1.22f;
    private static final float GUIDE_MAX_HEIGHT_RATIO = 0.42f;
    private static final int COLOR_BACKGROUND = Color.rgb(8, 17, 30);
    private static final int COLOR_CARD = Color.argb(224, 13, 29, 47);
    private static final int COLOR_ACCENT = Color.rgb(54, 226, 194);
    private static final int COLOR_MUTED = Color.rgb(167, 187, 204);
    private static final int COLOR_TRACK = Color.rgb(39, 62, 80);
    private static final String[] LABELS = {
            "Anger", "Disgust", "Happy", "Neutral", "Surprise"
    };
    private static final float[] MEAN = {0.5f, 0.5f, 0.5f};
    private static final float[] STD = {0.5f, 0.5f, 0.5f};

    private final ExecutorService cameraExecutor = Executors.newSingleThreadExecutor();
    private final ExecutorService modelExecutor = Executors.newSingleThreadExecutor();
    private final AtomicBoolean analyzing = new AtomicBoolean(false);

    private PreviewView previewView;
    private TextView statusText;
    private TextView emotionText;
    private TextView confidenceText;
    private TextView latencyText;
    private final ProgressBar[] scoreBars = new ProgressBar[LABELS.length];
    private final TextView[] scoreTexts = new TextView[LABELS.length];

    private OrtEnvironment ortEnvironment;
    private OrtSession ortSession;
    private String inputName;
    private float[] smoothedScores;
    private long lastAnalyzedAt;

    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) {
                    setStatus("Opening camera...", COLOR_ACCENT);
                    startCamera();
                } else {
                    setStatus("Camera permission is required.", Color.rgb(255, 176, 102));
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
        initializeModel();
        requestCamera();
    }

    private void buildUi() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(COLOR_BACKGROUND);

        previewView = new PreviewView(this);
        previewView.setImplementationMode(PreviewView.ImplementationMode.COMPATIBLE);
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        root.addView(previewView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        root.addView(new FaceGuideView(this), new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
        ));

        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.VERTICAL);
        header.setPadding(dp(18), dp(16), dp(18), dp(16));
        header.setBackground(roundedBackground(COLOR_CARD, 20));

        TextView title = makeText("Real-time 5-class FER", 20, Color.WHITE, true);
        header.addView(title);

        TextView subtitle = makeText(
                "Anger, disgust, happy, neutral, and surprise from the guide box.",
                12,
                COLOR_MUTED,
                false
        );
        LinearLayout.LayoutParams subtitleParams = wrapParams();
        subtitleParams.topMargin = dp(4);
        header.addView(subtitle, subtitleParams);

        statusText = makeText("Loading model...", 12, COLOR_ACCENT, true);
        statusText.setPadding(dp(10), dp(6), dp(10), dp(6));
        statusText.setBackground(roundedBackground(Color.argb(112, 18, 78, 74), 14));
        LinearLayout.LayoutParams statusParams = wrapParams();
        statusParams.topMargin = dp(12);
        header.addView(statusText, statusParams);

        FrameLayout.LayoutParams headerParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
        headerParams.setMargins(dp(14), dp(14), dp(14), 0);
        headerParams.gravity = Gravity.TOP;
        root.addView(header, headerParams);

        LinearLayout resultCard = new LinearLayout(this);
        resultCard.setOrientation(LinearLayout.VERTICAL);
        resultCard.setPadding(dp(18), dp(14), dp(18), dp(16));
        resultCard.setBackground(roundedBackground(COLOR_CARD, 22));

        LinearLayout primaryRow = new LinearLayout(this);
        primaryRow.setGravity(Gravity.CENTER_VERTICAL);

        LinearLayout primaryText = new LinearLayout(this);
        primaryText.setOrientation(LinearLayout.VERTICAL);
        primaryText.setLayoutParams(new LinearLayout.LayoutParams(
                0,
                ViewGroup.LayoutParams.WRAP_CONTENT,
                1f
        ));
        TextView primaryLabel = makeText("Current emotion", 11, COLOR_MUTED, true);
        primaryText.addView(primaryLabel);
        emotionText = makeText("Waiting", 27, Color.WHITE, true);
        LinearLayout.LayoutParams emotionParams = wrapParams();
        emotionParams.topMargin = dp(2);
        primaryText.addView(emotionText, emotionParams);
        primaryRow.addView(primaryText);

        confidenceText = makeText("--%", 24, COLOR_ACCENT, true);
        primaryRow.addView(confidenceText);
        resultCard.addView(primaryRow);

        latencyText = makeText("The model is loading.", 11, COLOR_MUTED, false);
        LinearLayout.LayoutParams latencyParams = wrapParams();
        latencyParams.topMargin = dp(2);
        latencyParams.bottomMargin = dp(10);
        resultCard.addView(latencyText, latencyParams);

        for (int i = 0; i < LABELS.length; i++) {
            resultCard.addView(makeScoreRow(i));
        }

        ScrollView resultScroll = new ScrollView(this);
        resultScroll.setFillViewport(true);
        resultScroll.addView(resultCard);
        FrameLayout.LayoutParams resultParams = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                dp(340)
        );
        resultParams.setMargins(dp(14), 0, dp(14), dp(14));
        resultParams.gravity = Gravity.BOTTOM;
        root.addView(resultScroll, resultParams);

        setContentView(root);
    }

    private View makeScoreRow(int index) {
        LinearLayout row = new LinearLayout(this);
        row.setGravity(Gravity.CENTER_VERTICAL);
        row.setPadding(0, dp(3), 0, dp(3));

        TextView label = makeText(LABELS[index], 12, Color.WHITE, true);
        row.addView(label, new LinearLayout.LayoutParams(dp(64), ViewGroup.LayoutParams.WRAP_CONTENT));

        ProgressBar bar = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        bar.setMax(1000);
        bar.setProgress(0);
        bar.setProgressTintList(ColorStateList.valueOf(COLOR_ACCENT));
        bar.setProgressBackgroundTintList(ColorStateList.valueOf(COLOR_TRACK));
        LinearLayout.LayoutParams barParams = new LinearLayout.LayoutParams(0, dp(8), 1f);
        barParams.setMargins(dp(5), 0, dp(10), 0);
        row.addView(bar, barParams);
        scoreBars[index] = bar;

        TextView score = makeText("0%", 11, COLOR_MUTED, true);
        score.setGravity(Gravity.END);
        row.addView(score, new LinearLayout.LayoutParams(dp(48), ViewGroup.LayoutParams.WRAP_CONTENT));
        scoreTexts[index] = score;
        return row;
    }

    private void requestCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            startCamera();
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA);
        }
    }

    private void initializeModel() {
        modelExecutor.execute(() -> {
            try {
                ortEnvironment = OrtEnvironment.getEnvironment();
                OrtSession.SessionOptions options = new OrtSession.SessionOptions();
                options.setIntraOpNumThreads(2);
                File modelFile = prepareModelFile(MODEL_ASSET);
                ortSession = ortEnvironment.createSession(modelFile.getAbsolutePath(), options);
                inputName = ortSession.getInputNames().iterator().next();
                runOnUiThread(() -> {
                    setStatus("Offline model is ready.", COLOR_ACCENT);
                    latencyText.setText(R.string.selfie_ready);
                });
            } catch (Exception exception) {
                runOnUiThread(() -> {
                    setStatus("Model load failed.", Color.rgb(255, 121, 121));
                    latencyText.setText(exception.getClass().getSimpleName());
                });
            }
        });
    }

    private File prepareModelFile(String assetName) throws IOException {
        File modelDir = new File(getFilesDir(), "models");
        if (!modelDir.isDirectory() && !modelDir.mkdirs()) {
            throw new IOException("Could not create model directory");
        }

        File outputFile = new File(modelDir, assetName);
        long assetLength = -1L;
        try (android.content.res.AssetFileDescriptor descriptor = getAssets().openFd(assetName)) {
            assetLength = descriptor.getLength();
        } catch (IOException ignored) {
            // Some asset configurations do not expose a length. Copy verification still runs below.
        }

        if (outputFile.isFile() && (assetLength < 0L || outputFile.length() == assetLength)) {
            return outputFile;
        }

        File tempFile = new File(modelDir, assetName + ".tmp");
        try (InputStream input = getAssets().open(assetName);
             FileOutputStream output = new FileOutputStream(tempFile)) {
            byte[] buffer = new byte[1024 * 1024];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
        }

        if (assetLength >= 0L && tempFile.length() != assetLength) {
            tempFile.delete();
            throw new IOException("Copied model size mismatch");
        }
        if (outputFile.exists() && !outputFile.delete()) {
            tempFile.delete();
            throw new IOException("Could not replace existing model");
        }
        if (!tempFile.renameTo(outputFile)) {
            tempFile.delete();
            throw new IOException("Could not move model into place");
        }
        return outputFile;
    }

    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> providerFuture =
                ProcessCameraProvider.getInstance(this);
        providerFuture.addListener(() -> {
            try {
                bindCamera(providerFuture.get(), CameraSelector.DEFAULT_FRONT_CAMERA);
            } catch (Exception frontCameraException) {
                try {
                    bindCamera(providerFuture.get(), CameraSelector.DEFAULT_BACK_CAMERA);
                    setStatus("Using the rear camera.", COLOR_ACCENT);
                } catch (Exception exception) {
                    setStatus("No camera is available.", Color.rgb(255, 121, 121));
                }
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private void bindCamera(ProcessCameraProvider cameraProvider, CameraSelector selector) {
        Preview preview = new Preview.Builder().build();
        preview.setSurfaceProvider(previewView.getSurfaceProvider());

        ImageAnalysis analysis = new ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build();
        analysis.setAnalyzer(cameraExecutor, this::analyzeFrame);

        cameraProvider.unbindAll();
        cameraProvider.bindToLifecycle(this, selector, preview, analysis);
        if (ortSession == null) {
            setStatus("Camera is open. Waiting for the model.", COLOR_ACCENT);
        }
    }

    private void analyzeFrame(@NonNull ImageProxy image) {
        long now = SystemClock.elapsedRealtime();
        if (ortSession == null || now - lastAnalyzedAt < ANALYSIS_INTERVAL_MS
                || !analyzing.compareAndSet(false, true)) {
            image.close();
            return;
        }
        lastAnalyzedAt = now;

        try {
            long startedAt = SystemClock.elapsedRealtime();
            Bitmap bitmap = imageProxyToBitmap(image);
            Rect guideBounds = guideCropBounds(bitmap);
            Bitmap faceCrop = cropGuideArea(bitmap, guideBounds);
            bitmap.recycle();
            Bitmap modelInput = Bitmap.createScaledBitmap(faceCrop, MODEL_SIZE, MODEL_SIZE, true);
            faceCrop.recycle();
            FloatBuffer input = bitmapToNormalizedTensor(modelInput, MODEL_SIZE, MEAN, STD);
            float[] scores = runInference(input);
            long elapsed = SystemClock.elapsedRealtime() - startedAt;
            updateScores(scores, elapsed);
        } catch (Exception exception) {
            runOnUiThread(() -> setStatus(
                    "Analysis failed. Re-align your face.",
                    Color.rgb(255, 176, 102)
            ));
        } finally {
            image.close();
            analyzing.set(false);
        }
    }

    private Bitmap imageProxyToBitmap(ImageProxy image) {
        ImageProxy.PlaneProxy plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int width = image.getWidth();
        int height = image.getHeight();
        int rowStride = plane.getRowStride();
        int pixelStride = plane.getPixelStride();
        int[] pixels = new int[width * height];

        for (int y = 0; y < height; y++) {
            int rowOffset = y * rowStride;
            for (int x = 0; x < width; x++) {
                int offset = rowOffset + x * pixelStride;
                int red = buffer.get(offset) & 0xff;
                int green = buffer.get(offset + 1) & 0xff;
                int blue = buffer.get(offset + 2) & 0xff;
                pixels[y * width + x] = Color.rgb(red, green, blue);
            }
        }

        Bitmap bitmap = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888);
        int rotation = image.getImageInfo().getRotationDegrees();
        if (rotation == 0) {
            return bitmap;
        }

        Matrix matrix = new Matrix();
        matrix.postRotate(rotation);
        Bitmap rotated = Bitmap.createBitmap(bitmap, 0, 0, width, height, matrix, true);
        bitmap.recycle();
        return rotated;
    }

    private Bitmap cropGuideArea(Bitmap bitmap, Rect guideBounds) {
        return Bitmap.createBitmap(
                bitmap,
                guideBounds.left,
                guideBounds.top,
                guideBounds.width(),
                guideBounds.height()
        );
    }

    private Rect guideCropBounds(Bitmap bitmap) {
        RectF guide = guideBoundsFor(bitmap.getWidth(), bitmap.getHeight());
        int left = clamp(Math.round(guide.left), 0, bitmap.getWidth() - 1);
        int top = clamp(Math.round(guide.top), 0, bitmap.getHeight() - 1);
        int right = clamp(Math.round(guide.right), left + 1, bitmap.getWidth());
        int bottom = clamp(Math.round(guide.bottom), top + 1, bitmap.getHeight());
        return new Rect(left, top, right, bottom);
    }

    private static RectF guideBoundsFor(float width, float height) {
        float boxWidth = width * GUIDE_WIDTH_RATIO;
        float boxHeight = Math.min(boxWidth * GUIDE_HEIGHT_FROM_WIDTH, height * GUIDE_MAX_HEIGHT_RATIO);
        float left = (width - boxWidth) / 2f;
        float top = height * GUIDE_TOP_RATIO;
        return new RectF(left, top, left + boxWidth, top + boxHeight);
    }

    private int clamp(int value, int minimum, int maximum) {
        return Math.max(minimum, Math.min(value, maximum));
    }

    private FloatBuffer bitmapToNormalizedTensor(Bitmap bitmap, int size, float[] mean, float[] std) {
        int[] pixels = new int[size * size];
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size);
        bitmap.recycle();

        FloatBuffer tensor = ByteBuffer
                .allocateDirect(3 * size * size * Float.BYTES)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer();
        int planeSize = size * size;
        float[] values = new float[3 * planeSize];
        for (int i = 0; i < pixels.length; i++) {
            int color = pixels[i];
            values[i] = ((Color.red(color) / 255f) - mean[0]) / std[0];
            values[planeSize + i] = ((Color.green(color) / 255f) - mean[1]) / std[1];
            values[2 * planeSize + i] = ((Color.blue(color) / 255f) - mean[2]) / std[2];
        }
        tensor.put(values);
        tensor.rewind();
        return tensor;
    }

    private float[] runInference(FloatBuffer input) throws OrtException {
        long[] shape = {1, 3, MODEL_SIZE, MODEL_SIZE};
        try (OnnxTensor tensor = OnnxTensor.createTensor(ortEnvironment, input, shape);
             OrtSession.Result result = ortSession.run(
                     Collections.singletonMap(inputName, tensor))) {
            float[][] probabilities = (float[][]) result.get(0).getValue();
            return normalizeProbabilities(probabilities[0]);
        }
    }

    private float[] normalizeProbabilities(float[] probabilities) {
        float sum = 0f;
        float[] scores = new float[probabilities.length];
        for (int i = 0; i < probabilities.length; i++) {
            scores[i] = Math.max(0f, probabilities[i]);
            sum += scores[i];
        }
        if (sum <= 0f) {
            return scores;
        }
        for (int i = 0; i < scores.length; i++) {
            scores[i] /= sum;
        }
        return scores;
    }

    private void updateScores(float[] freshScores, long elapsedMs) {
        if (freshScores.length != LABELS.length) {
            runOnUiThread(() -> setStatus("Unexpected model output shape.", Color.rgb(255, 121, 121)));
            return;
        }

        if (smoothedScores == null) {
            smoothedScores = freshScores.clone();
        } else {
            for (int i = 0; i < freshScores.length; i++) {
                smoothedScores[i] = smoothedScores[i] * 0.55f + freshScores[i] * 0.45f;
            }
        }

        int bestIndex = 0;
        for (int i = 1; i < smoothedScores.length; i++) {
            if (smoothedScores[i] > smoothedScores[bestIndex]) {
                bestIndex = i;
            }
        }
        int finalBestIndex = bestIndex;
        float[] displayScores = smoothedScores.clone();
        runOnUiThread(() -> {
            setStatus("Guide-box 5-class analysis is running.", COLOR_ACCENT);
            emotionText.setText(LABELS[finalBestIndex]);
            confidenceText.setText(percent(displayScores[finalBestIndex]));
            latencyText.setText(String.format(
                    Locale.US,
                    "Latest analysis: %d ms. Frames stay on device.",
                    elapsedMs
            ));
            for (int i = 0; i < displayScores.length; i++) {
                scoreBars[i].setProgress(Math.round(displayScores[i] * 1000f), true);
                scoreTexts[i].setText(percent(displayScores[i]));
            }
        });
    }

    private String percent(float score) {
        return String.format(Locale.US, "%.1f%%", score * 100f);
    }

    private void setStatus(String message, int color) {
        if (statusText == null) {
            return;
        }
        statusText.setText(message);
        statusText.setTextColor(color);
    }

    private TextView makeText(String text, int sizeSp, int color, boolean bold) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(sizeSp);
        view.setTextColor(color);
        if (bold) {
            view.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        }
        return view;
    }

    private GradientDrawable roundedBackground(int color, int radiusDp) {
        GradientDrawable drawable = new GradientDrawable();
        drawable.setColor(color);
        drawable.setCornerRadius(dp(radiusDp));
        return drawable;
    }

    private LinearLayout.LayoutParams wrapParams() {
        return new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        );
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    @Override
    protected void onDestroy() {
        cameraExecutor.shutdown();
        modelExecutor.shutdown();
        if (ortSession != null) {
            try {
                ortSession.close();
            } catch (OrtException ignored) {
                // The app is already closing.
            }
        }
        super.onDestroy();
    }

    private static class FaceGuideView extends View {
        private final Paint shadePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint clearPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint guidePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint lensPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
        private final RectF faceBox = new RectF();

        FaceGuideView(Context context) {
            super(context);
            setLayerType(View.LAYER_TYPE_SOFTWARE, null);
            shadePaint.setColor(Color.argb(104, 0, 0, 0));
            clearPaint.setXfermode(new PorterDuffXfermode(PorterDuff.Mode.CLEAR));
            guidePaint.setColor(COLOR_ACCENT);
            guidePaint.setStrokeWidth(dp(context, 4));
            guidePaint.setStyle(Paint.Style.STROKE);
            guidePaint.setStrokeCap(Paint.Cap.ROUND);
            lensPaint.setColor(Color.argb(176, 54, 226, 194));
            lensPaint.setStrokeWidth(dp(context, 2));
            lensPaint.setStyle(Paint.Style.STROKE);
            textPaint.setColor(Color.WHITE);
            textPaint.setTextSize(dp(context, 14));
            textPaint.setTextAlign(Paint.Align.CENTER);
            textPaint.setTypeface(Typeface.DEFAULT_BOLD);
            textPaint.setShadowLayer(dp(context, 5), 0, dp(context, 2), Color.BLACK);
        }

        @Override
        protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float width = getWidth();
            float height = getHeight();
            faceBox.set(guideBoundsFor(width, height));
            float corner = dp(getContext(), 35);
            float radius = dp(getContext(), 28);

            canvas.drawRect(0, 0, width, height, shadePaint);
            canvas.drawRoundRect(faceBox, radius, radius, clearPaint);
            canvas.drawRoundRect(faceBox, radius, radius, guidePaint);
            drawCorner(canvas, faceBox.left, faceBox.top, corner, 1, 1);
            drawCorner(canvas, faceBox.right, faceBox.top, corner, -1, 1);
            drawCorner(canvas, faceBox.left, faceBox.bottom, corner, 1, -1);
            drawCorner(canvas, faceBox.right, faceBox.bottom, corner, -1, -1);
            canvas.drawCircle(
                    faceBox.centerX(),
                    faceBox.centerY(),
                    Math.min(faceBox.width(), faceBox.height()) * 0.16f,
                    lensPaint
            );
            canvas.drawLine(
                    faceBox.centerX() - faceBox.width() * 0.12f,
                    faceBox.centerY(),
                    faceBox.centerX() + faceBox.width() * 0.12f,
                    faceBox.centerY(),
                    lensPaint
            );
            canvas.drawLine(
                    faceBox.centerX(),
                    faceBox.centerY() - faceBox.height() * 0.09f,
                    faceBox.centerX(),
                    faceBox.centerY() + faceBox.height() * 0.09f,
                    lensPaint
            );
            canvas.drawText(
                    "Only the guide-box crop is analyzed",
                    width / 2f,
                    faceBox.bottom + dp(getContext(), 28),
                    textPaint
            );
        }

        private void drawCorner(Canvas canvas, float x, float y, float length, int xDirection, int yDirection) {
            canvas.drawLine(x, y, x + length * xDirection, y, guidePaint);
            canvas.drawLine(x, y, x, y + length * yDirection, guidePaint);
        }

        private static int dp(Context context, int value) {
            return Math.round(value * context.getResources().getDisplayMetrics().density);
        }
    }
}
