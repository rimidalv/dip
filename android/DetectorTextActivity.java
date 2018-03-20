/*
 * Copyright 2016 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package ru.avangard.tensorflow;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.Button;

import org.tensorflow.demo.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Vector;

import ru.avangard.tensorflow.OverlayView.DrawCallback;
import ru.avangard.tensorflow.env.BorderedText;
import ru.avangard.tensorflow.env.ImageUtils;
import ru.avangard.tensorflow.env.Logger;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorTextActivity extends CameraActivity implements OnImageAvailableListener {
    private static final Logger LOGGER = new Logger();

    private static final int INPUT_W = 300;
    private static final int INPUT_H = 199;
    private static final String INPUT_NAME = "Placeholder,Placeholder_1";
    private static final String OUTPUT_NAMES = "rois/Reshape_4";

    private static final String MODEL_FILE = "file:///android_asset/frozen_opt_text_detect.pb";


    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
    // or YOLO.

    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_YOLO = 0.4f;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Integer sensorOrientation;

    private Classifier detector;

    private long lastProcessingTimeMs;
    private long lastProcessingTimeMsRec;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

//    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;


    // =======================================

    // =======================================

    private ResultsView resultsView;

    private Bitmap croppedBitmapRec = null;
    private Bitmap cropCopyBitmapRec = null;

    private static final int INPUT_W_REC = 400;
    private static final int INPUT_H_REC = 48;
    //    private static final int INPUT_W_REC = 148;
//    private static final int INPUT_H_REC = 64;
    private static final int IMAGE_MEAN_REC = 0;
    private static final float IMAGE_STD_REC = 255;
    private static final String INPUT_NAME_REC = "the_input";
    private static final String OUTPUT_NAME_REC = "softmax/truediv";

    private static final String MODEL_FILE_REC = "file:///android_asset/frozen_opt_text_rec.pb";
    private static final String LABEL_FILE_REC = "file:///android_asset/labels_text.txt";

    private static final boolean MAINTAIN_ASPECT = true;


    private Classifier classifier;
    private Matrix frameToCropTransformRec;
    private float pixelSizePx;
    private int index = 0;


    private String[] photos = new String[]{
            "image_0000.JPEG",
            "image_0001.JPEG",
            "image_0002.JPEG",
            "image_0003.JPEG",
            "image_0004.JPEG",
            "image_0005.JPEG",
            "image_0006.JPEG",
            "image_0009.JPEG",
            "image_0010.JPEG",
            "image_0011.JPEG",
            "image_0012.JPEG",
            "image_0017.JPEG",
            "image_0020.JPEG",
            "image_0021.JPEG",
            "image_0023.JPEG",
            "image_0024.JPEG",
            "image_0027.JPEG",
            "image_0028.JPEG",
            "image_0030.JPEG",
            "image_0031.JPEG"};

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        final float textSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());

        findViewById(R.id.button).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                index++;
                if (index >= photos.length) {
                    index = 0;
                }
                ((Button) v).setText("index: " + index);
            }
        });
        pixelSizePx = TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, 1, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

//        tracker = new MultiBoxTracker(this);

        try {
            detector = TensorFlowObjectDetectionAPIModel.Companion.create(getAssets(),
                    MODEL_FILE, INPUT_W, INPUT_H, INPUT_NAME, OUTPUT_NAMES);
        } catch (IOException e) {
            e.printStackTrace();
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);

        croppedBitmap = Bitmap.createBitmap(INPUT_W, INPUT_H, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        INPUT_W, INPUT_H,
                        sensorOrientation, true);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
//                        tracker.draw(canvas);
//                        if (isDebug()) {
//                            tracker.drawDebug(canvas);
//                        }
                    }
                });

        addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
//                        if (!isDebug()) {
//                            return;
//                        }
                        final Bitmap copy = cropCopyBitmap;
                        if (copy == null) {
                            return;
                        }

                        final int backgroundColor = Color.argb(100, 0, 0, 0);
                        canvas.drawColor(backgroundColor);

                        final Matrix matrix = new Matrix();
                        final float scaleFactor = 2;
                        matrix.postScale(scaleFactor, scaleFactor);
                        matrix.postTranslate(
                                canvas.getWidth() - copy.getWidth() * scaleFactor,
                                canvas.getHeight() - copy.getHeight() * scaleFactor);
                        canvas.drawBitmap(copy, matrix, new Paint());

                        final Vector<String> lines = new Vector<String>();
                        if (detector != null) {
                            final String statString = detector.getStatString();
                            final String[] statLines = statString.split("\n");
                            for (final String line : statLines) {
                                lines.add(line);
                            }
                        }
//                        lines.add("");
//
//                        lines.add("Frame: " + previewWidth + "x" + previewHeight);
//                        lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
//                        lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
//                        lines.add("Rotation: " + sensorOrientation);
//                        lines.add("Inference time detect: " + lastProcessingTimeMs + "ms");
//                        lines.add("Inference time rec: " + lastProcessingTimeMsRec + "ms");
//
//                        borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
                    }
                });

        // =================== RECOGNITION =====================

        classifier = TensorFlowTextImageClassifier.create(getAssets(), MODEL_FILE_REC, LABEL_FILE_REC, INPUT_W_REC, INPUT_H_REC,
                IMAGE_MEAN_REC, IMAGE_STD_REC, INPUT_NAME_REC, OUTPUT_NAME_REC);

        croppedBitmapRec = Bitmap.createBitmap(INPUT_W_REC, INPUT_H_REC, Config.ARGB_8888);

    }

    OverlayView trackingOverlay;

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();
//        tracker.onFrame(
//                previewWidth,
//                previewHeight,
//                getLuminanceStride(),
//                sensorOrientation,
//                originalLuminance,
//                timestamp);
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
        readyForNextImage();

//        rgbFrameBitmap = TensorFlowImageClassifier.getBitmapFromAsset(getAssets(), photos[index]);

        final Canvas canvas = new Canvas(croppedBitmap);

//        frameToCropTransform =
//                ImageUtils.getTransformationMatrix(
//                        rgbFrameBitmap.getWidth(), rgbFrameBitmap.getHeight(),
//                        YOLO_INPUT_SIZE, YOLO_INPUT_SIZE,
//                        0, true);

        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                        LOGGER.i("Detect: %s", results);
                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        final List<Bitmap> mappedBitmaps = new LinkedList<>();

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null){// && result.getConfidence() >= MINIMUM_CONFIDENCE_YOLO) {
                                canvas.drawRect(location, paint);

                                int x = Math.max(0, (int) (location.left));
                                int y = Math.max(0, (int) (location.top));
                                int w = (int) ((location.right - location.left));
                                int h = (int) ((location.bottom - location.top));
                                if(x + w > DESIRED_PREVIEW_SIZE.getWidth()){
                                    w -= x + w - DESIRED_PREVIEW_SIZE.getWidth();
                                }
                                if(y + h > DESIRED_PREVIEW_SIZE.getHeight()){
                                    h -= y + h - DESIRED_PREVIEW_SIZE.getHeight();
                                }

                                Bitmap cropped = Bitmap.createBitmap(cropCopyBitmap, x,y,w,h);
                                Bitmap bitmap = Bitmap.createBitmap(croppedBitmapRec);
                                final Canvas canvasRec = new Canvas(bitmap);
                                canvasRec.drawColor(Color.WHITE);
                                canvasRec.drawBitmap(cropped, new Matrix(), null);

                                cropToFrameTransform.mapRect(location);
                                result.setLocation(location);
                                mappedRecognitions.add(result);


//                                LOGGER.i("x: %d", x);
//                                LOGGER.i("y: %d", y);
//                                LOGGER.i("w: %d", w);
//                                LOGGER.i("h: %d", h);

//                                Bitmap bitmap = Bitmap.createBitmap(croppedBitmapRec);
//                                final Canvas canvasRec = new Canvas(bitmap);
//                                frameToCropTransformRec = ImageUtils.getTransformationMatrix(w, h, w, h,
//                                        sensorOrientation, MAINTAIN_ASPECT);
//                                canvasRec.drawBitmap(cropped, frameToCropTransformRec, null);
                                mappedBitmaps.add(bitmap);
                                ImageUtils.saveBitmap(bitmap);

                            }
                        }
                        ArrayList<Classifier.Recognition> resultsRecs = new ArrayList<>();
                        final long startTimeRec = SystemClock.uptimeMillis();
                        for (Bitmap bitmap : mappedBitmaps) {
                            final List<Classifier.Recognition> resultsRec = classifier.recognizeImage(bitmap);
                            resultsRecs.addAll(resultsRec);
                            lastProcessingTimeMsRec = SystemClock.uptimeMillis() - startTimeRec;
                            LOGGER.i("Recognition: %s", resultsRec);
//                            cropCopyBitmap = Bitmap.createBitmap(bitmap);


                        }
                        if (resultsView == null) {
                            resultsView = (ResultsView) findViewById(R.id.results);
                        }
                        if(mappedBitmaps.size() > 0){
                            resultsView.setResults(resultsRecs);
                        }

//                        tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                        trackingOverlay.postInvalidate();

                        requestRender();
                        computingDetection = false;
                    }
                });

//        runInBackground(
//                new Runnable() {
//                    @Override
//                    public void run() {
//                        LOGGER.i("Running detection on image " + currTimestamp);
//                        final long startTime = SystemClock.uptimeMillis();
//                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
//                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
//                        LOGGER.i("Time: %s", lastProcessingTimeMs);
//                        LOGGER.i("Detect: %s", results);
//                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
//                        final Canvas canvas = new Canvas(cropCopyBitmap);
//                        final Paint paint = new Paint();
//                        paint.setColor(Color.RED);
//                        paint.setStyle(Style.STROKE);
//                        paint.setStrokeWidth(2.0f);
//
//                        final List<Classifier.Recognition> mappedRecognitions =
//                                new LinkedList<Classifier.Recognition>();
//
//                        final List<Bitmap> mappedBitmaps = new LinkedList<>();
//
//                        for (final Classifier.Recognition result : results) {
//                            final RectF location = result.getLocation();
//                            canvas.drawRect(location, paint);
//                            cropToFrameTransform.mapRect(location);
//                            result.setLocation(location);
//                            mappedRecognitions.add(result);
//                        }
//                        if (resultsView == null) {
//                            resultsView = (ResultsView) findViewById(R.id.results);
//                        }
//
//                        requestRender();
//                        computingDetection = false;
//                    }
//                });


    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onSetDebug(final boolean debug) {
        detector.enableStatLogging(debug);
    }
}
