/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package ru.avangard.tensorflow;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Trace;
import android.util.Log;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Vector;

import ru.avangard.tensorflow.env.ImageUtils;

/**
 * A classifier specialized to label images using TensorFlow.
 */
public class TensorFlowImageClassifier implements Classifier {
    private static final String TAG = "TensorFlowImageClassif";

    // Only return this many results with at least this confidence.
    private static final int MAX_RESULTS = 3;
    private static final float THRESHOLD = 0.1f;

    // Config values.
    private String inputName;
    private String outputName;
    private int inputW;
    private int inputH;
    private int imageMean;
    private float imageStd;

    // Pre-allocated buffers.
    private Vector<String> labels = new Vector<String>();
    private int[] intValues;
    private float[] the_input;

    private float[] outputs;
    private String[] outputNames;

    private boolean logStats = false;

    private TensorFlowInferenceInterface inferenceInterface;

    Bitmap template;
    private static int numLines;
    private static int numClasses;

    private TensorFlowImageClassifier() {
    }

    /**
     * Initializes a native TensorFlow session for classifying images.
     *
     * @param assetManager  The asset manager to be used to load assets.
     * @param modelFilename The filepath of the model GraphDef protocol buffer.
     * @param labelFilename The filepath of label file for classes.
     * @param inputW        The input size. A square image of inputSize x inputSize is assumed.
     * @param imageMean     The assumed mean of the image values.
     * @param imageStd      The assumed std of the image values.
     * @param inputName     The label of the image input node.
     * @param outputName    The label of the output node.
     * @throws IOException
     */
    public static Classifier create(AssetManager assetManager, String modelFilename, String labelFilename, int
            inputW, int inputH, int imageMean, float imageStd, String inputName, String outputName) {
        TensorFlowImageClassifier c = new TensorFlowImageClassifier();
        c.inputName = inputName;
        c.outputName = outputName;
//        c.template = getBitmapFromAsset(assetManager, "A119AO777.png");
        // Read the label names into memory.
        // TODO(andrewharp): make this handle non-assets.
        String actualFilename = labelFilename.split("file:///android_asset/")[1];


        Log.i(TAG, "Reading labels from: " + actualFilename);
        BufferedReader br = null;
        try {
            br = new BufferedReader(new InputStreamReader(assetManager.open(actualFilename)));
            String line;
            while ((line = br.readLine()) != null) {
                c.labels.add(line);
            }
            br.close();
        } catch (IOException e) {
            throw new RuntimeException("Problem reading label file!", e);
        }

        c.inferenceInterface = new TensorFlowInferenceInterface(assetManager, modelFilename);

        // The shape of the output is [N, NUM_CLASSES], where N is the batch size.
        final Operation operation = c.inferenceInterface.graphOperation(outputName);
        final int numButch = (int) operation.output(0).shape().size(0);
        numLines = (int) operation.output(0).shape().size(1);
        numClasses = (int) operation.output(0).shape().size(2);
        Log.i(TAG, "numButch: " + numButch + " numLines: " + numLines + " numClasses: " + numClasses);
        Log.i(TAG, "Read " + c.labels.size() + " labels, output layer size is " + numClasses);

        // Ideally, inputSize could have been retrieved from the shape of the input operation.  Alas,
        // the placeholder node for input in the graphdef typically used does not specify a shape, so it
        // must be passed in as a parameter.
        c.inputW = inputW;
        c.inputH = inputH;
        c.imageMean = imageMean;
        c.imageStd = imageStd;

        // Pre-allocate buffers.
        c.outputNames = new String[]{outputName};
        c.intValues = new int[inputW * inputH];
        c.the_input = new float[inputW * inputH];
        c.outputs = new float[numLines * numClasses];

        return c;
    }

    public float[][] monoToBidi(final float[] array, final int rows, final int cols) {
        if (array.length != (rows * cols)) throw new IllegalArgumentException("Invalid array length");
        float[][] bidi = new float[rows][cols];
        for (int i = 0; i < rows; i++)
            System.arraycopy(array, (i * cols), bidi[i], 0, cols);

        return bidi;
    }

    public static Bitmap getBitmapFromAsset(AssetManager assetManager, String filePath) {

        InputStream istr;
        Bitmap bitmap = null;
        try {
            istr = assetManager.open(filePath);
            bitmap = BitmapFactory.decodeStream(istr);
//            bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true);

        } catch (IOException e) {
            e.printStackTrace();
            // handle exception
        }

        return bitmap;
    }

    @Override
    public List<Recognition> recognizeImage(final Bitmap bitmap) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

//        int red = 0;
//        int green = 0;
        int blue = 0;

        int index = 0;
        int globIndex = 0;
        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
//            red = ((val >> 16) & 0xFF) - imageMean;
//            green = ((val >> 8) & 0xFF) - imageMean ;
            blue = (val & 0xFF) - imageMean;
            the_input[index+globIndex] = (float) (blue / 255.0);//(float) ((red * 0.299 + green * 0.587 + blue * 0.114) / 255.0);
            index+=inputH;
            if(index >= intValues.length){
                index = 0;
                globIndex++;
            }
        }

        // For examining the actual TF input.

        Trace.endSection();


        // Copy the input data into TensorFlow.
        Trace.beginSection("feed" + inputName);
        inferenceInterface.feed(inputName, the_input, 1, inputW, inputH, 1);
        Trace.endSection();

//        Iterator<Operation> operations = inferenceInterface.graph().operations();
//        while(operations.hasNext()) {
//            Operation operation = operations.next();
//            Log.d(TAG, "op name: "+ operation);
//        }
        // Run the inference call.
        Trace.beginSection("run");
        inferenceInterface.run(outputNames, logStats);
        Trace.endSection();

        // Copy the output Tensor back into the output array.
        Trace.beginSection("fetch");
        inferenceInterface.fetch(outputName, outputs);
        Trace.endSection();

        float[][] out = monoToBidi(outputs, numLines, numClasses);

        StringBuffer sb = ctcLossString(out);

        // Find the best classifications.
        PriorityQueue<Recognition> pq = new PriorityQueue<Recognition>(3, new Comparator<Recognition>() {
            @Override
            public int compare(Recognition lhs, Recognition rhs) {
                // Intentionally reversed to put high confidence at the head of the queue.
                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
            }
        });
        if (sb.length() > 1) {
            pq.add(new Recognition("1", sb.toString(), 0f, null));
        }
        final ArrayList<Recognition> recognitions = new ArrayList<Recognition>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }
        Trace.endSection(); // "recognizeImage"
        return recognitions;
    }

    private StringBuffer ctcLossString(float[][] out) {
        ArrayList<Integer> out_best = new ArrayList<>();
        ArrayList<Integer> out_grouped = new ArrayList<>();
        for (int i = 2; i < out.length; i++) {
            int argmax = 0;
            double max = 0;
            for (int am = 0; am < out[i].length; am++) {
                if (out[i][am] > max) {
                    max = out[i][am];
                    argmax = am;
                }
            }
            out_best.add(argmax);
        }
        int lastchar = -1;
        for (Integer best : out_best) {
            if (lastchar != best) {
                out_grouped.add(best);
            }
            lastchar = best;
        }
        StringBuffer sb = new StringBuffer("");
        for (Integer c : out_grouped) {
            if (c < labels.size()) {
                sb.append(labels.get(c));
            }
        }
        return sb;
    }

    @Override
    public void enableStatLogging(boolean logStats) {
        this.logStats = logStats;
    }

    @Override
    public String getStatString() {
        return inferenceInterface.getStatString();
    }

    @Override
    public void close() {
        inferenceInterface.close();
    }


}
