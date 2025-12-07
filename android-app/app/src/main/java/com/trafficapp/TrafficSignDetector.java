package com.trafficapp;

import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.util.Log;
import org.tensorflow.lite.Interpreter;
//Removed GPU delegate imports
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;

public class TrafficSignDetector {
    private static final String TAG = "TrafficSignDetector";
    private static final String MODEL_FILE = "traffic_signs_yolov11_int8.tflite";
    private static final int INPUT_SIZE = 320;
    private static final int NUM_THREADS = 4;
    
    private Interpreter interpreter;
    // Removed GpuDelegate declaration
    
    // Bengali names for traffic signs
    private static final String[] CLASS_NAMES = {
        "থামুন", // Stop
        "গতি সীমা", // Speed limit
        "নো এন্ট্রি", // No entry
        "বাঁ দিকে যান", // Turn left
        "ডান দিকে যান", // Turn right
    };

    public TrafficSignDetector(android.content.Context context) {
        try {
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(NUM_THREADS);
            
            // GPU acceleration code removed for now
            
            MappedByteBuffer modelBuffer = loadModelFile(context);
            interpreter = new Interpreter(modelBuffer, options);
            Log.d(TAG, "Model loaded successfully");
        } catch (Exception e) {
            Log.e(TAG, "Error loading model", e);
        }
    }

    private MappedByteBuffer loadModelFile(android.content.Context context) throws Exception {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(MODEL_FILE);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public List<Detection> detect(Bitmap bitmap) {
        if (interpreter == null) {
            Log.e(TAG, "Interpreter not initialized");
            return new ArrayList<>();
        }

        // Preprocess image
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true);
        ByteBuffer inputBuffer = preprocessImage(resizedBitmap);

        // Run inference
        float[][][] output = new float[1][25200][85]; // YOLOv11 output format
        interpreter.run(inputBuffer, output);

        // Post-process results
        return postprocess(output[0], bitmap.getWidth(), bitmap.getHeight());
    }

    private ByteBuffer preprocessImage(Bitmap bitmap) {
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3);
        buffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[INPUT_SIZE * INPUT_SIZE];
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        for (int pixel : pixels) {
            // Normalize to [0, 1]
            buffer.putFloat(((pixel >> 16) & 0xFF) / 255.0f); // R
            buffer.putFloat(((pixel >> 8) & 0xFF) / 255.0f);  // G
            buffer.putFloat((pixel & 0xFF) / 255.0f);         // B
        }

        return buffer;
    }

    private List<Detection> postprocess(float[][] outputs, int originalWidth, int originalHeight) {
        List<Detection> detections = new ArrayList<>();
        float confThreshold = 0.5f;

        for (float[] output : outputs) {
            float confidence = output[4];
            if (confidence < confThreshold) continue;

            // Get class with highest score
            int classId = 0;
            float maxClassScore = 0;
            for (int i = 5; i < output.length; i++) {
                if (output[i] > maxClassScore) {
                    maxClassScore = output[i];
                    classId = i - 5;
                }
            }

            float finalScore = confidence * maxClassScore;
            if (finalScore < confThreshold) continue;

            // Convert coordinates
            float cx = output[0] * originalWidth / INPUT_SIZE;
            float cy = output[1] * originalHeight / INPUT_SIZE;
            float w = output[2] * originalWidth / INPUT_SIZE;
            float h = output[3] * originalHeight / INPUT_SIZE;

            float x1 = cx - w / 2;
            float y1 = cy - h / 2;
            float x2 = cx + w / 2;
            float y2 = cy + h / 2;

            String label = classId < CLASS_NAMES.length ? 
                          CLASS_NAMES[classId] : "অজানা চিহ্ন";
            
            detections.add(new Detection(x1, y1, x2, y2, finalScore, classId, label));
        }

        return detections;
    }

    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
        // GpuDelegate is not initialized, so no need to close
    }

    public static class Detection {
        public float x1, y1, x2, y2;
        public float confidence;
        public int classId;
        public String label;

        public Detection(float x1, float y1, float x2, float y2, 
                        float confidence, int classId, String label) {
            this.x1 = x1;
            this.y1 = y1;
            this.x2 = x2;
            this.y2 = y2;
            this.confidence = confidence;
            this.classId = classId;
            this.label = label;
        }
    }
}
