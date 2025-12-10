package com.trafficapp;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Size;
import android.widget.TextView;
import android.widget.Toast;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.google.common.util.concurrent.ListenableFuture;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class CameraActivity extends AppCompatActivity {
    private static final int CAMERA_PERMISSION_CODE = 100;
    private static final int DETECTION_INTERVAL_MS = 500; // Detect every 500ms
    
    private PreviewView previewView;
    private DetectionOverlayView overlayView;
    private TextView fpsTextView;
    private TrafficSignDetector detector;
    private ExecutorService cameraExecutor;
    private ProcessCameraProvider cameraProvider;
    
    private long lastDetectionTime = 0;
    private boolean isDetecting = false;
    
    // FPS tracking
    private long frameCount = 0;
    private long lastFpsTime = 0;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);
        
        previewView = findViewById(R.id.previewView);
        overlayView = findViewById(R.id.overlayView);
        fpsTextView = findViewById(R.id.fpsTextView);
        
        // Back button listener
        findViewById(R.id.backButton).setOnClickListener(v -> finish());
        
        cameraExecutor = Executors.newSingleThreadExecutor();
        
        // Initialize detector
        try {
            detector = new TrafficSignDetector(this);
        } catch (Exception e) {
            Toast.makeText(this, "ডিটেক্টর লোড ব্যর্থ", Toast.LENGTH_SHORT).show();
            finish();
            return;
        }
        
        if (checkPermissions()) {
            startCamera();
        } else {
            requestPermissions();
        }
    }
    
    private boolean checkPermissions() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
                == PackageManager.PERMISSION_GRANTED;
    }
    
    private void requestPermissions() {
        ActivityCompat.requestPermissions(this, 
            new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
    }
    
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, 
                                          @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION_CODE) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                startCamera();
            } else {
                Toast.makeText(this, "ক্যামেরা অনুমতি প্রয়োজন", Toast.LENGTH_SHORT).show();
                finish();
            }
        }
    }
    
    private void startCamera() {
        ListenableFuture<ProcessCameraProvider> cameraProviderFuture = 
            ProcessCameraProvider.getInstance(this);
        
        cameraProviderFuture.addListener(() -> {
            try {
                cameraProvider = cameraProviderFuture.get();
                bindCameraUseCases();
            } catch (Exception e) {
                Toast.makeText(this, "ক্যামেরা শুরু করতে ব্যর্থ", Toast.LENGTH_SHORT).show();
            }
        }, ContextCompat.getMainExecutor(this));
    }
    
    private void bindCameraUseCases() {
        // Preview use case
        Preview preview = new Preview.Builder()
            .build();
        
        preview.setSurfaceProvider(previewView.getSurfaceProvider());
        
        // Image analysis use case for detection
        ImageAnalysis imageAnalysis = new ImageAnalysis.Builder()
            .setTargetResolution(new Size(640, 480))
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build();
        
        imageAnalysis.setAnalyzer(cameraExecutor, this::analyzeImage);
        
        // Camera selector
        CameraSelector cameraSelector = new CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build();
        
        // Bind to lifecycle
        try {
            cameraProvider.unbindAll();
            cameraProvider.bindToLifecycle(
                this, 
                cameraSelector, 
                preview, 
                imageAnalysis
            );
        } catch (Exception e) {
            Toast.makeText(this, "ক্যামেরা বাইন্ড ব্যর্থ", Toast.LENGTH_SHORT).show();
        }
    }
    
    private void analyzeImage(ImageProxy image) {
        // Throttle detection to avoid overwhelming the system
        long currentTime = System.currentTimeMillis();
        if (isDetecting || currentTime - lastDetectionTime < DETECTION_INTERVAL_MS) {
            image.close();
            return;
        }
        
        isDetecting = true;
        lastDetectionTime = currentTime;
        
        // Convert ImageProxy to Bitmap
        Bitmap bitmap = imageProxyToBitmap(image);
        
        if (bitmap != null && detector != null) {
            // Run detection
            List<TrafficSignDetector.Detection> detections = detector.detect(bitmap);
            
            // Update FPS counter
            frameCount++;
            if (currentTime - lastFpsTime >= 1000) {
                final long fps = frameCount;
                runOnUiThread(() -> fpsTextView.setText("FPS: " + fps));
                frameCount = 0;
                lastFpsTime = currentTime;
            }
            
            // Update overlay on UI thread
            runOnUiThread(() -> {
                overlayView.setDetections(detections);
                overlayView.invalidate();
            });
        }
        
        isDetecting = false;
        image.close();
    }
    
    private Bitmap imageProxyToBitmap(ImageProxy image) {
        try {
            ImageProxy.PlaneProxy[] planes = image.getPlanes();
            ByteBuffer yBuffer = planes[0].getBuffer();
            ByteBuffer uBuffer = planes[1].getBuffer();
            ByteBuffer vBuffer = planes[2].getBuffer();
            
            int ySize = yBuffer.remaining();
            int uSize = uBuffer.remaining();
            int vSize = vBuffer.remaining();
            
            byte[] nv21 = new byte[ySize + uSize + vSize];
            
            yBuffer.get(nv21, 0, ySize);
            vBuffer.get(nv21, ySize, vSize);
            uBuffer.get(nv21, ySize + vSize, uSize);
            
            // Convert to bitmap
            android.graphics.YuvImage yuvImage = new android.graphics.YuvImage(
                nv21, 
                android.graphics.ImageFormat.NV21, 
                image.getWidth(), 
                image.getHeight(), 
                null
            );
            
            try (ByteArrayOutputStream out = new ByteArrayOutputStream()) {
                yuvImage.compressToJpeg(
                    new android.graphics.Rect(0, 0, image.getWidth(), image.getHeight()), 
                    100, 
                    out
                );
                
                byte[] imageBytes = out.toByteArray();
                Bitmap bitmap = android.graphics.BitmapFactory.decodeByteArray(
                    imageBytes, 0, imageBytes.length
                );
                
                // Rotate bitmap based on image rotation
                return rotateBitmap(bitmap, image.getImageInfo().getRotationDegrees());
            }
            
        } catch (IOException e) {
            return null;
        }
    }
    
    private Bitmap rotateBitmap(Bitmap bitmap, int degrees) {
        if (degrees == 0) return bitmap;
        
        android.graphics.Matrix matrix = new android.graphics.Matrix();
        matrix.postRotate(degrees);
        
        return Bitmap.createBitmap(
            bitmap, 0, 0, 
            bitmap.getWidth(), bitmap.getHeight(), 
            matrix, true
        );
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (overlayView != null) {
            overlayView.cleanup();
        }
        if (cameraExecutor != null) {
            cameraExecutor.shutdown();
        }
        if (detector != null) {
            detector.close();
        }
    }
}
