package com.trafficapp;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.speech.tts.TextToSpeech;
import android.util.AttributeSet;
import android.view.View;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class DetectionOverlayView extends View {
    private List<TrafficSignDetector.Detection> detections = new ArrayList<>();
    private Paint boxPaint;
    private Paint textPaint;
    private Paint textBackgroundPaint;
    private TextToSpeech tts;
    private String lastSpokenSign = "";
    private long lastSpeakTime = 0;
    private static final long SPEAK_INTERVAL_MS = 3000; // Speak every 3 seconds
    
    public DetectionOverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        init(context);
    }
    
    private void init(Context context) {
        // Box paint
        boxPaint = new Paint();
        boxPaint.setColor(Color.parseColor("#00FF00")); // Green
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(5f);
        
        // Text paint
        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(40f);
        textPaint.setAntiAlias(true);
        textPaint.setStyle(Paint.Style.FILL);
        
        // Text background paint
        textBackgroundPaint = new Paint();
        textBackgroundPaint.setColor(Color.parseColor("#AA006A4E")); // Semi-transparent green
        textBackgroundPaint.setStyle(Paint.Style.FILL);
        
        // Initialize TTS
        tts = new TextToSpeech(context, status -> {
            if (status == TextToSpeech.SUCCESS) {
                tts.setLanguage(new Locale("bn", "BD"));
            }
        });
    }
    
    public void setDetections(List<TrafficSignDetector.Detection> detections) {
        this.detections = detections;
        
        // Speak the detected sign
        if (!detections.isEmpty()) {
            TrafficSignDetector.Detection bestDetection = detections.get(0);
            speakDetection(bestDetection);
        }
    }
    
    private void speakDetection(TrafficSignDetector.Detection detection) {
        long currentTime = System.currentTimeMillis();
        
        // Only speak if it's a different sign or enough time has passed
        if (!detection.label.equals(lastSpokenSign) || 
            currentTime - lastSpeakTime > SPEAK_INTERVAL_MS) {
            
            if (tts != null) {
                tts.speak(detection.label, TextToSpeech.QUEUE_FLUSH, null, null);
                lastSpokenSign = detection.label;
                lastSpeakTime = currentTime;
            }
        }
    }
    
    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        
        if (detections == null || detections.isEmpty()) {
            return;
        }
        
        float scaleX = (float) getWidth() / 640f;  // Assume 640 width from camera
        float scaleY = (float) getHeight() / 480f; // Assume 480 height from camera
        
        for (TrafficSignDetector.Detection detection : detections) {
            // Scale coordinates to view size
            float left = detection.x1 * scaleX;
            float top = detection.y1 * scaleY;
            float right = detection.x2 * scaleX;
            float bottom = detection.y2 * scaleY;
            
            // Draw bounding box
            RectF rect = new RectF(left, top, right, bottom);
            
            // Change color based on confidence
            if (detection.confidence > 0.8) {
                boxPaint.setColor(Color.parseColor("#00FF00")); // Green - high confidence
            } else if (detection.confidence > 0.6) {
                boxPaint.setColor(Color.parseColor("#FFFF00")); // Yellow - medium
            } else {
                boxPaint.setColor(Color.parseColor("#FF6600")); // Orange - low
            }
            
            canvas.drawRect(rect, boxPaint);
            
            // Draw label with background
            String label = detection.label + " " + 
                          String.format("%.0f%%", detection.confidence * 100);
            
            float textWidth = textPaint.measureText(label);
            float textHeight = textPaint.getTextSize();
            
            // Background rectangle for text
            RectF textBackground = new RectF(
                left, 
                top - textHeight - 10, 
                left + textWidth + 10, 
                top
            );
            canvas.drawRect(textBackground, textBackgroundPaint);
            
            // Draw text
            canvas.drawText(label, left + 5, top - 10, textPaint);
        }
    }
    
    public void cleanup() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
    }
}
