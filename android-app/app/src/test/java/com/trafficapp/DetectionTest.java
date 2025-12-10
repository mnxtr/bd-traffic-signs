package com.trafficapp;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Unit tests for TrafficSignDetector.Detection class
 */
public class DetectionTest {

    @Test
    public void testDetectionCreation() {
        TrafficSignDetector.Detection detection = new TrafficSignDetector.Detection(
            10.0f, 20.0f, 100.0f, 150.0f, 0.95f, 0, "থামুন"
        );
        
        assertEquals(10.0f, detection.x1, 0.001f);
        assertEquals(20.0f, detection.y1, 0.001f);
        assertEquals(100.0f, detection.x2, 0.001f);
        assertEquals(150.0f, detection.y2, 0.001f);
        assertEquals(0.95f, detection.confidence, 0.001f);
        assertEquals(0, detection.classId);
        assertEquals("থামুন", detection.label);
    }

    @Test
    public void testDetectionBoundingBox() {
        TrafficSignDetector.Detection detection = new TrafficSignDetector.Detection(
            0.0f, 0.0f, 50.0f, 50.0f, 0.8f, 1, "গতি সীমা"
        );
        
        float width = detection.x2 - detection.x1;
        float height = detection.y2 - detection.y1;
        
        assertEquals(50.0f, width, 0.001f);
        assertEquals(50.0f, height, 0.001f);
    }

    @Test
    public void testDetectionHighConfidence() {
        TrafficSignDetector.Detection detection = new TrafficSignDetector.Detection(
            10.0f, 10.0f, 60.0f, 60.0f, 0.99f, 2, "নো এন্ট্রি"
        );
        
        assertTrue(detection.confidence > 0.9f);
    }

    @Test
    public void testDetectionLowConfidence() {
        TrafficSignDetector.Detection detection = new TrafficSignDetector.Detection(
            10.0f, 10.0f, 60.0f, 60.0f, 0.55f, 3, "বাঁ দিকে যান"
        );
        
        assertTrue(detection.confidence >= 0.5f);
        assertTrue(detection.confidence < 0.6f);
    }

    @Test
    public void testAllClassIds() {
        String[] classNames = {"থামুন", "গতি সীমা", "নো এন্ট্রি", "বাঁ দিকে যান", "ডান দিকে যান"};
        
        for (int i = 0; i < classNames.length; i++) {
            TrafficSignDetector.Detection detection = new TrafficSignDetector.Detection(
                0, 0, 100, 100, 0.9f, i, classNames[i]
            );
            assertEquals(i, detection.classId);
            assertEquals(classNames[i], detection.label);
        }
    }

    @Test
    public void testDetectionWithNegativeCoordinates() {
        // Edge case: coordinates could be negative after transformation
        TrafficSignDetector.Detection detection = new TrafficSignDetector.Detection(
            -5.0f, -10.0f, 50.0f, 50.0f, 0.7f, 0, "থামুন"
        );
        
        assertEquals(-5.0f, detection.x1, 0.001f);
        assertEquals(-10.0f, detection.y1, 0.001f);
    }

    @Test
    public void testDetectionCenter() {
        TrafficSignDetector.Detection detection = new TrafficSignDetector.Detection(
            100.0f, 200.0f, 200.0f, 300.0f, 0.85f, 4, "ডান দিকে যান"
        );
        
        float centerX = (detection.x1 + detection.x2) / 2;
        float centerY = (detection.y1 + detection.y2) / 2;
        
        assertEquals(150.0f, centerX, 0.001f);
        assertEquals(250.0f, centerY, 0.001f);
    }
}
