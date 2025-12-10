# Add project specific ProGuard rules here.
# You can control the set of applied configuration files using the
# proguardFiles setting in build.gradle.

# TensorFlow Lite
-keep class org.tensorflow.** { *; }
-keepclassmembers class org.tensorflow.** { *; }
-dontwarn org.tensorflow.**

# Keep TFLite GPU delegate
-keep class org.tensorflow.lite.gpu.** { *; }

# Keep model class
-keep class com.trafficapp.TrafficSignDetector { *; }
-keep class com.trafficapp.TrafficSignDetector$Detection { *; }

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# CameraX
-keep class androidx.camera.** { *; }

# Material Components
-keep class com.google.android.material.** { *; }

# Keep custom views
-keep class com.trafficapp.DetectionOverlayView { *; }

# Keep R class members
-keepclassmembers class **.R$* {
    public static <fields>;
}

# Preserve line number information for debugging stack traces
-keepattributes SourceFile,LineNumberTable

# Hide original source file name
-renamesourcefileattribute SourceFile
