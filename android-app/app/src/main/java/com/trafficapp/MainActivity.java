package com.trafficapp;

import android.Manifest;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.speech.tts.TextToSpeech;
import android.view.View;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.google.android.material.appbar.MaterialToolbar;
import com.google.android.material.button.MaterialButton;
import com.google.android.material.card.MaterialCardView;
import com.google.android.material.floatingactionbutton.ExtendedFloatingActionButton;
import java.io.InputStream;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {
    private static final int CAMERA_PERMISSION_CODE = 100;
    private static final String PREFS_NAME = "TrafficSignStats";
    
    private TextToSpeech tts;
    private TrafficSignDetector detector;
    
    // UI Components
    private MaterialToolbar toolbar;
    private ImageView cameraPreview;
    private ImageView overlayImageView;
    private ImageView placeholderIcon;
    private ImageView signImageView;
    private ProgressBar loadingProgress;
    private TextView signNameTextView;
    private TextView signDescriptionTextView;
    private TextView confidenceTextView;
    private TextView totalScansTextView;
    private TextView detectedSignsTextView;
    private MaterialCardView resultCard;
    private MaterialButton captureButton;
    private MaterialButton galleryButton;
    private MaterialButton liveButton;
    private MaterialButton speakButton;
    private MaterialButton shareButton;
    private ExtendedFloatingActionButton detectFab;
    
    // Stats
    private int totalScans = 0;
    private int detectedSigns = 0;
    private String currentSignName = "";
    private Bitmap currentBitmap;
    
    // Activity launchers
    private ActivityResultLauncher<Intent> cameraLauncher;
    private ActivityResultLauncher<Intent> galleryLauncher;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        initializeViews();
        initializeTTS();
        initializeDetector();
        initializeActivityLaunchers();
        loadStats();
        checkPermissions();
        setupClickListeners();
    }
    
    private void initializeViews() {
        toolbar = findViewById(R.id.toolbar);
        cameraPreview = findViewById(R.id.cameraPreview);
        overlayImageView = findViewById(R.id.overlayImageView);
        placeholderIcon = findViewById(R.id.placeholderIcon);
        signImageView = findViewById(R.id.signImageView);
        loadingProgress = findViewById(R.id.loadingProgress);
        signNameTextView = findViewById(R.id.signNameTextView);
        signDescriptionTextView = findViewById(R.id.signDescriptionTextView);
        confidenceTextView = findViewById(R.id.confidenceTextView);
        totalScansTextView = findViewById(R.id.totalScansTextView);
        detectedSignsTextView = findViewById(R.id.detectedSignsTextView);
        resultCard = findViewById(R.id.resultCard);
        captureButton = findViewById(R.id.captureButton);
        galleryButton = findViewById(R.id.galleryButton);
        liveButton = findViewById(R.id.liveButton);
        speakButton = findViewById(R.id.speakButton);
        shareButton = findViewById(R.id.shareButton);
        detectFab = findViewById(R.id.detectFab);
        
        setSupportActionBar(toolbar);
        resultCard.setVisibility(View.GONE);
    }
    
    private void initializeActivityLaunchers() {
        cameraLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Bundle extras = result.getData().getExtras();
                    Bitmap imageBitmap = (Bitmap) extras.get("data");
                    if (imageBitmap != null) {
                        processImage(imageBitmap);
                    }
                }
            }
        );
        
        galleryLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    try {
                        InputStream inputStream = getContentResolver().openInputStream(imageUri);
                        Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                        processImage(bitmap);
                    } catch (Exception e) {
                        Toast.makeText(this, "ছবি লোড ব্যর্থ", Toast.LENGTH_SHORT).show();
                    }
                }
            }
        );
    }
    
    private void setupClickListeners() {
        captureButton.setOnClickListener(v -> openCamera());
        galleryButton.setOnClickListener(v -> openGallery());
        liveButton.setOnClickListener(v -> openLiveCamera());
        detectFab.setOnClickListener(v -> openLiveCamera());
        
        speakButton.setOnClickListener(v -> {
            if (!currentSignName.isEmpty()) {
                speakBengali(currentSignName);
            }
        });
        
        shareButton.setOnClickListener(v -> shareResult());
    }
    
    private void openLiveCamera() {
        Intent intent = new Intent(this, CameraActivity.class);
        startActivity(intent);
    }
    
    private void openCamera() {
        Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
            cameraLauncher.launch(takePictureIntent);
        } else {
            Toast.makeText(this, "ক্যামেরা উপলব্ধ নয়", Toast.LENGTH_SHORT).show();
        }
    }
    
    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        galleryLauncher.launch(intent);
    }
    
    private void processImage(Bitmap bitmap) {
        currentBitmap = bitmap;
        cameraPreview.setImageBitmap(bitmap);
        placeholderIcon.setVisibility(View.GONE);
        
        // Show loading
        loadingProgress.setVisibility(View.VISIBLE);
        
        // Run detection in background thread
        new Thread(() -> {
            List<TrafficSignDetector.Detection> detections = detector.detect(bitmap);
            
            runOnUiThread(() -> {
                loadingProgress.setVisibility(View.GONE);
                totalScans++;
                
                if (!detections.isEmpty()) {
                    TrafficSignDetector.Detection bestDetection = detections.get(0);
                    detectedSigns++;
                    showResult(bestDetection);
                } else {
                    Toast.makeText(this, "কোনো চিহ্ন শনাক্ত করা যায়নি", Toast.LENGTH_SHORT).show();
                }
                
                updateStats();
            });
        }).start();
    }
    
    private void showResult(TrafficSignDetector.Detection detection) {
        currentSignName = detection.label;
        signNameTextView.setText(detection.label);
        signDescriptionTextView.setText(getSignDescription(detection.classId));
        confidenceTextView.setText(String.format("%.1f%%", detection.confidence * 100));
        
        // Show the result card with animation
        resultCard.setVisibility(View.VISIBLE);
        resultCard.setAlpha(0f);
        resultCard.animate().alpha(1f).setDuration(300).start();
        
        // Automatically speak the sign name
        speakBengali(detection.label);
    }
    
    private String getSignDescription(int classId) {
        // Map class IDs to descriptions
        String[] descriptions = {
            "Stop Sign - সম্পূর্ণভাবে থামুন",
            "Speed Limit - গতি সীমা মেনে চলুন",
            "No Entry - প্রবেশ নিষেধ",
            "Turn Left - বাম দিকে ঘুরুন",
            "Turn Right - ডান দিকে ঘুরুন"
        };
        
        if (classId >= 0 && classId < descriptions.length) {
            return descriptions[classId];
        }
        return "ট্রাফিক চিহ্ন";
    }
    
    private void shareResult() {
        String shareText = "আমি " + currentSignName + " ট্রাফিক চিহ্ন শনাক্ত করেছি!\n\n" +
                          "BD Traffic Signs অ্যাপ ব্যবহার করুন।";
        
        Intent shareIntent = new Intent(Intent.ACTION_SEND);
        shareIntent.setType("text/plain");
        shareIntent.putExtra(Intent.EXTRA_TEXT, shareText);
        startActivity(Intent.createChooser(shareIntent, "শেয়ার করুন"));
    }
    
    private void checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) 
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, 
                new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_CODE);
        }
    }
    
    private void initializeDetector() {
        try {
            detector = new TrafficSignDetector(this);
            Toast.makeText(this, "ডিটেক্টর প্রস্তুত", Toast.LENGTH_SHORT).show();
        } catch (Exception e) {
            Toast.makeText(this, "ডিটেক্টর লোড ব্যর্থ", Toast.LENGTH_SHORT).show();
        }
    }

    
    private void loadStats() {
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        totalScans = prefs.getInt("totalScans", 0);
        detectedSigns = prefs.getInt("detectedSigns", 0);
        updateStats();
    }
    
    private void updateStats() {
        totalScansTextView.setText(String.valueOf(totalScans));
        detectedSignsTextView.setText(String.valueOf(detectedSigns));
        
        // Save stats
        SharedPreferences prefs = getSharedPreferences(PREFS_NAME, MODE_PRIVATE);
        SharedPreferences.Editor editor = prefs.edit();
        editor.putInt("totalScans", totalScans);
        editor.putInt("detectedSigns", detectedSigns);
        editor.apply();
    }

    private void initializeTTS() {
        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                Locale bengaliLocale = new Locale("bn", "BD");
                int result = tts.setLanguage(bengaliLocale);
                
                if (result == TextToSpeech.LANG_MISSING_DATA || 
                    result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Toast.makeText(this, "বাংলা ভাষা সমর্থিত নয়", Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    private void speakBengali(String text) {
        if (tts != null && !text.isEmpty()) {
            tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, null);
        }
    }

    @Override
    protected void onDestroy() {
        if (tts != null) {
            tts.stop();
            tts.shutdown();
        }
        if (detector != null) {
            detector.close();
        }
        super.onDestroy();
    }
}
