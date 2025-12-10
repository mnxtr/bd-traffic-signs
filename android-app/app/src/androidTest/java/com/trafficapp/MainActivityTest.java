package com.trafficapp;

import android.content.Context;
import android.content.Intent;
import android.view.View;

import androidx.test.core.app.ActivityScenario;
import androidx.test.core.app.ApplicationProvider;
import androidx.test.espresso.Espresso;
import androidx.test.espresso.action.ViewActions;
import androidx.test.espresso.assertion.ViewAssertions;
import androidx.test.espresso.matcher.ViewMatchers;
import androidx.test.ext.junit.rules.ActivityScenarioRule;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.filters.LargeTest;

import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;

import static androidx.test.espresso.Espresso.onView;
import static androidx.test.espresso.action.ViewActions.click;
import static androidx.test.espresso.assertion.ViewAssertions.matches;
import static androidx.test.espresso.matcher.ViewMatchers.isDisplayed;
import static androidx.test.espresso.matcher.ViewMatchers.withId;
import static androidx.test.espresso.matcher.ViewMatchers.withText;
import static org.hamcrest.Matchers.not;

/**
 * Espresso UI tests for MainActivity
 */
@RunWith(AndroidJUnit4.class)
@LargeTest
public class MainActivityTest {

    @Rule
    public ActivityScenarioRule<MainActivity> activityRule =
            new ActivityScenarioRule<>(MainActivity.class);

    @Test
    public void testToolbarIsDisplayed() {
        onView(withId(R.id.toolbar))
                .check(matches(isDisplayed()));
    }

    @Test
    public void testCaptureButtonIsDisplayed() {
        onView(withId(R.id.captureButton))
                .check(matches(isDisplayed()));
    }

    @Test
    public void testGalleryButtonIsDisplayed() {
        onView(withId(R.id.galleryButton))
                .check(matches(isDisplayed()));
    }

    @Test
    public void testLiveButtonIsDisplayed() {
        onView(withId(R.id.liveButton))
                .check(matches(isDisplayed()));
    }

    @Test
    public void testDetectFabIsDisplayed() {
        onView(withId(R.id.detectFab))
                .check(matches(isDisplayed()));
    }

    @Test
    public void testResultCardInitiallyHidden() {
        onView(withId(R.id.resultCard))
                .check(matches(not(isDisplayed())));
    }

    @Test
    public void testPlaceholderIconIsDisplayed() {
        onView(withId(R.id.placeholderIcon))
                .check(matches(isDisplayed()));
    }

    @Test
    public void testStatsTextViewsDisplayed() {
        onView(withId(R.id.totalScansTextView))
                .check(matches(isDisplayed()));
        onView(withId(R.id.detectedSignsTextView))
                .check(matches(isDisplayed()));
    }

    @Test
    public void testCameraPreviewDisplayed() {
        onView(withId(R.id.cameraPreview))
                .check(matches(isDisplayed()));
    }

    @Test
    public void testClickCaptureButton() {
        // Just verify the button is clickable (camera intent will be triggered)
        onView(withId(R.id.captureButton))
                .check(matches(isDisplayed()))
                .check(matches(ViewMatchers.isClickable()));
    }

    @Test
    public void testClickGalleryButton() {
        // Verify the button is clickable
        onView(withId(R.id.galleryButton))
                .check(matches(isDisplayed()))
                .check(matches(ViewMatchers.isClickable()));
    }

    @Test
    public void testClickLiveButton() {
        // Verify the button is clickable
        onView(withId(R.id.liveButton))
                .check(matches(isDisplayed()))
                .check(matches(ViewMatchers.isClickable()));
    }
}
